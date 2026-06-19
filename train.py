import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.transforms import Compose
from pytorch_pretrained_vit import ViT
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar

from lora import LoRA_ViT
from aug import ColorJitterCV, RandomGaussianBlur, RandomHorizontalFlip

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--out", default="best.pth")
p.add_argument("--rank", type=int, default=4)
p.add_argument("--frac", type=float, default=1.0)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--fold", type=int, default=0, help="fold index to run; -1 = all folds (CV), report mean+/-std")
p.add_argument("--img-size", type=int, default=0, help="ViT input size (interpolates pos emb); 0 = pretrained default (384)")
args = p.parse_args()

DATA_DIR = args.data_dir
OUT = args.out
RANK = args.rank
NUM_CLASSES = 8
FRAC = args.frac
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.1
SEED = 1
AUGMENT = True
NORM_MEAN = 0.5
NORM_STD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count() or 2

def make_backbone():
    if args.img_size:
        return ViT("B_16", pretrained=True, image_size=args.img_size)
    return ViT("B_16", pretrained=True)


backbone = make_backbone()
IMG_SIZE = backbone.image_size
IMG_SIZE = IMG_SIZE[0] if isinstance(IMG_SIZE, (tuple, list)) else IMG_SIZE

os.chdir(DATA_DIR)
train_features = pd.read_csv("train_features.csv", index_col="id")
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())

y = train_labels.sample(frac=FRAC, random_state=SEED)
x = train_features.loc[y.index].filepath.to_frame()
sites = train_features.loc[y.index, "site"]

n_folds = min(args.folds, sites.nunique())
splits = list(GroupKFold(n_splits=n_folds).split(x, y, groups=sites))
print(
    f"device: {device} ({gpu}), batch {BATCH_SIZE}, workers {NUM_WORKERS}, "
    f"frac {FRAC}, epochs {EPOCHS}, lr {LR}, rank {RANK}, wd {WEIGHT_DECAY}, img {IMG_SIZE}, "
    f"{n_folds}-fold groupkfold ({sites.nunique()} sites)"
)


def augment(sample):
    return Compose(
        [
            ColorJitterCV(brightness=0.8, contrast=0.1, gamma=0.2, temp=0.8, p=0.75),
            RandomGaussianBlur(),
            RandomHorizontalFlip(),
        ]
    )(sample)


class ImagesDataset(Dataset):
    def __init__(self, x_df, y_df=None, mode="eval"):
        self.data = x_df
        self.label = y_df
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data.iloc[idx]["filepath"])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sample = {"image": img}
        if self.mode == "train":
            sample = augment(sample)
        img = cv2.cvtColor(np.ascontiguousarray(sample["image"]), cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).float() / 255.0
        img = (img - NORM_MEAN) / NORM_STD
        out = {"image_id": self.data.index[idx], "image": img}
        if self.label is not None:
            out["label"] = torch.tensor(self.label.iloc[idx].values, dtype=torch.float)
        return out


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logloss_at_T(logits, y_true, T):
    return log_loss(y_true, softmax_np(logits / T), labels=species_labels)


def fit_temperature(logits, y_true):
    """One scalar T that minimizes eval log loss (post-hoc calibration)."""
    res = minimize_scalar(lambda T: logloss_at_T(logits, y_true, T), bounds=(0.5, 5.0), method="bounded")
    return float(res.x)


def run_fold(fold, train_idx, eval_idx, bb, save):
    x_tr, x_ev = x.iloc[train_idx], x.iloc[eval_idx]
    y_tr, y_ev = y.iloc[train_idx], y.iloc[eval_idx]
    y_true = y_ev.idxmax(axis=1).values

    train_dl = DataLoader(
        ImagesDataset(x_tr, y_tr, mode="train" if AUGMENT else "eval"),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    eval_dl = DataLoader(
        ImagesDataset(x_ev, y_ev, mode="eval"),
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = LoRA_ViT(bb, r=RANK, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    def eval_logits():
        model.eval()
        rows, ids = [], []
        with torch.no_grad():
            for batch in eval_dl:
                rows.append(model(batch["image"].to(device)).cpu().numpy())
                ids.extend(batch["image_id"])
        return pd.DataFrame(np.concatenate(rows), index=ids, columns=species_labels).loc[y_ev.index].values

    best_ll, best_logits = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_dl, total=len(train_dl)):
            optimizer.zero_grad()
            loss = criterion(model(batch["image"].to(device)), batch["label"].to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        logits = eval_logits()
        ll = logloss_at_T(logits, y_true, 1.0)
        print(f"fold {fold}  epoch {epoch:2d}  train_loss {running / len(train_dl):.4f}  eval_logloss {ll:.4f}")
        if ll < best_ll:
            best_ll, best_logits = ll, logits
            if save:
                torch.save(model.state_dict(), OUT)

    T = fit_temperature(best_logits, y_true)
    cal_ll = logloss_at_T(best_logits, y_true, T)
    print(f"fold {fold}  best raw {best_ll:.4f}  T {T:.3f}  calibrated {cal_ll:.4f}")

    if save:
        pred = pd.Series(np.array(species_labels)[best_logits.argmax(1)], index=y_ev.index)
        true = y_ev.idxmax(axis=1)
        print(f"fold {fold}  acc {(pred == true).mean():.4f}")
        for sp in species_labels:
            m = true == sp
            if m.sum():
                print(f"  {sp:18s} {(pred[m] == true[m]).mean():.3f}  (n={m.sum()})")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_ll, cal_ll, T


if args.fold >= 0:
    assert args.fold < n_folds, f"--fold {args.fold} out of range (n_folds={n_folds})"
    run_fold(args.fold, *splits[args.fold], backbone, save=True)
else:
    raws, cals, Ts = [], [], []
    for fold, (tr, ev) in enumerate(splits):
        bb = backbone if fold == 0 else make_backbone()
        raw, cal, T = run_fold(fold, tr, ev, bb, save=False)
        raws.append(raw)
        cals.append(cal)
        Ts.append(T)
    raws, cals = np.array(raws), np.array(cals)
    print(f"CV raw log loss: {raws.mean():.4f} +/- {raws.std():.4f}  ({', '.join(f'{v:.3f}' for v in raws)})")
    print(f"CV calibrated:   {cals.mean():.4f} +/- {cals.std():.4f}  ({', '.join(f'{v:.3f}' for v in cals)})  T={', '.join(f'{t:.2f}' for t in Ts)}")
