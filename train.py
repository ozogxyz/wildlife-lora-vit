import argparse
import os

import cv2
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from pytorch_pretrained_vit import ViT
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from lora import LoRA_ViT, TARGET_PRESETS
from aug import ColorJitterCV, RandomGaussianBlur, RandomHorizontalFlip

# ---------- args ----------
p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--out", default="best.pth")
p.add_argument("--frac", type=float, default=1.0)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--fold", type=int, default=0, help="fold to run; -1 = all folds")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--rank", type=int, default=4)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--lora-targets", choices=list(TARGET_PRESETS), default="qv")
args = p.parse_args()

NUM_CLASSES = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
workers = os.cpu_count() or 2

# ---------- data ----------
os.chdir(args.data_dir)
features = pd.read_csv("train_features.csv", index_col="id")
labels = pd.read_csv("train_labels.csv", index_col="id")
species = sorted(labels.columns)

labels = labels.sample(frac=args.frac, random_state=1)
features = features.loc[labels.index]
paths = features.filepath
sites = features.site

gkf = GroupKFold(n_splits=min(args.folds, sites.nunique()))
folds = list(gkf.split(paths, labels, groups=sites))

print(
    f"device {device}  workers {workers}  frac {args.frac}  epochs {args.epochs}  "
    f"lr {args.lr}  rank {args.rank}  img {args.img_size}  lora {args.lora_targets}  "
    f"{gkf.n_splits}-fold ({sites.nunique()} sites)"
)


# ---------- dataset ----------
def make_aug():
    return Compose(
        [
            ColorJitterCV(brightness=0.8, contrast=0.1, gamma=0.2, temp=0.8, p=0.75),
            RandomGaussianBlur(),
            RandomHorizontalFlip(),
        ]
    )


class Images(Dataset):
    def __init__(self, paths, labels=None, train=False):
        self.paths = paths
        self.labels = labels
        self.aug = make_aug() if train else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths.iloc[i])
        img = cv2.resize(img, (args.img_size, args.img_size))
        if self.aug:
            img = self.aug({"image": img})["image"]
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # .copy(): aug may return a flipped view
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = (img - 0.5) / 0.5
        sample = {"id": self.paths.index[i], "image": img}
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels.iloc[i].values, dtype=torch.float)
        return sample


# ---------- model / eval helpers ----------
def make_model():
    backbone = ViT("B_16", pretrained=True, image_size=args.img_size)
    targets = TARGET_PRESETS[args.lora_targets]
    return LoRA_ViT(backbone, r=args.rank, num_classes=NUM_CLASSES, targets=targets).to(device)


def predict(model, loader):
    """Run the model over a loader, return a DataFrame of raw logits indexed by image id."""
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).cpu()
            parts.append(pd.DataFrame(logits.tolist(), index=batch["id"], columns=species))
    return pd.concat(parts)


def logloss(logits, truth, temp=1.0):
    """log loss of softmax(logits / temp) against the true species per row."""
    probs = torch.softmax(torch.tensor(logits.values) / temp, dim=1)
    probs = pd.DataFrame(probs.tolist(), index=logits.index, columns=species)
    return log_loss(truth, probs, labels=species)


def best_temp(logits, truth):
    """Dumb grid search for the temperature that minimizes log loss."""
    grid = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
    scores = [(logloss(logits, truth, t), t) for t in grid]
    return min(scores)  # (best_logloss, best_temp)


def report(logits, truth):
    """Per-class accuracy and true-vs-predicted counts (spot class collapse)."""
    pred = logits.idxmax(axis=1)
    for sp in species:
        is_sp = truth == sp
        n = int(is_sp.sum())
        acc = (pred[is_sp] == sp).mean() if n else 0.0
        print(f"  {sp:18s} acc {acc:.3f}  true {n:4d}  pred {int((pred == sp).sum()):4d}")


# ---------- one fold ----------
def run_fold(k, train_idx, eval_idx, save):
    train_dl = DataLoader(
        Images(paths.iloc[train_idx], labels.iloc[train_idx], train=True),
        batch_size=args.batch, shuffle=True, num_workers=workers, pin_memory=True,
    )
    eval_labels = labels.iloc[eval_idx]
    eval_dl = DataLoader(
        Images(paths.iloc[eval_idx], eval_labels),
        batch_size=args.batch, num_workers=workers, pin_memory=True,
    )
    truth = eval_labels.idxmax(axis=1)

    model = make_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        [w for w in model.parameters() if w.requires_grad], lr=args.lr, weight_decay=1e-5
    )

    best_ll = float("inf")
    best_logits = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in tqdm(train_dl):
            optimizer.zero_grad()
            loss = criterion(model(batch["image"].to(device)), batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        logits = predict(model, eval_dl).loc[eval_labels.index]
        ll = logloss(logits, truth)
        print(f"fold {k}  epoch {epoch:2d}  train_loss {total / len(train_dl):.4f}  eval_logloss {ll:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_logits = logits
            if save:
                torch.save(model.state_dict(), args.out)

    cal_ll, temp = best_temp(best_logits, truth)
    print(f"fold {k}  best {best_ll:.4f}  temp {temp}  calibrated {cal_ll:.4f}")
    report(best_logits, truth)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_ll


# ---------- run ----------
chosen = range(len(folds)) if args.fold < 0 else [args.fold]
scores = [run_fold(k, folds[k][0], folds[k][1], save=len(chosen) == 1) for k in chosen]
if len(scores) > 1:
    print(f"CV log loss {sum(scores) / len(scores):.4f}  folds {[round(s, 3) for s in scores]}")
