import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from scipy.special import softmax
from pytorch_pretrained_vit import ViT
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from lora import LoRA_ViT, TARGET_PRESETS
from aug import ColorJitterCV, RandomGaussianBlur, RandomHorizontalFlip

# ---------- knobs I change between runs ----------
p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=0, help="fold to run; -1 = all folds")
p.add_argument("--lora-targets", choices=list(TARGET_PRESETS), default="qv")
p.add_argument("--rank", type=int, default=4)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--frac", type=float, default=1.0, help="data fraction, for quick smoke runs")
args = p.parse_args()

# ---------- fixed for this competition ----------
DATA_DIR = "data"
OUT = "best.pth"
IMG_SIZE = 224  # B_16 is native 224
BATCH = 32
FOLDS = 5
NUM_CLASSES = 8
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
workers = os.cpu_count() or 2

# ---------- data ----------
os.chdir(DATA_DIR)
features = pd.read_csv("train_features.csv", index_col="id")
labels = pd.read_csv("train_labels.csv", index_col="id")
species = sorted(labels.columns)

labels = labels.sample(frac=args.frac, random_state=1)
features = features.loc[labels.index]
paths = features.filepath
sites = features.site

n_folds = min(FOLDS, sites.nunique())
folds = list(GroupKFold(n_splits=n_folds).split(paths, labels, groups=sites))

print(
    f"device {device}  workers {workers}  frac {args.frac}  epochs {args.epochs}  "
    f"lr {args.lr}  rank {args.rank}  lora {args.lora_targets}  {n_folds}-fold ({sites.nunique()} sites)"
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
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if self.aug:
            img = self.aug({"image": img})["image"]
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # .copy(): aug may return a flipped view
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = (img - 0.5) / 0.5
        sample = {"image": img}
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels.iloc[i].values, dtype=torch.float)
        return sample


# ---------- model + eval ----------
def make_model():
    backbone = ViT("B_16", pretrained=True, image_size=IMG_SIZE)
    return LoRA_ViT(backbone, r=args.rank, num_classes=NUM_CLASSES, targets=TARGET_PRESETS[args.lora_targets]).to(device)


def predict(model, loader):
    """Raw logits for the whole loader as one [N, 8] numpy array, in dataset order."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            chunks.append(model(batch["image"].to(device)).cpu())
    return torch.cat(chunks).numpy()  # stack all batches, hand back a plain numpy array


def logloss(logits, truth, temp=1.0):
    # softmax(..., axis=1): row-wise -> each row sums to 1. truth = int class index per row.
    return log_loss(truth, softmax(logits / temp, axis=1), labels=list(range(NUM_CLASSES)))


def best_temp(logits, truth):
    grid = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
    return min((logloss(logits, truth, t), t) for t in grid)  # (best_logloss, best_temp)


def report(logits, truth):
    """Per-class accuracy + true-vs-pred counts (spot class collapse)."""
    pred = logits.argmax(axis=1)  # predicted class index per row
    for i, sp in enumerate(species):
        mask = truth == i  # boolean array: which rows really are class i
        true_n = int(mask.sum())  # True counts as 1
        acc = (pred[mask] == i).mean() if true_n else 0.0  # of those rows, fraction predicted right
        print(f"  {sp:18s} acc {acc:.3f}  true {true_n:4d}  pred {int((pred == i).sum()):4d}")


# ---------- one fold ----------
def run_fold(k, train_idx, eval_idx, save):
    train_dl = DataLoader(
        Images(paths.iloc[train_idx], labels.iloc[train_idx], train=True),
        batch_size=BATCH, shuffle=True, num_workers=workers, pin_memory=True,
    )
    eval_labels = labels.iloc[eval_idx]
    eval_dl = DataLoader(  # no shuffle, so logits come back in eval_labels order
        Images(paths.iloc[eval_idx], eval_labels),
        batch_size=BATCH, num_workers=workers, pin_memory=True,
    )
    truth = eval_labels[species].values.argmax(axis=1)  # column index of the 1.0 = true class

    model = make_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        [w for w in model.parameters() if w.requires_grad], lr=args.lr, weight_decay=WEIGHT_DECAY
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
        logits = predict(model, eval_dl)
        ll = logloss(logits, truth)
        print(f"fold {k}  epoch {epoch:2d}  train_loss {total / len(train_dl):.4f}  eval_logloss {ll:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_logits = logits
            if save:
                torch.save(model.state_dict(), OUT)

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
