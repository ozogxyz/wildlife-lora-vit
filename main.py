import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss

from aug import ColorJitterCV, RandomGaussianBlur, RandomHorizontalFlip

# ============================================================
#  The two parts that make this solution different:
#    (1) StratifiedGroupKFold — validate on UNSEEN, class-balanced site
#        folds. Test sites are disjoint from train; a random split would
#        leak per-site backgrounds and give a dishonest score.
#    (2) aug.py — custom cross-domain augmentations (train only).
#  Backbone: frozen DINOv2 ViT-B/14 (strong domain-robust self-supervised
#  features) + a linear head. Everything else is textbook transfer learning.
# ============================================================

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=0, help="which site-fold (0..4) to hold out as validation")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--frac", type=float, default=1.0, help="data fraction, for quick smoke runs")
p.add_argument("--img-size", type=int, default=224, help="input res; DINOv2 patch14 wants a multiple of 14 (224/392/518)")
args = p.parse_args()

BACKBONE = "vit_base_patch14_dinov2.lvd142m"
IMG_SIZE = args.img_size  # patch14: 224->16x16, 392->28x28, 518->37x37 (pos-emb interpolated)
BATCH = 32
FOLDS = 5
NUM_CLASSES = 8
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-5
MEAN = [0.485, 0.456, 0.406]  # DINOv2 uses ImageNet normalization (NOT 0.5/0.5)
STD = [0.229, 0.224, 0.225]

device = "cuda" if torch.cuda.is_available() else "cpu"
workers = os.cpu_count() or 2

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TAG = f"dino_lr{args.lr}_img{IMG_SIZE}_f{args.fold}"
if args.frac < 1.0:
    TAG += f"_frac{args.frac}"
REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "assets", f"best_{TAG}.pth")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---------- data ----------
os.chdir(os.path.join(REPO, "data"))
features = pd.read_csv("train_features.csv", index_col="id")
labels = pd.read_csv("train_labels.csv", index_col="id")
species = sorted(labels.columns)

labels = labels.sample(frac=args.frac, random_state=1)
features = features.loc[labels.index]
paths = features.filepath
sites = features.site

# (1) site-grouped + class-stratified split — sites disjoint, class mix balanced
n_folds = min(FOLDS, sites.nunique())
y = labels[species].values.argmax(1)
train_idx, val_idx = list(StratifiedGroupKFold(n_splits=n_folds).split(paths, y, groups=sites))[args.fold]

print(
    f"device {device}  workers {workers}  frac {args.frac}  epochs {args.epochs}  lr {args.lr}  "
    f"backbone dinov2-b14  fold {args.fold}/{n_folds} ({sites.nunique()} sites)"
)


# ---------- dataset ----------
def make_aug():  # (2) thesis augmentations — train only
    return Compose(
        [
            ColorJitterCV(brightness=0.8, contrast=0.1, gamma=0.2, temp=0.8, p=0.75),
            RandomGaussianBlur(),
            RandomHorizontalFlip(),
        ]
    )


mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)


class Images(Dataset):
    def __init__(self, paths, labels=None, train=False):
        self.paths = paths
        self.labels = labels
        self.aug = make_aug() if train else None  # val/test: clean preprocessing, no random aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths.iloc[i])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if self.aug:
            img = self.aug({"image": img})["image"]
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # .copy(): aug may return a flipped view
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = (img - mean) / std
        sample = {"image": img}
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels.iloc[i].values, dtype=torch.float)
        return sample


train_dl = DataLoader(
    Images(paths.iloc[train_idx], labels.iloc[train_idx], train=True),
    batch_size=BATCH, shuffle=True, num_workers=workers, pin_memory=True,
)
val_labels = labels.iloc[val_idx]
val_dl = DataLoader(  # no shuffle: logits come back in val_labels order
    Images(paths.iloc[val_idx], val_labels),
    batch_size=BATCH, num_workers=workers, pin_memory=True,
)
truth = val_labels[species].values.argmax(1)  # true class index (0..7) per val row

# ---------- model: frozen DINOv2 + linear head ----------
backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=IMG_SIZE)
for w in backbone.parameters():
    w.requires_grad = False
model = nn.Sequential(backbone, nn.Linear(backbone.num_features, NUM_CLASSES)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW([w for w in model.parameters() if w.requires_grad], lr=args.lr, weight_decay=WEIGHT_DECAY)


# ---------- standard train / eval ----------
def train_one_epoch():
    model.train()
    total = 0.0
    for batch in tqdm(train_dl):
        optimizer.zero_grad()
        loss = criterion(model(batch["image"].to(device)), batch["label"].to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(train_dl)


def evaluate():
    """Return (val log loss, raw logits as numpy [N, 8] in val order)."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in val_dl:
            chunks.append(model(batch["image"].to(device)).cpu())
    logits = torch.cat(chunks).numpy()
    ll = log_loss(truth, softmax(logits, axis=1), labels=list(range(NUM_CLASSES)))
    return ll, logits


best_ll = float("inf")
best_logits = None
for epoch in range(1, args.epochs + 1):
    train_loss = train_one_epoch()
    val_ll, logits = evaluate()
    print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_logloss {val_ll:.4f}")
    if val_ll < best_ll:
        best_ll = val_ll
        best_logits = logits
        torch.save(model.state_dict(), OUT)

# ---------- extras: calibration temperature + class-collapse check ----------
grid = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
cal_ll, temp = min((log_loss(truth, softmax(best_logits / t, axis=1), labels=list(range(NUM_CLASSES))), t) for t in grid)
print(f"best {best_ll:.4f}  temp {temp}  calibrated {cal_ll:.4f}  saved {OUT}")

pred = best_logits.argmax(axis=1)
for i, sp in enumerate(species):
    mask = truth == i
    acc = (pred[mask] == i).mean() if mask.sum() else 0.0
    print(f"  {sp:18s} acc {acc:.3f}  true {int(mask.sum()):4d}  pred {int((pred == i).sum()):4d}")

# ---------- predict the test set: best epoch + the temp we just fit ----------
model.load_state_dict(torch.load(OUT))
model.eval()
sub = pd.read_csv("submission_format.csv", index_col="id")
test_paths = pd.read_csv("test_features.csv", index_col="id").loc[sub.index].filepath
test_dl = DataLoader(Images(test_paths), batch_size=BATCH, num_workers=workers, pin_memory=True)
probs = []
with torch.no_grad():
    for batch in tqdm(test_dl):
        probs += torch.softmax(model(batch["image"].to(device)) / temp, dim=1).tolist()
out_csv = f"submission_{TAG}.csv"
pd.DataFrame(probs, index=test_paths.index, columns=species)[sub.columns].to_csv(out_csv)
print(f"wrote {os.path.abspath(out_csv)}  ({len(probs)} rows, temp {temp})")
