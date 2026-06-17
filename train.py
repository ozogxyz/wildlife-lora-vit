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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss

from fovea import LoRA_ViT, ColorJitterCV, RandomGaussianBlur, RandomHorizontalFlip

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--out", default="best.pth")
p.add_argument("--rank", type=int, default=8)
p.add_argument("--frac", type=float, default=1.0)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

DATA_DIR = args.data_dir
OUT = args.out
RANK = args.rank
NUM_CLASSES = 8
FRAC = args.frac
TEST_SIZE = 0.25
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 1
AUGMENT = True
NORM_MEAN = 0.5
NORM_STD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count() or 2
print(f"device: {device} ({gpu}), batch {BATCH_SIZE}, workers {NUM_WORKERS}, frac {FRAC}, epochs {EPOCHS}")

backbone = ViT("B_16", pretrained=True)
IMG_SIZE = backbone.image_size
IMG_SIZE = IMG_SIZE[0] if isinstance(IMG_SIZE, (tuple, list)) else IMG_SIZE

os.chdir(DATA_DIR)
train_features = pd.read_csv("train_features.csv", index_col="id")
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())

y = train_labels.sample(frac=FRAC, random_state=SEED)
x = train_features.loc[y.index].filepath.to_frame()
sites = train_features.loc[y.index, "site"]
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, eval_idx = next(gss.split(x, y, groups=sites))
x_train, x_eval = x.iloc[train_idx], x.iloc[eval_idx]
y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]


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


train_ds = ImagesDataset(x_train, y_train, mode="train" if AUGMENT else "eval")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
eval_ds = ImagesDataset(x_eval, y_eval, mode="eval")
eval_dl = DataLoader(eval_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

model = LoRA_ViT(backbone, r=RANK, num_classes=NUM_CLASSES).to(device)
counts = y_train[species_labels].sum().values
weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float, device=device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)


def evaluate():
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in eval_dl:
            preds = nn.functional.softmax(model(batch["image"].to(device)), dim=1)
            parts.append(pd.DataFrame(preds.cpu().numpy(), index=batch["image_id"], columns=species_labels))
    preds_df = pd.concat(parts).loc[y_eval.index]
    ll = log_loss(y_eval.idxmax(axis=1), preds_df[species_labels], labels=species_labels)
    return ll, preds_df


best_ll = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch in tqdm(train_dl, total=len(train_dl)):
        optimizer.zero_grad()
        loss = criterion(model(batch["image"].to(device)), batch["label"].to(device))
        loss.backward()
        optimizer.step()
    ll, _ = evaluate()
    print(f"epoch {epoch:2d}  train_loss {loss.item():.4f}  eval_logloss {ll:.4f}")
    if ll < best_ll:
        best_ll = ll
        torch.save(model.state_dict(), OUT)
print(f"best eval log loss: {best_ll:.4f}")

model.load_state_dict(torch.load(OUT))
_, preds_df = evaluate()
true = y_eval.idxmax(axis=1)
pred = preds_df.idxmax(axis=1)
print(f"best eval accuracy: {(pred == true).mean():.4f}")
for sp in species_labels:
    m = true == sp
    if m.sum():
        print(f"  {sp:18s} {(pred[m] == true[m]).mean():.3f}  (n={m.sum()})")
