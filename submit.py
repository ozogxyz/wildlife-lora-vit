import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_vit import ViT

from lora import LoRA_ViT

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--ckpt", default="best.pth")
p.add_argument("--out", default="submission.csv")
p.add_argument("--rank", type=int, default=4, help="must match the trained checkpoint")
p.add_argument("--temp", type=float, default=1.0, help="softmax temperature from CV (fold avg)")
p.add_argument("--tta", action="store_true", help="average probabilities over hflip")
p.add_argument("--batch", type=int, default=64)
args = p.parse_args()

NUM_CLASSES = 8
NORM_MEAN = 0.5
NORM_STD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count() or 2

backbone = ViT("B_16", pretrained=True)
IMG_SIZE = backbone.image_size
IMG_SIZE = IMG_SIZE[0] if isinstance(IMG_SIZE, (tuple, list)) else IMG_SIZE

os.chdir(args.data_dir)
sub = pd.read_csv("submission_format.csv", index_col="id")
species_labels = list(sub.columns)
test_features = pd.read_csv("test_features.csv", index_col="id").loc[sub.index]
print(f"device: {device}, {len(sub)} test images, ckpt {args.ckpt}, rank {args.rank}, temp {args.temp}, tta {args.tta}")


class TestDataset(Dataset):
    def __init__(self, x_df):
        self.data = x_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data.iloc[idx]["filepath"])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).float() / 255.0
        img = (img - NORM_MEAN) / NORM_STD
        return {"image_id": self.data.index[idx], "image": img}


dl = DataLoader(TestDataset(test_features), batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=True)

model = LoRA_ViT(backbone, r=args.rank, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()


def probs(img):
    return nn.functional.softmax(model(img) / args.temp, dim=1)


rows, ids = [], []
with torch.no_grad():
    for batch in tqdm(dl, total=len(dl)):
        img = batch["image"].to(device)
        p_out = probs(img)
        if args.tta:
            p_out = (p_out + probs(torch.flip(img, dims=[3]))) / 2
        rows.append(p_out.cpu().numpy())
        ids.extend(batch["image_id"])

preds = pd.DataFrame(np.concatenate(rows), index=ids, columns=species_labels).loc[sub.index]
preds.index.name = "id"
preds.to_csv(args.out)
print(f"wrote {os.path.abspath(args.out)}  {preds.shape}")
