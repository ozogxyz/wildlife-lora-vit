import argparse
import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_pretrained_vit import ViT

from lora import LoRA_ViT, TARGET_PRESETS

REPO = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.join(REPO, "assets", "best.pth"))
p.add_argument("--out", default="submission.csv")
p.add_argument("--rank", type=int, default=4, help="must match the checkpoint")
p.add_argument("--lora-targets", choices=list(TARGET_PRESETS), default="qv", help="must match the checkpoint")
p.add_argument("--temp", type=float, default=1.0, help="softmax temperature from CV")
p.add_argument("--tta", action="store_true", help="average probabilities over a horizontal flip")
args = p.parse_args()

IMG_SIZE = 224
BATCH = 64
NUM_CLASSES = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
workers = os.cpu_count() or 2

os.chdir(os.path.join(REPO, "data"))
sub = pd.read_csv("submission_format.csv", index_col="id")
species = list(sub.columns)
test = pd.read_csv("test_features.csv", index_col="id").loc[sub.index]
print(f"device {device}  {len(sub)} images  ckpt {args.ckpt}  rank {args.rank}  lora {args.lora_targets}  temp {args.temp}  tta {args.tta}")


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths.iloc[i])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # test: clean preprocessing only, no random aug
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = (img - 0.5) / 0.5
        return {"id": self.paths.index[i], "image": img}


loader = DataLoader(Images(test.filepath), batch_size=BATCH, num_workers=workers, pin_memory=True)

backbone = ViT("B_16", pretrained=True)  # native 224; dataset resizes to IMG_SIZE
model = LoRA_ViT(backbone, r=args.rank, num_classes=NUM_CLASSES, targets=TARGET_PRESETS[args.lora_targets]).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()


def probs(images):
    out = torch.softmax(model(images) / args.temp, dim=1)
    if args.tta:  # test-time augmentation: average over a horizontal flip
        out = (out + torch.softmax(model(torch.flip(images, dims=[3])) / args.temp, dim=1)) / 2
    return out


parts = []
with torch.no_grad():
    for batch in tqdm(loader):
        p_batch = probs(batch["image"].to(device)).cpu()
        parts.append(pd.DataFrame(p_batch.tolist(), index=batch["id"], columns=species))

preds = pd.concat(parts).loc[sub.index]
preds.index.name = "id"
preds.to_csv(args.out)
print(f"wrote {os.path.abspath(args.out)}  {preds.shape}")
