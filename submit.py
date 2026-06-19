import argparse
import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_pretrained_vit import ViT

from lora import LoRA_ViT, TARGET_PRESETS

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--ckpt", default="best.pth")
p.add_argument("--out", default="submission.csv")
p.add_argument("--rank", type=int, default=4)
p.add_argument("--temp", type=float, default=1.0, help="softmax temperature from CV")
p.add_argument("--tta", action="store_true", help="average probabilities over a horizontal flip")
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--lora-targets", choices=list(TARGET_PRESETS), default="qv")
p.add_argument("--batch", type=int, default=64)
args = p.parse_args()
# --rank / --img-size / --lora-targets must match the trained checkpoint.

NUM_CLASSES = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
workers = os.cpu_count() or 2

os.chdir(args.data_dir)
sub = pd.read_csv("submission_format.csv", index_col="id")
species = list(sub.columns)
test = pd.read_csv("test_features.csv", index_col="id").loc[sub.index]
print(
    f"device {device}  {len(sub)} images  ckpt {args.ckpt}  rank {args.rank}  "
    f"img {args.img_size}  lora {args.lora_targets}  temp {args.temp}  tta {args.tta}"
)


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths.iloc[i])
        img = cv2.resize(img, (args.img_size, args.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = (img - 0.5) / 0.5
        return {"id": self.paths.index[i], "image": img}


loader = DataLoader(Images(test.filepath), batch_size=args.batch, num_workers=workers, pin_memory=True)

backbone = ViT("B_16", pretrained=True, image_size=args.img_size)
model = LoRA_ViT(backbone, r=args.rank, num_classes=NUM_CLASSES, targets=TARGET_PRESETS[args.lora_targets]).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()


def probs(images):
    out = torch.softmax(model(images) / args.temp, dim=1)
    if args.tta:
        flipped = torch.softmax(model(torch.flip(images, dims=[3])) / args.temp, dim=1)
        out = (out + flipped) / 2
    return out


parts = []
with torch.no_grad():
    for batch in tqdm(loader):
        preds = probs(batch["image"].to(device)).cpu()
        parts.append(pd.DataFrame(preds.tolist(), index=batch["id"], columns=species))

preds = pd.concat(parts).loc[sub.index]
preds.index.name = "id"
preds.to_csv(args.out)
print(f"wrote {os.path.abspath(args.out)}  {preds.shape}")
