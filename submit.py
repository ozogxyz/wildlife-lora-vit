import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_pretrained_vit import ViT

from lora import LoRA_ViT, TARGET_PRESETS

TEMP = 1.0  # set to the 'temp' printed by train.py (post-hoc calibration)

REPO = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
os.chdir(os.path.join(REPO, "data"))

sub = pd.read_csv("submission_format.csv", index_col="id")
species = list(sub.columns)
test = pd.read_csv("test_features.csv", index_col="id").loc[sub.index]

# rebuild the trained model — the checkpoint records its own LoRA config
ckpt = torch.load(os.path.join(REPO, "assets", "best.pth"), map_location=device)
model = LoRA_ViT(ViT("B_16", pretrained=True), r=ckpt["rank"], num_classes=8, targets=TARGET_PRESETS[ckpt["targets"]]).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths.iloc[i])
        img = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return (img - 0.5) / 0.5


loader = DataLoader(Images(test.filepath), batch_size=64, num_workers=os.cpu_count() or 2)

probs = []
with torch.no_grad():
    for img in tqdm(loader):
        probs += torch.softmax(model(img.to(device)) / TEMP, dim=1).tolist()

pd.DataFrame(probs, index=test.index, columns=species).to_csv("submission.csv")
print(f"wrote submission.csv  ({len(probs)} rows, temp {TEMP})")
