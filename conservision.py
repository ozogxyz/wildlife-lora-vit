# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch_pretrained_vit"])

# %%
import os, shutil
os.makedirs('/content/data', exist_ok=True)
shutil.unpack_archive('/content/drive/MyDrive/datadriven/competition_VfIpjyh.zip', '/content/data')
print(os.listdir('/content/data'))

# %%
import math
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from pytorch_pretrained_vit import ViT
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss


# %%
class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        return self.w(x) + self.w_b(self.w_a(x))


class LoRA_ViT(nn.Module):
    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super().__init__()
        assert r > 0
        dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        self.w_As = []
        self.w_Bs = []

        for param in vit_model.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
batch_size = 64 if "A100" in gpu_name else 32
print(f"device: {device} ({gpu_name}), batch_size: {batch_size}")

backbone = ViT("B_16", pretrained=True)
IMG_SIZE = backbone.image_size
IMG_SIZE = IMG_SIZE[0] if isinstance(IMG_SIZE, (tuple, list)) else IMG_SIZE
print(f"image size: {IMG_SIZE}")

# %%
DATA_PATH = "/content/data/"
os.chdir(DATA_PATH)
train_features = pd.read_csv("train_features.csv", index_col="id")
test_features = pd.read_csv("test_features.csv", index_col="id")
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())

# %%
frac = 0.2
y = train_labels.sample(frac=frac, random_state=1)
x = train_features.loc[y.index].filepath.to_frame()
sites = train_features.loc[y.index, "site"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
train_idx, eval_idx = next(gss.split(x, y, groups=sites))
x_train, x_eval = x.iloc[train_idx], x.iloc[eval_idx]
y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

# %%
class ImagesDataset(Dataset):
    def __init__(self, x_df, y_df=None, augment=False):
        self.data = x_df
        self.label = y_df
        aug = (
            [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2)]
            if augment
            else []
        )
        self.transform = transforms.Compose(
            [transforms.Resize((IMG_SIZE, IMG_SIZE))]
            + aug
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)

# %%
train_dataset = ImagesDataset(x_train, y_train, augment=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# %%
model = LoRA_ViT(backbone, r=16, num_classes=8).to(device)
counts = y_train[species_labels].sum().values
weights = torch.tensor(
    counts.sum() / (len(counts) * counts), dtype=torch.float, device=device
)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4
)

# %%
num_epochs = 2
for epoch in range(1, num_epochs + 1):
    print(f"Starting epoch {epoch}")
    for batch_n, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        outputs = model(batch["image"].to(device))
        loss = criterion(outputs, batch["label"].to(device))
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} loss: {float(loss)}")

torch.save(model, "model.pth")

# %%
eval_dataset = ImagesDataset(x_eval, y_eval)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

preds_collector = []
model.eval()
with torch.no_grad():
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        logits = model.forward(batch["image"].to(device))
        preds = nn.functional.softmax(logits, dim=1)
        preds_df = pd.DataFrame(
            preds.detach().cpu().numpy(),
            index=batch["image_id"],
            columns=species_labels,
        )
        preds_collector.append(preds_df)
eval_preds_df = pd.concat(preds_collector).loc[y_eval.index]

eval_true = y_eval.idxmax(axis=1)
eval_predictions = eval_preds_df.idxmax(axis=1)

accuracy = (eval_predictions == eval_true).mean()
logloss = log_loss(eval_true, eval_preds_df[species_labels], labels=species_labels)
print(f"eval accuracy: {accuracy:.4f}")
print(f"eval log loss: {logloss:.4f}   (competition metric -- lower is better)")

print("per-class accuracy:")
for sp in species_labels:
    mask = eval_true == sp
    if mask.sum():
        print(f"  {sp:18s} {(eval_predictions[mask] == eval_true[mask]).mean():.3f}  (n={mask.sum()})")
