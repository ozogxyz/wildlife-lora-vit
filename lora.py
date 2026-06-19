import math

from pytorch_pretrained_vit import ViT
from torch import Tensor, nn

# LoRA surgery on a ViT, lifted from:
#   https://github.com/JamesQFreeman/LoRA-ViT
# ViT backbone from:
#   https://github.com/lukemelas/PyTorch-Pretrained-ViT  (pip install pytorch_pretrained_vit)


TARGET_PRESETS = {
    "qv": [("attn", "proj_q"), ("attn", "proj_v")],
    "qkv": [("attn", "proj_q"), ("attn", "proj_k"), ("attn", "proj_v")],
    "qkvm": [("attn", "proj_q"), ("attn", "proj_k"), ("attn", "proj_v"), ("pwff", "fc1"), ("pwff", "fc2")],
}


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        return self.w(x) + self.w_b(self.w_a(x))


class LoRA_ViT(nn.Module):
    """Low-rank adaptation on a frozen ViT.

    Freezes the backbone and grafts rank-r adapters onto each (sub, name) linear
    listed in TARGETS, for every (or selected) attention block. Defaults to the
    q/v projections; extend by overriding TARGETS, e.g. add the MLP:

        TARGETS = [("attn", "proj_q"), ("attn", "proj_v"),
                   ("pwff", "fc1"), ("pwff", "fc2")]

        model = ViT('B_16_imagenet1k')
        lora  = LoRA_ViT(model, r=8, num_classes=8)
        preds = lora(img)              # -> [B, num_classes]
    """

    TARGETS = [("attn", "proj_q"), ("attn", "proj_v")]

    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None, targets=None):
        super().__init__()
        assert r > 0
        self.targets = targets if targets is not None else self.TARGETS

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))

        self.w_As = []  # the A (down) projections
        self.w_Bs = []  # the B (up) projections

        for param in vit_model.parameters():
            param.requires_grad = False

        for i, blk in enumerate(vit_model.transformer.blocks):
            if i not in self.lora_layer:
                continue
            for sub, name in self.targets:
                parent = getattr(blk, sub)
                linear = getattr(parent, name)
                w_a = nn.Linear(linear.in_features, r, bias=False)
                w_b = nn.Linear(r, linear.out_features, bias=False)
                self.w_As.append(w_a)
                self.w_Bs.append(w_b)
                setattr(parent, name, _LoRALayer(linear, w_a, w_b))

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)
