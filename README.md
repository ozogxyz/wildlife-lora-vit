# wildlife-lora-vit

LoRA-adapted ViT-B/16 for DrivenData's
[Conser-vision](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/page/409/)
camera-trap benchmark: 8-way species classification (Taï National Park, Côte
d'Ivoire), scored on log loss.

ImageNet ViT-B/16, backbone frozen. Rank-4 LoRA on the q/v projections of every
attention block, plus an 8-class head. Only the adapters and head train: ~150K
parameters, under 0.2% of the 86M backbone.

The dataset's real difficulty is its site structure, not the eight classes. Each
camera contributes long runs of near-identical frames, so a random split leaks
backgrounds across train and validation and inflates the score, while the held-out
test set is entirely unseen cameras. Validation here is grouped by site, so the
metric reflects the only thing that transfers. Class-weighted loss and light
augmentation cover the tail (bird, hog, civet, rodent).
