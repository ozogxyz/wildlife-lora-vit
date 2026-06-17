"""One-time: pull the conservision dataset from HF into ./data.

huggingface_hub picks up auth from HF_TOKEN or your `huggingface-cli login` cache.
"""
import os
import tarfile

from huggingface_hub import hf_hub_download

tar = hf_hub_download(
    "motorbreath/conservision", "conservision.tar", repo_type="dataset", local_dir="."
)
with tarfile.open(tar) as t:
    t.extractall("data")
os.remove(tar)
print("data ready:", len(os.listdir("data/train_features")), "train imgs")
