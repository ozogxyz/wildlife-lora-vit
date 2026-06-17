#!/usr/bin/env bash
# One-time setup on the GCP VM — run once from the repo root; persistent disk keeps it.
# fovea installs over ssh (box ssh key must be on GitHub). HF_TOKEN comes from ./.env.
set -e
set -a; [ -f .env ] && . ./.env; set +a

pip install -q -r requirements.txt
pip install -q "git+ssh://git@github.com/ozogxyz/fovea.git"

# conservision data from HF, once
if [ ! -d data/train_features ]; then
  python3 - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tar = hf_hub_download("motorbreath/conservision", "conservision.tar",
                      repo_type="dataset", local_dir=".",
                      token=os.environ.get("HF_TOKEN"))
with tarfile.open(tar) as t:
    t.extractall("data")
os.remove(tar)
print("data:", len(os.listdir("data/train_features")), "train imgs")
PY
else
  echo "data: present"
fi
