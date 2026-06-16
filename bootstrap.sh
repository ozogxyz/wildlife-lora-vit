#!/usr/bin/env bash
# Runs ON the pod. Restores the ephemeral python env (fovea+deps) after every
# pod start; data/weights persist on the volume. GH_PAT falls back to the copy
# stored on the volume, so a restart needs no tokens. HF_TOKEN only needed to
# seed data the first time.
set -e
: "${GH_PAT:=$(cat /workspace/.ghpat 2>/dev/null)}"

# opencv needs these system libs
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null

# python deps + fovea (private repo, installed via the read PAT)
pip install -q --root-user-action=ignore --break-system-packages \
  pytorch_pretrained_vit huggingface_hub pandas scikit-learn tqdm \
  "git+https://${GH_PAT}@github.com/ozogxyz/fovea.git"

# data lives on the network volume — seed it from HF once, then it persists across pods
if [ ! -d /workspace/data/train_features ]; then
  python3 - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tar = hf_hub_download("motorbreath/conservision", "conservision.tar",
                      repo_type="dataset", local_dir="/workspace",
                      token=os.environ["HF_TOKEN"])
with tarfile.open(tar) as t:
    t.extractall("/workspace/data")
os.remove(tar)
print("data seeded:", len(os.listdir("/workspace/data/train_features")), "train imgs")
PY
else
  echo "data: already on volume"
fi
