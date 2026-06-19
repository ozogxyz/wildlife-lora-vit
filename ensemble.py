import os
import sys

import pandas as pd

# Average several submission CSVs into one (CV-fold ensemble).
#   python3 ensemble.py data/submission_qv_r4_lr0.0005_f*.csv
files = sys.argv[1:]
if not files:
    sys.exit("usage: python3 ensemble.py sub1.csv sub2.csv ...")

dfs = [pd.read_csv(f, index_col="id") for f in files]
avg = sum(dfs) / len(dfs)  # mean of per-class probabilities; rows still sum to 1
avg.to_csv("ensemble.csv")
print(f"averaged {len(dfs)} files -> {os.path.abspath('ensemble.csv')}")
