# Conser-vision — camera-trap species classification

A LoRA-adapted ViT-B/16 for the 8-class wildlife species task in DrivenData's
[Conser-vision Practice Area](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).
The ViT backbone is frozen; low-rank adapters are injected into the q/v attention
projections, plus an 8-class head. Metric: multiclass log loss (lower is better).

## Result

Eval log loss ~1.0 (official benchmark: ~1.8).

## Run (free Colab T4)

1. Upload the competition data zip to `MyDrive/datadriven/` on Google Drive.
2. Generate the notebook: `jupytext --to notebook conservision.py -o conservision.ipynb`
3. Open it in Colab, set the runtime to **T4 GPU**, and **Run all**.

The notebook mounts Drive, installs deps, unzips the data to `/content/data`,
trains, and prints eval log loss + per-class accuracy.

## Note

Competition data is not included — DrivenData's rules forbid redistribution.
Set your own Drive path for the zip in `conservision.py`.
