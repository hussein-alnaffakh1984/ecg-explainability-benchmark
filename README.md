# ECG Explanation-Faithfulness Benchmark (PTB-XL)

A reproducible benchmark that separates **explainer faithfulness** from **clinical
validity** for deep ECG diagnosis on the public PTB-XL dataset. Six attribution
methods (Saliency, Grad×Input, DeepLIFT, Integrated Gradients, Grad-CAM, DeepSHAP)
and a random baseline are scored along three orthogonal axes plus a calibrated
shortcut-injection ground-truth test.

> **Status: unpublished manuscript under peer review.**
> This repository contains code and reproduction scripts only. It does **not**
> contain the manuscript text, abstract, or figures-as-published. Please do not
> redistribute or index the contents as a publication. A citation will be added
> here once the article is accepted.

## What this repo reproduces
- 10-seed ResNet1D baseline on PTB-XL (test macro-AUROC ≈ 0.9262 ± 0.0009).
- Axis A (faithfulness, leakage-controlled ROAD/ABPC), Axis B (clinical validity,
  lead-aware regions), Axis C (randomization sensitivity), for 6 methods × 5
  superclasses × 10 seeds.
- Calibrated shortcut-injection detection for all 6 methods.
- Nested mixed model on per-record ABPC.
- All publication figures.

## Layout
```
src/01_rebuild_and_train.py                 # rebuild X/Y from raw PTB-XL + train 10 seeds (resumable)
src/02_confirmatory_axes_injection_glmm.py  # axes A/B/C + injection + mixed model (uses saved models)
src/03_make_figures.py                      # regenerate all figures into figures/
docs/pre_registration.md                    # pre-registered hypotheses H1–H4
docs/DATA.md                                # how to obtain PTB-XL (not redistributed here)
docs/REPRODUCE.md                           # step-by-step
models/WEIGHTS.md                           # how to obtain/regenerate the 10 trained weights
results/results_summary.md                  # numeric results (values only)
```

## Quick start
```bash
pip install -r requirements.txt
# 1) rebuild data + train (needs raw PTB-XL, see docs/DATA.md); resumable, saves each seed
python src/01_rebuild_and_train.py
# 2) compute all axes + injection + mixed model from the saved models
python src/02_confirmatory_axes_injection_glmm.py
# 3) regenerate figures
python src/03_make_figures.py
```
Paths default to a Kaggle layout (`/kaggle/working`, `/kaggle/input/...`). Edit the
`WORK` and dataset paths at the top of each script for a local run.

## Reproducibility
Training is seed-deterministic (seeds `[42,1,7,2,3,4,5,6,8,9]`); the data rebuild is
deterministic from the raw release. Re-running `01` reproduces the same weights and
the same AUROC to within reported tolerance.

## License
Code is released under the MIT License (see `LICENSE`). PTB-XL is **not** included
and is governed by its own license (PhysioNet, see `docs/DATA.md`).
