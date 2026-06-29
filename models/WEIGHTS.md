# Trained weights

The ten ResNet1D baselines (~8.74M parameters each, seeds [42,1,7,2,3,4,5,6,8,9])
are **not committed to this repository** because of size and to keep the repo light.

Two supported options:

1. **Regenerate (recommended, deterministic).**
   `python src/01_rebuild_and_train.py` reproduces all ten weights from the raw
   PTB-XL release. Training is seed-deterministic; the resulting test macro-AUROC is
   0.9262 ± 0.0009 (range [0.9249, 0.9276]).

2. **Download.**
   Host the weights outside the git tree and link them here:
   - GitHub Releases (attach `resnet1d_seed*.pt`), or
   - Git LFS (`.gitattributes` already tracks `*.pt`), or
   - a public/private dataset (e.g., Kaggle/Zenodo) — add the DOI/URL here.

Place the downloaded files in `models/` as `resnet1d_seed{42,1,7,2,3,4,5,6,8,9}.pt`.
