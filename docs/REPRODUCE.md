# Step-by-step reproduction

1. **Install**: `pip install -r requirements.txt`
2. **Get PTB-XL** (see docs/DATA.md) and set the raw path in `src/01_rebuild_and_train.py`.
3. **Rebuild data + train 10 seeds** (resumable; saves each seed):
   `python src/01_rebuild_and_train.py`
   Expected: test macro-AUROC ≈ 0.9262 ± 0.0009, range [0.9249, 0.9276].
4. **Compute axes + injection + mixed model** (uses the saved models, no retraining):
   `python src/02_confirmatory_axes_injection_glmm.py`
   Produces: `ext_seeds/seed_*.json`, `axes_10seed_summary.json`,
   `step24_calibration.json`, `mixedmodel_abpc.txt`.
5. **Figures**: `python src/03_make_figures.py` → writes to `figures/`.

Notes:
- Scripts default to a Kaggle layout; edit `WORK`/dataset paths for local runs.
- Step 4 is resumable and memory-safe; re-run if a session is interrupted.
- The neighbourhood imputer uses the validated gap=15 setting; do not change it.
