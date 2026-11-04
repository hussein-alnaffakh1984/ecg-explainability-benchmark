# Step-by-step reproduction

The scripts were run on Kaggle (NVIDIA T4). Paths default to a Kaggle layout; edit the data and
working paths at the top of each script for a local run. Every heavy script is resumable.

1. **Install:** `pip install -r requirements.txt`
2. **Data and ResNet1D training:** obtain PTB-XL (see `docs/DATA.md`) and run
   `python src/01_rebuild_and_train.py`. It rebuilds `X.npy`, `Y.npy`, `split_index.csv` and trains
   ten seeds (mean test macro-AUROC 0.9262). Alternatively, download the trained weights
   (see `models/WEIGHTS.md`).
3. **Axes A, B, C (ResNet1D, 10 seeds):** `python src/04_axes_resnet_10seeds.py`
   → `ext_seeds/seed_*.json` (about 6 min per seed on a T4).
4. **Shortcut injection, Grad-CAM variants, control maps, ROAD curves, runtime:**
   `python src/05_injection_gradcam_controls.py` → `part2/`.
5. **Transformer:** `python src/06_transformer.py` → `part3/`.
6. **Statistics:** `python src/07_statistics.py` (CPU; set `WORK` to the folder holding `ext_seeds/`) → `statistics.json`.
   **Imputer leakage ablation:** `python src/10_imputer_leakage.py` (CPU) → leakage table.
7. **Collect and plot:** copy the outputs of steps 3-5 into `results/raw/resnet_axes/`,
   `results/raw/injection_controls/` and `results/raw/transformer/`, then run
   `python src/08_collect_results.py` and `python src/09_make_figures.py`.

The stored outputs in `results/raw/` reproduce `results/master_results.json` and all figures exactly.

Notes
- The neighbourhood imputer uses the validated setting k = 8, guard band g = 15.
- Attributions explain the pre-sigmoid logit of the target class; Integrated Gradients uses a zero
  baseline and 32 steps; GradientSHAP uses a zero and a N(0, 0.1²) reference.
