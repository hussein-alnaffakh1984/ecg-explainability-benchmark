# ecg-explainability-benchmark

Code, trained Transformer weights, per-run results and figure scripts for an audit protocol
that evaluates attribution methods for 12-lead ECG classifiers trained on PTB-XL (100 Hz).

Six attribution methods (Saliency, Grad×Input, DeepLIFT, Integrated Gradients, GradientSHAP,
Grad-CAM) are scored on two architectures (ResNet1D, ten seeds; Transformer encoder, three seeds):

- **Axis A, faithfulness:** leakage-controlled ROAD with a neighbourhood imputer (k = 8, guard band 15),
  reported as ABPC minus a random map, with amplitude (|x|), smooth-random and QRS-mask control maps;
- **Axis B, clinical validity:** share of positive attribution inside delineated diagnostic regions;
- **Axis C, randomization sensitivity:** signed Spearman similarity after weight re-initialization
  (reported as a diagnostic);
- **Shortcut injection:** eight artefact settings with a reliance check and a threshold calibrated on
  held-out records.

## Repository layout

```
src/                       pipeline scripts (run in numerical order)
  01_rebuild_and_train.py           data tensors from raw PTB-XL + ResNet1D training (10 seeds)
  02_confirmatory_axes_injection_glmm.py   first confirmatory pipeline
  03_make_figures.py                first-round figures
  04_axes_resnet_10seeds.py         Axes A, B, C for ResNet1D (10 seeds)
  05_injection_gradcam_controls.py  shortcut injection, Grad-CAM variants, control maps, ROAD curves, runtime
  06_transformer.py                 Transformer training + Axes A-C + shortcut injection
  07_statistics.py                  seed-level CIs, bootstrap, mixed model, Holm, TOST, Axis C summaries
  08_collect_results.py             builds results/master_results.json from results/raw/
  09_make_figures.py                Figures 1-8 (600 dpi) from results/
docs/                      data access, reproduction steps, analysis plan
models/                    weight instructions; Transformer weights (transformer_weights.zip)
results/master_results.json  every number reported in the manuscript
results/statistics.json      output of 07_statistics.py
results/raw/                 per-seed and per-setting outputs of scripts 04-06
figures/                     Figures 1-8
```

In code and JSON files, the key `DeepSHAP` denotes GradientSHAP (Captum `GradientShap`) and
`GradXInput` denotes Grad×Input.

## Quick start

```bash
pip install -r requirements.txt
# reproduce the reported numbers and figures from the stored run outputs (CPU, about a minute):
python src/08_collect_results.py
python src/09_make_figures.py
```

To recompute everything from raw PTB-XL on a GPU, follow `docs/REPRODUCE.md`.

## Data

PTB-XL is **not redistributed here**; obtain it from PhysioNet under its license (see `docs/DATA.md`).

## License

Code is released under the MIT License (see `LICENSE`). PTB-XL retains its original license.
