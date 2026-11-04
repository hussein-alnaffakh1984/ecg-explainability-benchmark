# Numeric results (summary)

All values are read from `results/master_results.json`, which `src/08_collect_results.py` rebuilds from `results/raw/`.

## Axis A — ABPC minus random, ResNet1D (mean over 10 seeds, seed-level 95% CI)

| Method | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| GradientSHAP | +0.289 [+0.280, +0.299] | +0.023 [-0.003, +0.049] | +0.045 [+0.029, +0.062] | -0.007 [-0.022, +0.009] | +0.273 [+0.253, +0.293] |
| Integrated Gradients | +0.292 [+0.282, +0.301] | +0.023 [-0.003, +0.049] | +0.045 [+0.028, +0.061] | -0.006 [-0.021, +0.009] | +0.275 [+0.256, +0.293] |
| DeepLIFT | +0.300 [+0.292, +0.308] | +0.029 [+0.001, +0.056] | +0.038 [+0.023, +0.054] | -0.008 [-0.022, +0.007] | +0.266 [+0.246, +0.287] |
| Grad×Input | +0.271 [+0.256, +0.285] | +0.021 [-0.005, +0.047] | +0.040 [+0.025, +0.055] | -0.007 [-0.021, +0.007] | +0.269 [+0.248, +0.289] |
| Saliency | +0.024 [+0.004, +0.045] | +0.010 [-0.003, +0.023] | -0.014 [-0.019, -0.009] | +0.028 [+0.017, +0.038] | +0.219 [+0.203, +0.235] |
| Grad-CAM | +0.052 [+0.048, +0.057] | +0.011 [+0.002, +0.020] | +0.001 [-0.004, +0.007] | +0.006 [+0.002, +0.009] | +0.021 [+0.016, +0.027] |

## Shortcut detection, ResNet1D (% of 200 evaluation records above the held-out threshold)

| Map | base_aVR_A8_W40 | late_t600 | lowamp_A2 | morph_gauss_A2 | narrow_W10 | physio_LAelectrode | vlowamp_A0.5 | wide_lowfreq_W200 |
|---|---|---|---|---|---|---|---|---|
| DeepLIFT | 100 | 100 | 100 | 99 | 100 | 100 | 98 | 99 |
| Grad×Input | 100 | 100 | 100 | 98 | 100 | 95 | 98 | 99 |
| Integrated Gradients | 100 | 100 | 100 | 99 | 100 | 98 | 99 | 99 |
| GradientSHAP | 100 | 100 | 100 | 98 | 100 | 96 | 99 | 98 |
| Saliency | 0 | 0 | 96 | 98 | 0 | 0 | 100 | 54 |
| Grad-CAM (block 4) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| |x| amplitude | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| Random | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Shortcut detection, Transformer (% of 200 evaluation records, held-out threshold)

| Map | base_aVR_A8_W40 | lowamp_A2 | morph_gauss_A2 |
|---|---|---|---|
| Integrated Gradients | 100 | 100 | 100 |
| GradientSHAP | 99 | 99 | 93 |
| Grad×Input (= DeepLIFT) | 32 | 60 | 40 |
| Saliency | 0 | 3 | 12 |
| Grad-CAM (stem) | 0 | 0 | 0 |
| |x| amplitude | 1 | 0 | 0 |

## Runtime (ms per ECG, NVIDIA T4)

Saliency: 6.0, GradXInput: 6.9, DeepLIFT: 12.6, IG: 40.6, GradientSHAP: 7.3, GradCAM_L4: 4.8, GradCAMpp_L4: 7.2
