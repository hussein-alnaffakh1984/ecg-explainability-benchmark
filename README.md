# ecg-xai-eval

Evaluation code for assessing attribution methods on a 12-lead ECG classifier
trained on a public dataset. The pipeline scores several gradient- and
perturbation-based attribution methods on three independent criteria and a
controlled sanity test, then aggregates across multiple random seeds.

> **Note.** This repository accompanies a research project that is **not yet
> published**. It is provided for reproducibility of the computational pipeline
> only. It deliberately contains **no manuscript text, abstract, figures, or
> results narrative**. Please do not index or cite this repository as a
> publication; a formal reference will be added after the associated article
> appears.

## Scope

The code lets a user:

- rebuild the input tensors from the raw signal release and train the classifier
  across a fixed set of random seeds (deterministic, resumable);
- score attribution maps on three criteria — a removal-based criterion, a
  region-overlap criterion, and a model-randomization criterion;
- run a controlled artefact-insertion sanity test that checks whether a method
  localizes a planted signal;
- fit a per-record mixed-effects model and regenerate all plots.

## Repository layout

```
src/        pipeline stages (train, evaluate, plot)
docs/        environment and data-access notes; analysis plan
models/      instructions for obtaining or regenerating the trained weights
results/     machine-readable run outputs
figures/     generated plots
```

## Usage

```bash
pip install -r requirements.txt
python src/01_rebuild_and_train.py                 # data + training (see docs for the data source)
python src/02_confirmatory_axes_injection_glmm.py  # all evaluation criteria + sanity test
python src/03_make_figures.py                      # plots
```

Paths default to a hosted-notebook layout; edit the working and dataset paths at
the top of each script to run locally. Training is seed-deterministic, so a fresh
run reproduces the same weights and scores within tolerance.

## Data

The signal dataset is **not redistributed here** and must be obtained from its
official source under its own license (see `docs/DATA.md`).

## License

Code is released under the MIT License (see `LICENSE`). The dataset retains its
original license.
