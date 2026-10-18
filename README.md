# BCLNet

Implementation of **Boundary-Calibrated Lightweight Network for Domain-Generalized Polyp Segmentation**.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`x[N,3,H,W]` and binary `y[N,1,H,W]` contain paired RGB images and polyp masks. Use source-only Kvasir/CVC-ClinicDB training groups and explicit held-out dataset split codes for CVC-ColonDB, ETIS, or another declared target. No target images enter source training.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent source units; related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

`demo` generates synthetic inputs for this tensor interface; application-specific results require acquired data.

For paired folders with matching filenames, replace the `demo` command below with `bclnet pack --images data/images --targets data/masks --output data/example.npz --size 64 --mask`. Set `group` to actual source-image or patient/sequence IDs and supply the declared `split` codes before `prepare` when using grouped or external target data.

```bash
bclnet demo --output data/example.npz --count 96 --config config.json --seed 13
bclnet prepare --input data/example.npz --output data/prepared
```

## Training and inference

```bash
bclnet train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

## Evaluation

```bash
bclnet evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
bclnet plot --input runs --output runs/figures
```

## Ablations

Supported controls: `baseline`, `no_temperature`, `no_calibration`, `no_consistency`. Keep the input groups, seed and budget fixed.

```bash
bclnet train --input data/prepared --config config.json --output runs/no_temperature --variant no_temperature --epochs 100 --batch-size 16 --seed 13
```

## Outputs

`data/prepared/` contains train/validation/calibration/test NPZs. Training writes `model.pt`, `history.json` and `training.json` with run arguments. Evaluation writes `predictions.npz` and `metrics.json`; plotting reads saved metric files.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training or full evaluation was run. The GitHub Actions workflow checks source syntax and package installation.
