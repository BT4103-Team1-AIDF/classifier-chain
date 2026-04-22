# Classifier Chain Models

This repository contains classifier-chain experiments and submission artifacts for multi-horizon default prediction.

## Contents

- `benchmarking/catboost_classifier_chain.py`: CatBoost-based classifier chain with Optuna tuning and horizon-level probability outputs.
- `benchmarking/lightgbm_classifier_chain.py`: LightGBM-based classifier chain with class weighting, Optuna tuning, and probability forecasts.
- `benchmarking/xgboost_classifier_chain.py`: XGBoost-based classifier chain with Optuna tuning and multi-horizon evaluation.
- `Final_Submission/Competition/Ingestion.zip`: Packaged Codabench ingestion program.
- `Final_Submission/Competition/Scoring.zip`: Packaged Codabench scoring program.
- `Final_Submission/Models/CatBoost.zip`: Codabench-ready CatBoost submission bundle.
- `Final_Submission/Models/LightGBM.zip`: Codabench-ready LightGBM submission bundle.
- `Final_Submission/Models/XGboost.zip`: Codabench-ready XGBoost submission bundle.
- `Final_Submission/Models/model_classifier_chain_logistic.zip`: Logistic-regression classifier-chain submission bundle.
- `Final_Submission/Models/model_classifier_chain_random_forest.zip`: Random-forest classifier-chain submission bundle.
- `Final_Submission/Models/*.pdf`: Model cards for the packaged submissions.

## Problem Setup

The project uses monthly horizon labels:

- `y_1m`
- `y_3m`
- `y_6m`
- `y_12m`
- `y_24m`
- `y_36m`
- `y_48m`
- `y_60m`

These correspond to evaluation at:

- 1, 3, 6, 12, 24, 36, 48, and 60 months.

The benchmarking scripts train a classifier chain over these ordered horizons, so predictions for earlier horizons are fed into later-horizon models.

The workflows assume a 3-class target setup:

- `0`: non-default / negative class
- `1`: default class
- `2`: an additional class handled by the repository's modeling code

## Inputs And Outputs

Training and test tables are expected to include identifier columns:

- `CompNo`
- `yyyy`
- `mm`

The benchmarking scripts look for target columns with either of these prefixes:

- `y_1m ... y_60m`
- `default_1m ... default_60m`

By default, the scripts expect data files named:

- `train_multilabel_upto2014_1to60m.csv`
- `test_multilabel_from2015_1to60m.csv`

These filenames are defaults in the scripts and can be overridden with command-line arguments.

Benchmarking outputs are saved as probability columns:

- `prob_1m_class_0`, `prob_1m_class_1`, `prob_1m_class_2`
- `prob_3m_class_0`, `prob_3m_class_1`, `prob_3m_class_2`
- ...
- `prob_60m_class_0`, `prob_60m_class_1`, `prob_60m_class_2`

## Environment

Recommended: Python 3.10+ in a virtual environment.

Common dependencies used in this repository:

- `pandas`
- `numpy`
- `scikit-learn`
- `optuna`
- `catboost`
- `lightgbm`
- `xgboost`

If you plan to load Excel files with `pandas.read_excel`, install an Excel engine such as `openpyxl` as well.

Example setup:

```bash
python -m venv .venv
pip install pandas numpy scikit-learn optuna catboost lightgbm xgboost openpyxl
```

Activate the environment with:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

## Running The Benchmarking Scripts

Run any of the benchmark workflows from the repository root:

```bash
python benchmarking/catboost_classifier_chain.py
python benchmarking/lightgbm_classifier_chain.py
python benchmarking/xgboost_classifier_chain.py
```

Useful options supported by the scripts include:

- `--train-path`: path to the training CSV or Excel file
- `--test-path`: path to the test CSV or Excel file
- `--no-tuning`: disable Optuna hyperparameter tuning
- `--n-trials`: number of Optuna trials
- `--validation-size`: validation split used during tuning
- `--tuning-sample-size`: cap the number of rows used during tuning
- `--save-results`: output CSV path for predicted probabilities

Example:

```bash
python benchmarking/lightgbm_classifier_chain.py \
  --train-path train_multilabel_upto2014_1to60m.csv \
  --test-path test_multilabel_from2015_1to60m.csv \
  --n-trials 10 \
  --save-results outputs/lightgbm_classifier_chain_probabilities.csv
```

Each script performs:

- feature preprocessing with numeric coercion and median imputation
- simple feature engineering for month and selected financial variables
- optional Optuna hyperparameter tuning
- sequential multi-horizon classifier-chain training
- held-out evaluation using default AUC, multiclass AUC, accuracy, macro F1, and log loss

## Codabench Programs

### Ingestion Program

Packaged path:

- `Final_Submission/Competition/Ingestion.zip`

Expected command from the included metadata:

```bash
python ingestion.py <input_dir> <output_dir> <submission_dir>
```

The ingestion program:

- loads a participant `model.py` from `<submission_dir>`
- optionally calls `Model().fit(X_train, y_train)` if a training CSV is present
- calls `model.predict_proba(features)`
- expects a probability array of shape `(n_rows, 24)` for 8 horizons x 3 classes
- writes `predictions.csv`
- requires identifier columns `CompNo`, `yyyy`, `mm`
- exports class-1 risk columns `p1_1m`, `p1_3m`, `p1_6m`, `p1_12m`, `p1_24m`, `p1_36m`, `p1_48m`, `p1_60m`

Only the class-1 probability is exported for scoring.

### Scoring Program

Packaged path:

- `Final_Submission/Competition/Scoring.zip`

Expected command from the included metadata:

```bash
python scoring.py <input_dir> <output_dir>
```

Inputs:

- `ref/labels.csv`
- `res/predictions.csv`

Outputs:

- `scores.txt`
- `scores.json`
- `per_horizon_metrics.csv`

Primary metric:

- `AUC_MEAN_8H`, the mean ROC AUC across valid horizons

The scorer also records per-horizon metrics as:

- `AUC_1M`
- `AUC_3M`
- `AUC_6M`
- `AUC_12M`
- `AUC_24M`
- `AUC_36M`
- `AUC_48M`
- `AUC_60M`

## Repository Notes

- The current repository mostly contains scripts and packaged submission artifacts rather than raw datasets.
- Default training and test filenames are referenced in the benchmarking scripts, but the CSV files are not committed in this repository.
- Some packaged submission bundles contain precomputed lookup files in addition to `model.py`.
- A `.DS_Store` file and `__pycache__` artifacts are present and can be ignored.

## License

No license file is currently included in this repository.
