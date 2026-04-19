import argparse
from copy import deepcopy
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRAIN_PATH = str(PROJECT_ROOT / "train_multilabel_upto2014_1to60m.csv")
DEFAULT_TEST_PATH = str(PROJECT_ROOT / "test_multilabel_from2015_1to60m.csv")
DEFAULT_RESULTS_PATH = str(PROJECT_ROOT / "outputs" / "catboost_classifier_chain_probabilities.csv")
RANDOM_STATE = 42
HORIZONS = ["1m", "3m", "6m", "12m", "24m", "36m", "48m", "60m"]
TARGET_PREFIX_CANDIDATES = ["default_", "y_"]
ID_COLUMNS = ["CompNo", "yyyy", "mm"]
DEFAULT_TUNING_SAMPLE_SIZE = 15000
CLASSES = [0, 1, 2]
BASE_PARAMS = {
    "loss_function": "MultiClass",
    "iterations": 150,
    "depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "verbose": 0,
}


class ConstantProbabilityModel:
    def __init__(self, constant_value):
        self.constant_value = int(constant_value)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n_rows = len(X)
        out = np.zeros((n_rows, len(CLASSES)), dtype=float)
        out[:, self.constant_value] = 1.0
        return out


class ClassifierChain:
    def __init__(self, base_estimator, horizons):
        self.base_estimator = base_estimator
        self.horizons = horizons
        self.models = {}

    def fit(self, X, y):
        X_extended = X.copy()
        for i, horizon in enumerate(self.horizons):
            y_horizon = y.iloc[:, i]
            if y_horizon.nunique() < 2:
                model = ConstantProbabilityModel(y_horizon.iloc[0])
            else:
                model = deepcopy(self.base_estimator)
            model.fit(X_extended, y_horizon)
            self.models[horizon] = model

            if i < len(self.horizons) - 1:
                pred_proba = self._predict_proba_aligned(model, X_extended)
                pred_cols = [f"pred_{horizon}_class_{cls}" for cls in CLASSES]
                X_extended = pd.concat(
                    [X_extended, pd.DataFrame(pred_proba, columns=pred_cols, index=X_extended.index)],
                    axis=1,
                )
        return self

    def predict_proba(self, X):
        X_extended = X.copy()
        predictions = {}
        for i, horizon in enumerate(self.horizons):
            model = self.models[horizon]
            pred_proba = self._predict_proba_aligned(model, X_extended)
            predictions[horizon] = pred_proba

            if i < len(self.horizons) - 1:
                pred_cols = [f"pred_{horizon}_class_{cls}" for cls in CLASSES]
                X_extended = pd.concat(
                    [X_extended, pd.DataFrame(pred_proba, columns=pred_cols, index=X_extended.index)],
                    axis=1,
                )
        return predictions

    @staticmethod
    def _predict_proba_aligned(model, X):
        raw = np.asarray(model.predict_proba(X), dtype=float)
        out = np.zeros((raw.shape[0], len(CLASSES)), dtype=float)
        classes = np.asarray(getattr(model, "classes_", CLASSES), dtype=int)
        for i, cls in enumerate(classes):
            if 0 <= cls < len(CLASSES):
                out[:, cls] = raw[:, i]
        row_sum = out.sum(axis=1, keepdims=True)
        good = row_sum.squeeze() > 0
        out[good] = out[good] / row_sum[good]
        return out


class CatBoostWorkflow:
    def __init__(self):
        self.drop_cols = {"CompNo"}
        self.feature_cols = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.use_scaler = False
        self.best_params_ = {}
        self.chain = None

    def _resolve_target_columns(self, columns):
        for prefix in TARGET_PREFIX_CANDIDATES:
            target_cols = [f"{prefix}{h}" for h in HORIZONS]
            if all(col in columns for col in target_cols):
                return target_cols
        raise ValueError("Could not find expected target columns in the dataset.")

    @staticmethod
    def _to_float(value):
        if value is None:
            return np.nan
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return float(value)
        except Exception:
            text = str(value).strip().lower()
            if text in {"", "nan", "none", "null"}:
                return np.nan
            if text == "true":
                return 1.0
            if text == "false":
                return 0.0
            try:
                return float(text)
            except Exception:
                return np.nan

    def _prepare_matrix(self, X, fit=False):
        if hasattr(X, "columns"):
            if fit or self.feature_cols is None:
                cols = [c for c in X.columns if (c not in self.drop_cols and not c.startswith("y_"))]
                self.feature_cols = cols if cols else list(X.columns)
            cols = [c for c in self.feature_cols if c in X.columns]
            frame = X[cols].copy() if cols else X.copy()
            for col in frame.columns:
                frame[col] = [self._to_float(v) for v in frame[col].values]

            if "mm" in frame.columns:
                mm = np.clip(frame["mm"].astype(float), 1.0, 12.0)
                frame["mm_sin"] = np.sin(2.0 * np.pi * mm / 12.0)
                frame["mm_cos"] = np.cos(2.0 * np.pi * mm / 12.0)

            for col in ["dtdlevel", "dtdtrend", "DTDmedianNonFin", "sigma", "m2b", "ni2talevel", "liqnonfinlevel"]:
                if col in frame.columns:
                    v = frame[col].astype(float)
                    frame[f"{col}_logabs"] = np.sign(v) * np.log1p(np.abs(v))

            if "dtdlevel" in frame.columns and "dtdtrend" in frame.columns:
                frame["dtd_interaction"] = frame["dtdlevel"].astype(float) * frame["dtdtrend"].astype(float)

            arr = frame.to_numpy(dtype=float)
            columns = frame.columns.tolist()
        else:
            arr = np.asarray(X, dtype=float)
            columns = None

        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(arr.shape[0], 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        arr[~np.isfinite(arr)] = np.nan
        return arr, columns

    def _transform_features(self, X, fit=False):
        arr, columns = self._prepare_matrix(X, fit=fit)
        if fit:
            arr = self.imputer.fit_transform(arr)
            if self.use_scaler:
                arr = self.scaler.fit_transform(arr)
        else:
            arr = self.imputer.transform(arr)
            if self.use_scaler:
                arr = self.scaler.transform(arr)
        return pd.DataFrame(arr, columns=columns, index=X.index)

    def preprocess_data(self, data, fit=True):
        data_proc = data.copy()
        target_cols = self._resolve_target_columns(data_proc.columns.tolist())
        feature_frame = data_proc.drop(columns=target_cols)
        X = self._transform_features(feature_frame, fit=fit)
        y = data_proc[target_cols].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0).astype(int)
        return X, y

    def _create_model(self, params=None):
        merged = {**BASE_PARAMS, **(params or {})}
        return cb.CatBoostClassifier(**merged)

    def _suggest_params(self, trial):
        return {
            "iterations": trial.suggest_int("iterations", 100, 250, step=50),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 128),
        }

    @staticmethod
    def _default_auc(y_true, y_pred_proba, default_class=1):
        y_true = np.asarray(y_true).astype(int)
        y_binary = (y_true == int(default_class)).astype(int)
        if np.unique(y_binary).size < 2:
            return float("nan")
        return float(roc_auc_score(y_binary, y_pred_proba[:, default_class]))

    @staticmethod
    def _multiclass_auc(y_true, y_pred_proba):
        return float(
            roc_auc_score(
                y_true=np.asarray(y_true),
                y_score=np.asarray(y_pred_proba),
                labels=CLASSES,
                multi_class="ovr",
                average="macro",
            )
        )

    def _score_chain(self, chain, X, y):
        predictions = chain.predict_proba(X)
        auc_scores = []
        for idx, horizon in enumerate(HORIZONS):
            y_true = y.iloc[:, idx]
            score = self._default_auc(y_true, predictions[horizon])
            if np.isfinite(score):
                auc_scores.append(score)
        if not auc_scores:
            raise ValueError("Unable to compute AUC on validation split.")
        return float(np.mean(auc_scores))

    def tune_hyperparameters(self, X, y, n_trials=3, validation_size=0.2, tuning_sample_size=DEFAULT_TUNING_SAMPLE_SIZE):
        if tuning_sample_size is not None and len(X) > tuning_sample_size:
            sample_fraction = tuning_sample_size / len(X)
            stratify_sample = y.iloc[:, 3] if y.shape[1] >= 4 and y.iloc[:, 3].nunique() > 1 else None
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=sample_fraction,
                random_state=RANDOM_STATE,
                stratify=stratify_sample,
            )

        stratify_col = y.iloc[:, 3] if y.shape[1] >= 4 and y.iloc[:, 3].nunique() > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=RANDOM_STATE,
            stratify=stratify_col,
        )

        def objective(trial):
            params = self._suggest_params(trial)
            chain = ClassifierChain(self._create_model(params), HORIZONS)
            chain.fit(X_train, y_train)
            score = self._score_chain(chain, X_val, y_val)
            trial.set_user_attr("params", params)
            return score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params_ = study.best_trial.user_attrs["params"]
        return self.best_params_

    def fit(self, X, y, tune_hyperparameters=False, n_trials=3, validation_size=0.2, tuning_sample_size=DEFAULT_TUNING_SAMPLE_SIZE):
        params = {}
        if tune_hyperparameters:
            params = self.tune_hyperparameters(
                X,
                y,
                n_trials=n_trials,
                validation_size=validation_size,
                tuning_sample_size=tuning_sample_size,
            )

        self.chain = ClassifierChain(self._create_model(params), HORIZONS)
        self.chain.fit(X, y)
        return self

    def predict_probabilities(self, X):
        predictions_proba = self.chain.predict_proba(X)
        output = {}
        for horizon in HORIZONS:
            for cls in CLASSES:
                output[f"prob_{horizon}_class_{cls}"] = predictions_proba[horizon][:, cls]
        return pd.DataFrame(output, index=X.index)

    def evaluate_probabilities(self, X, y):
        predictions = self.chain.predict_proba(X)
        results = []
        for idx, horizon in enumerate(HORIZONS):
            y_true = y.iloc[:, idx]
            y_pred_proba = predictions[horizon]
            y_pred = np.asarray(CLASSES)[np.argmax(y_pred_proba, axis=1)]
            results.append(
                {
                    "Model": "CC_CatBoost",
                    "Horizon": horizon,
                    "Default AUC": self._default_auc(y_true, y_pred_proba),
                    "Multiclass AUC Macro": self._multiclass_auc(y_true, y_pred_proba) if y_true.nunique() > 1 else np.nan,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                    "Log Loss": log_loss(y_true, y_pred_proba, labels=CLASSES),
                }
            )
        return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the CatBoost classifier chain.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH, help="Path to the training CSV/XLSX file.")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH, help="Path to the test CSV/XLSX file.")
    parser.add_argument("--no-tuning", action="store_true", help="Disable Optuna hyperparameter tuning.")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of Optuna trials.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation split size used during tuning.")
    parser.add_argument(
        "--tuning-sample-size",
        type=int,
        default=DEFAULT_TUNING_SAMPLE_SIZE,
        help="Number of training rows to use during tuning.",
    )
    parser.add_argument("--save-results", default=DEFAULT_RESULTS_PATH, help="Optional output CSV path for evaluation results.")
    return parser.parse_args()


def load_table(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type for {path}. Use CSV or Excel.")


def main():
    args = parse_args()
    print("Loading datasets...")
    train_df = load_table(args.train_path)
    test_df = load_table(args.test_path)
    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")

    workflow = CatBoostWorkflow()
    print("\nPreprocessing training data...")
    X_train, y_train = workflow.preprocess_data(train_df, fit=True)
    print("Preprocessing test data...")
    X_test, y_test = workflow.preprocess_data(test_df, fit=False)

    print("\nTraining CatBoost classifier chain...")
    workflow.fit(
        X_train,
        y_train,
        tune_hyperparameters=not args.no_tuning,
        n_trials=args.n_trials,
        validation_size=args.validation_size,
        tuning_sample_size=args.tuning_sample_size,
    )

    if workflow.best_params_:
        print("\nBest hyperparameters:")
        print(workflow.best_params_)

    print("\nGenerating probability forecasts on held-out test set...")
    probability_results = workflow.predict_probabilities(X_test)
    metric_results = workflow.evaluate_probabilities(X_test, y_test)
    print("\nProbability-based evaluation summary:")
    print(metric_results.to_string(index=False))

    output_df = pd.DataFrame(index=test_df.index)
    for col in ID_COLUMNS:
        if col in test_df.columns:
            output_df[col] = test_df[col]
    output_df = pd.concat([output_df, probability_results], axis=1)

    print("\nSample probability forecasts:")
    print(output_df.head().to_string(index=False))

    if args.save_results:
        output_df.to_csv(args.save_results, index=False)
        print(f"\nSaved results to {args.save_results}")


if __name__ == "__main__":
    main()
