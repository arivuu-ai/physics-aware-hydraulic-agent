"""
waterops/train.py — Component C: Train & Evaluate Models
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_COL = "risk_in_next_120s"
META_COLS = {"incident_id", "tsms", LABEL_COL}
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Threshold Selection
# ---------------------------------------------------------------------------

def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """Choose an operating threshold from the Precision-Recall curve."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    precision_arr = precision_arr[:-1]
    recall_arr = recall_arr[:-1]

    f1_arr = np.where(
        (precision_arr + recall_arr) > 0,
        2 * precision_arr * recall_arr / (precision_arr + recall_arr),
        0.0,
    )

    if strategy == "f1":
        idx = np.argmax(f1_arr)
    elif strategy == "high_recall":
        mask = recall_arr >= 0.90
        if mask.any():
            candidates = np.where(mask)[0]
            idx = candidates[np.argmax(precision_arr[candidates])]
        else:
            idx = np.argmax(recall_arr)
    elif strategy == "high_prec":
        mask = precision_arr >= 0.90
        if mask.any():
            candidates = np.where(mask)[0]
            idx = candidates[np.argmax(recall_arr[candidates])]
        else:
            idx = np.argmax(precision_arr)
    else:
        idx = np.argmax(f1_arr)

    chosen_t = float(thresholds[idx])
    return chosen_t, {
        "precision": float(precision_arr[idx]),
        "recall": float(recall_arr[idx]),
        "f1": float(f1_arr[idx]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute a comprehensive set of metrics for imbalanced classification."""
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "threshold": round(threshold, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"positive": int(y_true.sum()), "negative": int((1 - y_true).sum())},
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class WaterOpsTrainer:
    def __init__(
        self,
        model_dir: str = "./artifacts",
        threshold_strategy: str = "f1",
        log_level: str = "INFO",
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_strategy = threshold_strategy
        self.logger = _setup_logger("waterops.train", log_level)

    def _load_training_table(self, data_path: str) -> pd.DataFrame:
        p = Path(data_path)
        candidates = [
            p / "training_table.parquet",
            p,
        ]
        for c in candidates:
            if c.exists() and (c.is_file() or (c / "training_table.parquet").exists()):
                target = c if c.is_file() else c / "training_table.parquet"
                df = pd.read_parquet(target)
                self.logger.info(
                    f"Loaded training table: {len(df)} rows from {target}"
                )
                return df
        raise FileNotFoundError(f"No training table found at {data_path}")

    def _temporal_split(
        self, df: pd.DataFrame, train_frac: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_parts, test_parts = [], []
        for incident_id, grp in df.groupby("incident_id"):
            grp = grp.sort_values("tsms")  # updated from ts_ms
            split_idx = int(len(grp) * train_frac)
            train_parts.append(grp.iloc[:split_idx])
            test_parts.append(grp.iloc[split_idx:])
        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        self.logger.info(
            f"Temporal split: {len(train_df)} train / {len(test_df)} test"
        )
        return train_df, test_df

    def _prepare_Xy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        if LABEL_COL not in df.columns:
            raise ValueError(f"Label column '{LABEL_COL}' not found in data")
        feature_cols = sorted([
            c for c in df.columns
            if c not in META_COLS
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ])
        if not feature_cols:
            raise ValueError("No numeric feature columns found")
        X = df[feature_cols]
        y = df[LABEL_COL].astype(int).values
        pos_rate = y.mean()
        self.logger.info(
            f"Features: {len(feature_cols)} | "
            f"Label distribution: {pos_rate:.1%} positive, {1-pos_rate:.1%} negative"
        )
        if pos_rate < 0.01 or pos_rate > 0.99:
            self.logger.warning(
                f"Extreme class imbalance detected ({pos_rate:.3%} positive). "
                "Model may struggle — consider adjusting threshold strategy."
            )
        return X, y, feature_cols

    def _build_baseline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_SEED,
            )),
        ])

    def _build_xgboost(self, pos_weight: float) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=pos_weight,
                eval_metric="aucpr",
                early_stopping_rounds=30,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
            )),
        ])

    def train_and_evaluate(self, data_path: str) -> Dict[str, Any]:
        df = self._load_training_table(data_path)
        train_df, test_df = self._temporal_split(df)
        X_train, y_train, feature_cols = self._prepare_Xy(train_df)
        X_test, y_test, _ = self._prepare_Xy(test_df)
        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)
        pos_weight = n_neg / n_pos
        self.logger.info(f"scale_pos_weight = {pos_weight:.2f} (neg/pos ratio)")

        self.logger.info("Training baseline (Logistic Regression)...")
        baseline = self._build_baseline()
        baseline.fit(X_train, y_train)
        baseline_probs = baseline.predict_proba(X_test)[:, 1]
        baseline_threshold, _ = select_threshold(
            y_test, baseline_probs, self.threshold_strategy
        )
        baseline_metrics = evaluate_model(y_test, baseline_probs, baseline_threshold)
        self.logger.info(f"Baseline metrics: {json.dumps(baseline_metrics, indent=2)}")

        self.logger.info("Training improved model (XGBoost)...")
        xgb_pipeline = self._build_xgboost(pos_weight)
        imputer = xgb_pipeline.named_steps["imputer"]
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        xgb_model = xgb_pipeline.named_steps["model"]
        xgb_model.fit(
            X_train_imp,
            y_train,
            eval_set=[(X_test_imp, y_test)],
            verbose=False,
        )
        xgb_pipeline.named_steps["imputer"] = imputer
        xgb_probs = xgb_model.predict_proba(X_test_imp)[:, 1]

        xgb_threshold, _ = select_threshold(
            y_test, xgb_probs, self.threshold_strategy
        )
        xgb_metrics = evaluate_model(y_test, xgb_probs, xgb_threshold)
        self.logger.info(f"XGBoost metrics: {json.dumps(xgb_metrics, indent=2)}")

        results = {
            "baseline_logistic_regression": baseline_metrics,
            "improved_xgboost": xgb_metrics,
        }

        if xgb_metrics["pr_auc"] >= baseline_metrics["pr_auc"]:
            best_name = "xgboost"
            best_pipeline = xgb_pipeline
            best_threshold = xgb_threshold
            best_metrics = xgb_metrics
        else:
            best_name = "logistic_regression"
            best_pipeline = baseline
            best_threshold = baseline_threshold
            best_metrics = baseline_metrics
            self.logger.warning(
                "Baseline outperformed XGBoost on PR-AUC — saving baseline as best model."
            )

        self._save_artifact(
            pipeline=best_pipeline,
            model_name=best_name,
            feature_cols=feature_cols,
            threshold=best_threshold,
            metrics=best_metrics,
            all_results=results,
        )
        self._log_comparison(results)
        return results

    def _save_artifact(
        self,
        pipeline: Pipeline,
        model_name: str,
        feature_cols: List[str],
        threshold: float,
        metrics: Dict,
        all_results: Dict,
    ):
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_file = self.model_dir / f"model_{version}.pkl"
        meta_file = self.model_dir / f"metadata_{version}.json"

        joblib.dump(pipeline, model_file)

        metadata = {
            "version": version,
            "model_type": model_name,
            "threshold": round(threshold, 4),
            "threshold_strategy": self.threshold_strategy,
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "metrics": metrics,
            "all_model_results": all_results,
            "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "threshold_tradeoff_note": (
                "In firefighting, a missed instability event (false negative) "
                "is far more dangerous than a false alarm (false positive). "
                f"Threshold {round(threshold, 4)} was selected via '{self.threshold_strategy}' "
                "strategy on the Precision-Recall curve. To favor recall, "
                "lower the threshold or re-train with strategy='high_recall'."
            ),
        }
        meta_file.write_text(json.dumps(metadata, indent=2))

        latest_model = self.model_dir / "model_latest.pkl"
        latest_meta = self.model_dir / "metadata_latest.json"
        if latest_model.exists():
            latest_model.unlink()
        if latest_meta.exists():
            latest_meta.unlink()
        import shutil
        shutil.copy2(model_file, latest_model)
        shutil.copy2(meta_file, latest_meta)

        self.logger.info(f"Model saved → {model_file}")
        self.logger.info(f"Metadata saved → {meta_file}")
        self.logger.info(
            "Latest copies updated → model_latest.pkl, metadata_latest.json"
        )

    def _log_comparison(self, results: Dict):
        self.logger.info("=" * 60)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("=" * 60)
        header = f"{'Model':<30} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'MCC':>8} {'Thresh':>8}"
        self.logger.info(header)
        self.logger.info("-" * 60)
        for name, m in results.items():
            row = (
                f"{name:<30} "
                f"{m['pr_auc']:>8.4f} "
                f"{m['roc_auc']:>8.4f} "
                f"{m['f1_score']:>8.4f} "
                f"{m['mcc']:>8.4f} "
                f"{m['threshold']:>8.4f}"
            )
            self.logger.info(row)
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="WaterOps Component C — Train & Evaluate Models"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to training table directory (output of Component B)",
    )
    parser.add_argument(
        "--model-out", default="./artifacts",
        help="Path to save model artifacts (default: ./artifacts)",
    )
    parser.add_argument(
        "--threshold-strategy", default="f1",
        choices=["f1", "high_recall", "high_prec"],
        help="Threshold selection strategy (default: f1)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    trainer = WaterOpsTrainer(
        model_dir=args.model_out,
        threshold_strategy=args.threshold_strategy,
        log_level=args.log_level,
    )
    trainer.train_and_evaluate(data_path=args.data)


if __name__ == "__main__":
    main()
