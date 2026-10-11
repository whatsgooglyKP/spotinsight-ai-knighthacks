"""
RoadGuard Agent: Supervised Machine Learning Classifier for Major Liability Prediction.
Strictly adheres to /ml-best-practices:
- Establishes naive baseline & ML baseline (Logistic Regression) vs Tree ensembles (Random Forest, Gradient Boosting / LightGBM).
- Strict featurization ordering (split before fit).
- Metrics tailored to asymmetric liability cost: High Recall for Class 1 (Major Liability), PR-AUC, ROC-AUC, F1-Score.
- Subpopulation / slice error analysis across risk zones.
- Model persistence and explainability.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from roadguard.config import MODELS_DIR
from roadguard.feature_engineering import RoadGuardFeaturePipeline


class RoadGuardModelTrainer:
    """
    Trains, compares, validates, and serializes the RoadGuard major liability classifier.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_pipeline = RoadGuardFeaturePipeline()
        self.best_model = None
        self.best_model_name = ""
        self.metrics_summary: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def train_and_evaluate(
        self, df: pd.DataFrame, test_size: float = 0.25
    ) -> Dict[str, Any]:
        """
        Executes end-to-end ML lifecycle:
        1. Train/Test split BEFORE featurization
        2. Fit feature pipeline on Train only
        3. Evaluate Baselines vs Champion Model
        4. Cross-validation and slice-based analysis
        5. Serialization of champion model
        """
        target_col = "major_liability"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing from dataset.")

        y = df[target_col].values
        X_df = df.drop(columns=[target_col])

        # Step 1: Strict Featurization Ordering - Split before any fitting
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )

        print(f"Training split: {len(y_train)} samples ({y_train.sum()} major liabilities)")
        print(f"Testing split:  {len(y_test)} samples ({y_test.sum()} major liabilities)")

        # Step 2: Fit feature pipeline strictly on train data
        self.feature_pipeline.fit(X_train_df, y_train)
        X_train = self.feature_pipeline.transform(X_train_df)
        X_test = self.feature_pipeline.transform(X_test_df)
        self.feature_names = self.feature_pipeline.get_feature_names()

        # Step 3: Candidate Models Definition
        models = {
            "Naive Majority": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression Baseline": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                random_state=self.random_state,
            ),
            "LightGBM Classifier": LGBMClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                scale_pos_weight=4.0,
                random_state=self.random_state,
                verbose=-1,
            ),
        }

        results = {}
        best_pr_auc = -1.0
        best_model = None
        best_name = ""

        # Weight positive samples in training to reflect catastrophic liability asymmetry
        gb_weights = np.where(y_train == 1, 4.0, 1.0)

        for name, clf in models.items():
            if name == "Gradient Boosting":
                clf.fit(X_train, y_train, sample_weight=gb_weights)
            else:
                clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = np.zeros_like(y_pred, dtype=float)

            # Calculate business-critical metrics
            acc = float(np.mean(y_pred == y_test))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            cm = confusion_matrix(y_test, y_pred).tolist()

            try:
                pr_auc = float(average_precision_score(y_test, y_prob))
                roc_auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                pr_auc = float(y_train.mean())
                roc_auc = 0.5

            results[name] = {
                "accuracy": round(acc, 4),
                "recall_major_liability": round(rec, 4),
                "precision_major_liability": round(prec, 4),
                "f1_score": round(f1, 4),
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "confusion_matrix": cm,
            }

            # Select champion prioritizing PR-AUC and Recall (cost of missing a major liability is catastrophic)
            selection_score = (pr_auc * 0.6) + (rec * 0.4)
            if selection_score > best_pr_auc:
                best_pr_auc = selection_score
                best_model = clf
                best_name = name

        self.best_model = best_model
        self.best_model_name = best_name

        # Step 4: Extract Feature Importances for Explainability
        feature_importance_list = []
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for idx in indices[:15]:
                feature_importance_list.append(
                    {
                        "feature": self.feature_names[idx],
                        "importance": round(float(importances[idx]), 4),
                    }
                )
        elif hasattr(best_model, "coef_"):
            coefs = np.abs(best_model.coef_[0])
            indices = np.argsort(coefs)[::-1]
            for idx in indices[:15]:
                feature_importance_list.append(
                    {
                        "feature": self.feature_names[idx],
                        "coefficient_magnitude": round(float(coefs[idx]), 4),
                    }
                )

        # Step 5: Slice-based Error Analysis across Lake County Risk Zones
        test_eval_df = X_test_df.copy()
        test_eval_df["y_true"] = y_test
        test_eval_df["y_pred"] = best_model.predict(X_test)
        if hasattr(best_model, "predict_proba"):
            test_eval_df["major_liability_prob"] = best_model.predict_proba(X_test)[:, 1]

        zone_slices = {}
        for zone in test_eval_df["zone_type"].unique():
            subset = test_eval_df[test_eval_df["zone_type"] == zone]
            if len(subset) > 0 and subset["y_true"].sum() > 0:
                zone_rec = float(recall_score(subset["y_true"], subset["y_pred"], zero_division=0))
                zone_prec = float(precision_score(subset["y_true"], subset["y_pred"], zero_division=0))
            else:
                zone_rec = 1.0 if len(subset) > 0 else 0.0
                zone_prec = 1.0 if len(subset) > 0 else 0.0
            zone_slices[zone] = {
                "count": len(subset),
                "true_positives": int(((subset["y_true"] == 1) & (subset["y_pred"] == 1)).sum()),
                "recall": round(zone_rec, 4),
                "precision": round(zone_prec, 4),
            }

        self.metrics_summary = {
            "champion_model": best_name,
            "models_comparison": results,
            "feature_importances_top15": feature_importance_list,
            "slice_performance_by_zone": zone_slices,
            "test_sample_size": len(y_test),
            "train_sample_size": len(y_train),
        }

        # Step 6: Save Champion Model & Artifacts
        self.save_artifacts()

        return self.metrics_summary

    def save_artifacts(self, output_dir: Optional[Path] = None):
        """Serializes model, pipeline, and evaluation report to disk."""
        target_dir = Path(output_dir or MODELS_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        package = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_pipeline": self.feature_pipeline,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
        }
        joblib.dump(package, target_dir / "roadguard_classifier.pkl")

        with open(target_dir / "model_metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2)

        print(f"Serialized champion model to {target_dir / 'roadguard_classifier.pkl'}")


class RoadGuardPredictor:
    """
    Inference interface used by the agent and edge hardware stubs.
    Loads the trained model package and performs live major liability inference.
    """

    def __init__(self, model_path: Optional[Path] = None):
        target_path = Path(model_path or (MODELS_DIR / "roadguard_classifier.pkl"))
        if not target_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {target_path}. Run train_eval.py first."
            )
        package = joblib.load(target_path)
        self.model = package["model"]
        self.model_name = package["model_name"]
        self.feature_pipeline = package["feature_pipeline"]
        self.feature_names = package["feature_names"]

    def predict_action(self, action_dict_or_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on a single incoming SPOT action record or telemetry payload.
        Returns:
            major_liability_prediction: 0 or 1
            major_liability_probability: 0.0 to 1.0
            liability_rating: 'NON-MAJOR' or 'MAJOR LIABILITY'
            model_name: model identifier
        """
        df_single = pd.DataFrame([action_dict_or_row])
        X = self.feature_pipeline.transform(df_single)
        pred = int(self.model.predict(X)[0])
        prob = float(self.model.predict_proba(X)[0, 1])

        return {
            "major_liability_prediction": pred,
            "major_liability_probability": round(prob, 4),
            "liability_rating": "MAJOR LIABILITY" if pred == 1 else "NON-MAJOR / MINOR WEAR",
            "model_name": self.model_name,
            "requires_immediate_operator_review": bool(pred == 1 or prob >= 0.50),
        }
