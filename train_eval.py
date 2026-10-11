"""
RoadGuard Agent: Training and Model Evaluation Script.
Executes supervised learning workflow strictly following /ml-best-practices:
- Train/test splitting before any transformation
- Contextual imputation & encoding
- Model comparison: Naive vs Logistic Regression vs Random Forest vs Gradient Boosting
- Recall/Precision/PR-AUC evaluation for Major Liability detection
- Subpopulation slice evaluation & feature importance visualization
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roadguard.config import DEFAULT_DATA_DIR, MODELS_DIR, OUTPUT_DIR
from roadguard.data_loader import SpotDataLoader
from roadguard.models.classifier import RoadGuardModelTrainer


def main():
    print("=" * 70)
    print(" RoadGuard Agent: Supervised Classifier Training & Evaluation")
    print(" Scoped for Lake County, FL Roadway & Infrastructure Major Liability")
    print("=" * 70)

    # 1. Load and build unified enriched dataset
    loader = SpotDataLoader(DEFAULT_DATA_DIR)
    print(f"\n[1/5] Ingesting SPOT Challenge Kit from: {DEFAULT_DATA_DIR}")
    df = loader.build_unified_dataset()
    print(f"      Total records loaded: {len(df)}")
    print(f"      Positive Class (major_liability = 1): {df['major_liability'].sum()} ({df['major_liability'].mean()*100:.2f}%)")
    print(f"      Negative Class (major_liability = 0): {(df['major_liability'] == 0).sum()} ({(1-df['major_liability'].mean())*100:.2f}%)")

    # 2. Train and evaluate models with strict featurization ordering
    print("\n[2/5] Training candidate models with strict train/test split...")
    trainer = RoadGuardModelTrainer(random_state=42)
    metrics = trainer.train_and_evaluate(df, test_size=0.25)

    print("\n[3/5] Model Comparison Results on Unseen Test Split:")
    print("-" * 75)
    print(f"{'Model':30s} | {'Acc':6s} | {'Recall':6s} | {'Prec':6s} | {'F1':6s} | {'PR-AUC':6s} | {'ROC-AUC':6s}")
    print("-" * 75)
    for name, res in metrics["models_comparison"].items():
        print(
            f"{name:30s} | {res['accuracy']:<6.4f} | {res['recall_major_liability']:<6.4f} | "
            f"{res['precision_major_liability']:<6.4f} | {res['f1_score']:<6.4f} | "
            f"{res['pr_auc']:<6.4f} | {res['roc_auc']:<6.4f}"
        )
    print("-" * 75)
    print(f"\n[CHAMPION MODEL SELECTED]: {metrics['champion_model']}")

    # 3. Print Top Feature Importances
    print("\n[4/5] Top 10 Explanatory Features for Major Liability:")
    for i, feat in enumerate(metrics["feature_importances_top15"][:10], start=1):
        score = feat.get("importance", feat.get("coefficient_magnitude", 0.0))
        print(f"  {i:2d}. {feat['feature']:35s} : {score:.4f}")

    # 4. Print Slice-based Evaluation across Risk Zones
    print("\n[5/5] Slice-Based Evaluation across Operating Risk Zones:")
    for zone, sdata in metrics["slice_performance_by_zone"].items():
        print(f"  Zone {zone:12s} -> Count: {sdata['count']:2d} | Recall: {sdata['recall']:.2f} | Precision: {sdata['precision']:.2f}")

    # 5. Generate and Save Visual Evaluation Plot
    plt.figure(figsize=(12, 5))

    # Subplot 1: Model Comparison Bar Chart
    plt.subplot(1, 2, 1)
    model_names = list(metrics["models_comparison"].keys())
    short_names = ["Naive", "LogReg", "RF", "GradBoost*", "LightGBM"][:len(model_names)]
    recalls = [metrics["models_comparison"][m]["recall_major_liability"] for m in model_names]
    pr_aucs = [metrics["models_comparison"][m]["pr_auc"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, recalls, width, label="Recall (Catch Major)", color="#D32F2F")
    plt.bar(x + width/2, pr_aucs, width, label="PR-AUC", color="#1976D2")
    plt.xticks(x, short_names, rotation=15)
    plt.ylabel("Score")
    plt.title("Model Benchmark (Prioritizing High Recall)")
    plt.legend()
    plt.ylim(0, 1.1)

    # Subplot 2: Top Feature Importances
    plt.subplot(1, 2, 2)
    top_features = metrics["feature_importances_top15"][:7]
    f_names = [f["feature"][:20] for f in reversed(top_features)]
    f_scores = [f.get("importance", f.get("coefficient_magnitude", 0.0)) for f in reversed(top_features)]
    plt.barh(f_names, f_scores, color="#388E3C")
    plt.title("Top Feature Importances (Model Interpretability)")
    plt.xlabel("Importance Weight")

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "model_evaluation_metrics.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nSaved evaluation chart to: {chart_path}")
    print("\n Training and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
