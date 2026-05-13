"""
train_model.py
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Evaluates all models, saves the best one, and generates diagnostic plots.
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from preprocessing import run_pipeline, FEATURE_COLS

RANDOM_STATE = 42
DATA_PATH = "data/diabetes.csv"
MODEL_SAVE_PATH = "models/best_model.pkl"
SCALER_SAVE_PATH = "models/scaler.pkl"
IMAGES_DIR = "images"


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }


def print_results_table(results: dict):
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("=" * 70)
    for name, m in results.items():
        print(
            f"{name:<25} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['roc_auc']:>6.3f}"
        )
    print("=" * 70)


def plot_confusion_matrices(results: dict, y_test, save_dir: str):
    for name, m in results.items():
        cm = confusion_matrix(y_test, m["y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            ax=ax
        )
        ax.set_title(f'Confusion Matrix — {name}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        fig.tight_layout()
        filename = f"cm_{name.lower().replace(' ', '_')}.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close(fig)
        print(f"Saved: {filename}")


def plot_roc_curves(results: dict, y_test, save_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for (name, m), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150)
    plt.close(fig)
    print("Saved: roc_curves.png")


def plot_feature_importance(model, save_dir: str):
    """Plot feature importance from the Random Forest model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_features[::-1], sorted_importance[::-1],
                   color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlim(0, sorted_importance.max() * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150)
    plt.close(fig)
    print("Saved: feature_importance.png")


def select_best_model(models: dict, results: dict) -> tuple:
    """Return the model with the highest F1-score."""
    best_name = max(results, key=lambda n: results[n]["f1"])
    return best_name, models[best_name]


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, _ = run_pipeline(DATA_PATH, SCALER_SAVE_PATH)

    print("\nTraining models...")
    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"  Fitting {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    print_results_table(results)

    for name, m in results.items():
        print(f"\n--- {name} ---")
        print(classification_report(y_test, m["y_pred"],
                                    target_names=["No Diabetes", "Diabetes"]))

    print("\nGenerating plots...")
    plot_confusion_matrices(results, y_test, IMAGES_DIR)
    plot_roc_curves(results, y_test, IMAGES_DIR)

    best_name, best_model = select_best_model(models, results)
    plot_feature_importance(best_model, IMAGES_DIR)

    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"\nBest model: {best_name}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
