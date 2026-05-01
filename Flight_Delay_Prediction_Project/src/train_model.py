from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Airlines.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "flight_delay_best_model.pkl"
METRICS_PATH = REPORT_DIR / "metrics.json"
REPORT_PATH = REPORT_DIR / "model_evaluation_report.md"

RANDOM_STATE = 42
MISSING_VALUE_THRESHOLD = 0.45
TARGET_COLUMN = "Delay"


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_and_prepare_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)

    missing_ratio = df.isna().mean()
    columns_to_drop = missing_ratio[missing_ratio > MISSING_VALUE_THRESHOLD].index.tolist()
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    if "FlightDate" in df.columns:
        flight_date = pd.to_datetime(df["FlightDate"], errors="coerce")
        df["FlightWeekday"] = flight_date.dt.dayofweek
        df["FlightMonth"] = flight_date.dt.month
        df = df.drop(columns=["FlightDate"])

    rename_map = {}
    if "Origin" in df.columns and "AirportFrom" not in df.columns:
        rename_map["Origin"] = "AirportFrom"
    if "Dest" in df.columns and "AirportTo" not in df.columns:
        rename_map["Dest"] = "AirportTo"
    if rename_map:
        df = df.rename(columns=rename_map)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in {data_path}.")

    df = df.drop_duplicates()
    y = df[TARGET_COLUMN].astype(int)
    x = df.drop(columns=[TARGET_COLUMN])

    id_like_columns = [column for column in ["id", "Flight"] if column in x.columns]
    if id_like_columns:
        x = x.drop(columns=id_like_columns)

    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_features = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = x.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        max_depth=18,
                        min_samples_leaf=4,
                        n_estimators=80,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def save_figures(best_model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    predictions = best_model.predict(x_test)
    probabilities = best_model.predict_proba(x_test)[:, 1]

    plt.figure(figsize=(7, 5))
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["On Time", "Delayed"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Best Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve.png", dpi=150)
    plt.close()


def write_report(
    metrics: dict[str, dict],
    best_model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(model_metrics["accuracy"], 4),
                "Precision": round(model_metrics["precision"], 4),
                "Recall": round(model_metrics["recall"], 4),
                "ROC AUC": round(model_metrics["roc_auc"], 4),
            }
        )

    metrics_table = pd.DataFrame(rows).to_markdown(index=False)
    best_report = metrics[best_model_name]["classification_report"]
    best_confusion_matrix = metrics[best_model_name]["confusion_matrix"]

    report = f"""# Flight Delay Prediction - Model Evaluation Report

## Project Objective
Predict whether a flight will be delayed or on time using airline, route, and flight schedule data.

## Dataset Summary
- Source file: `data/raw/Airlines.csv`
- Total rows after cleaning duplicates: {len(x):,}
- Feature columns used: {x.shape[1]}
- Target column: `{TARGET_COLUMN}`
- On-time records: {(y == 0).sum():,}
- Delayed records: {(y == 1).sum():,}

## Preprocessing
- Dropped columns with more than {MISSING_VALUE_THRESHOLD:.0%} missing values.
- Converted `FlightDate` to weekday and month if that column exists.
- Encoded categorical route and airline columns with OneHotEncoder.
- Imputed missing numeric values with median and categorical values with most frequent value.
- Dropped ID-like columns: `id`, `Flight` when present.

## Model Performance
{metrics_table}

## Best Model
The selected model is **{best_model_name}**, chosen by the highest ROC AUC score.

## Best Model Confusion Matrix
```text
{best_confusion_matrix}
```

## Best Model Classification Report
```text
{best_report}
```

## Generated Files
- Saved model: `models/flight_delay_best_model.pkl`
- Metrics JSON: `reports/metrics.json`
- Confusion matrix: `reports/figures/confusion_matrix.png`
- ROC curve: `reports/figures/roc_curve.png`
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_and_prepare_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = get_models(build_preprocessor(x))
    metrics = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        metrics[model_name] = evaluate_model(model, x_test, y_test)
        print(
            f"{model_name}: "
            f"accuracy={metrics[model_name]['accuracy']:.4f}, "
            f"precision={metrics[model_name]['precision']:.4f}, "
            f"recall={metrics[model_name]['recall']:.4f}, "
            f"roc_auc={metrics[model_name]['roc_auc']:.4f}"
        )

    best_model_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    best_model = models[best_model_name]

    joblib.dump(best_model, MODEL_PATH)
    serializable_metrics = {
        model_name: {
            key: value
            for key, value in model_metrics.items()
            if key != "classification_report"
        }
        for model_name, model_metrics in metrics.items()
    }
    serializable_metrics["best_model"] = best_model_name
    METRICS_PATH.write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")

    save_figures(best_model, x_test, y_test)
    write_report(metrics, best_model_name, x, y)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
