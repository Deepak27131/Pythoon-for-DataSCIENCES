from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .config import ensure_output_dirs, find_csv
except ImportError:
    from config import ensure_output_dirs, find_csv


RAW_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_churn_data(project_root: Path) -> pd.DataFrame:
    raw_path = find_csv(project_root / "data" / "raw", RAW_FILE, "churn")
    return pd.read_csv(raw_path)


def clean_and_engineer_churn(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned.replace(" ", np.nan)
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")

    cleaned = cleaned.dropna(subset=["TotalCharges", "tenure", "MonthlyCharges", "Churn"])
    cleaned = cleaned[cleaned["tenure"] >= 0].copy()

    tenure_bins = [0, 12, 24, 36, 48, 60, 72]
    tenure_labels = ["0-12 months", "13-24 months", "25-36 months", "37-48 months", "49-60 months", "61-72 months"]
    cleaned["TenureGroup"] = pd.cut(
        cleaned["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
        right=True,
    )

    cleaned["AvgMonthlySpend"] = np.where(
        cleaned["tenure"] > 0,
        cleaned["TotalCharges"] / cleaned["tenure"],
        cleaned["MonthlyCharges"],
    )

    yes_no_columns = []
    for column in cleaned.columns:
        values = set(cleaned[column].dropna().astype(str).unique())
        if values and values.issubset({"Yes", "No"}):
            yes_no_columns.append(column)

    for column in yes_no_columns:
        cleaned[column] = cleaned[column].map({"Yes": 1, "No": 0})

    if "gender" in cleaned.columns:
        cleaned["gender"] = cleaned["gender"].map({"Female": 1, "Male": 0})

    encoded_columns = [
        column
        for column in ["Contract", "InternetService", "PaymentMethod", "TenureGroup"]
        if column in cleaned.columns
    ]
    cleaned = pd.get_dummies(cleaned, columns=encoded_columns, drop_first=True, dtype=int)

    return cleaned


def save_churn_visuals(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=raw_df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Customer Churn Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_countplot.png", dpi=150)
    plt.close()

    if {"Contract", "Churn"}.issubset(raw_df.columns):
        plt.figure(figsize=(8, 4))
        contract_churn = raw_df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()).reset_index()
        contract_churn["ChurnRate"] = contract_churn["Churn"] * 100
        sns.barplot(data=contract_churn, x="Contract", y="ChurnRate", hue="Contract", palette="Set3", legend=False)
        plt.ylabel("Churn Rate (%)")
        plt.title("Churn Rate by Contract Type")
        plt.tight_layout()
        plt.savefig(figures_dir / "churn_contract_barplot.png", dpi=150)
        plt.close()

    numeric_df = cleaned_df.select_dtypes(include=["number"])
    plt.figure(figsize=(11, 8))
    corr = numeric_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.2)
    plt.title("Customer Churn Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=raw_df, x="Churn", y="MonthlyCharges", hue="Churn", palette="Set2", legend=False)
    plt.title("Monthly Charges by Churn")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_monthlycharges_boxplot.png", dpi=150)
    plt.close()


def churn_insights(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> list[str]:
    insights = []
    churn_rate = (raw_df["Churn"] == "Yes").mean() * 100
    insights.append(f"Overall churn rate is {churn_rate:.1f}%, showing the share of customers who left.")

    if "Contract" in raw_df.columns:
        contract_rates = raw_df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False)
        insights.append(f"Highest churn appears in the {contract_rates.index[0]} contract group.")

    avg_churned_charge = raw_df.loc[raw_df["Churn"] == "Yes", "MonthlyCharges"].mean()
    avg_retained_charge = raw_df.loc[raw_df["Churn"] == "No", "MonthlyCharges"].mean()
    insights.append(
        f"Churned customers average {avg_churned_charge:.2f} monthly charges versus {avg_retained_charge:.2f} for retained customers."
    )

    if "tenure" in raw_df.columns:
        tenure_churned = raw_df.loc[raw_df["Churn"] == "Yes", "tenure"].mean()
        tenure_retained = raw_df.loc[raw_df["Churn"] == "No", "tenure"].mean()
        insights.append(f"Churned customers have lower average tenure ({tenure_churned:.1f}) than retained customers ({tenure_retained:.1f}).")

    if "AvgMonthlySpend" in cleaned_df.columns:
        insights.append("Average monthly spend was engineered from total charges and tenure to compare customer value more fairly.")

    return insights[:5]


def run_churn_project(project_root: Path | None = None) -> pd.DataFrame:
    project_root = project_root or Path(__file__).resolve().parents[1]
    processed_dir, figures_dir = ensure_output_dirs(project_root)

    raw_df = load_churn_data(project_root)
    print("\nData info:")
    raw_df.info()
    print("\nData description:")
    print(raw_df.describe(include="all"))

    cleaned_df = clean_and_engineer_churn(raw_df)
    cleaned_df.to_csv(processed_dir / "customer_churn_cleaned.csv", index=False)
    save_churn_visuals(raw_df, cleaned_df, figures_dir)

    insights = churn_insights(raw_df, cleaned_df)
    report = ["# Customer Churn: Top 5 Insights", ""]
    report.extend([f"{i}. {insight}" for i, insight in enumerate(insights, start=1)])
    (project_root / "reports" / "churn_insights.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Saved cleaned churn data to {processed_dir / 'customer_churn_cleaned.csv'}")
    return cleaned_df


if __name__ == "__main__":
    run_churn_project()
