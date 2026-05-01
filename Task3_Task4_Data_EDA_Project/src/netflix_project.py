from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .config import ensure_output_dirs, find_csv
except ImportError:
    from config import ensure_output_dirs, find_csv


RAW_FILE = "netflix_titles.csv"


def load_netflix_data(project_root: Path) -> pd.DataFrame:
    raw_path = find_csv(project_root / "data" / "raw", RAW_FILE, "netflix")
    return pd.read_csv(raw_path)


def clean_and_engineer_netflix(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["date_added"] = pd.to_datetime(cleaned["date_added"].astype(str).str.strip(), errors="coerce")
    for column in ["director", "cast", "country", "rating", "listed_in"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown")

    cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")
    cleaned["release_decade"] = (cleaned["release_year"] // 10 * 10).astype("Int64").astype(str) + "s"
    cleaned["content_type"] = cleaned["type"]
    cleaned["is_movie"] = (cleaned["type"] == "Movie").astype(int)
    cleaned["year_added"] = cleaned["date_added"].dt.year
    cleaned["month_added"] = cleaned["date_added"].dt.month

    return cleaned


def explode_multilabel(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].dropna().str.split(",").explode().str.strip()


def save_netflix_visuals(cleaned_df: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    type_counts = cleaned_df["type"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Set2"))
    plt.title("Netflix Content Type Share")
    plt.tight_layout()
    plt.savefig(figures_dir / "netflix_movie_tv_pie.png", dpi=150)
    plt.close()

    yearly = cleaned_df.dropna(subset=["release_year"]).groupby("release_year").size().reset_index(name="content_count")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=yearly, x="release_year", y="content_count", marker="o")
    plt.title("Content Released per Year")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.tight_layout()
    plt.savefig(figures_dir / "netflix_release_year_lineplot.png", dpi=150)
    plt.close()

    genres = explode_multilabel(cleaned_df, "listed_in").value_counts().head(10).reset_index()
    genres.columns = ["genre", "count"]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=genres, x="count", y="genre", hue="genre", palette="Set3", legend=False)
    plt.title("Top 10 Netflix Genres")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(figures_dir / "netflix_top10_genres_barplot.png", dpi=150)
    plt.close()

    countries = explode_multilabel(cleaned_df, "country")
    top_countries = countries[countries != "Unknown"].value_counts().head(10).index
    country_rows = cleaned_df.assign(country_split=cleaned_df["country"].str.split(",")).explode("country_split")
    country_rows["country_split"] = country_rows["country_split"].str.strip()
    heatmap_data = (
        country_rows[country_rows["country_split"].isin(top_countries)]
        .pivot_table(index="country_split", columns="type", values="show_id", aggfunc="count", fill_value=0)
    )
    heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(7, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Top Countries by Content Volume")
    plt.xlabel("Content Type")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(figures_dir / "netflix_country_content_heatmap.png", dpi=150)
    plt.close()

    rating_counts = cleaned_df["rating"].value_counts().head(10).reset_index()
    rating_counts.columns = ["rating", "count"]
    plt.figure(figsize=(8, 4))
    sns.barplot(data=rating_counts, x="rating", y="count", hue="rating", palette="Set2", legend=False)
    plt.title("Top Content Ratings")
    plt.tight_layout()
    plt.savefig(figures_dir / "netflix_ratings_barplot.png", dpi=150)
    plt.close()


def netflix_insights(cleaned_df: pd.DataFrame) -> list[str]:
    insights = []

    type_counts = cleaned_df["type"].value_counts()
    top_type = type_counts.idxmax()
    insights.append(f"{top_type} titles dominate the catalog with {type_counts.max()} records.")

    top_country = explode_multilabel(cleaned_df, "country")
    top_country = top_country[top_country != "Unknown"].value_counts().idxmax()
    insights.append(f"{top_country} is the most frequent production country, showing strong catalog representation.")

    top_genre = explode_multilabel(cleaned_df, "listed_in").value_counts().idxmax()
    insights.append(f"{top_genre} is the most common listed genre/category.")

    busiest_year = cleaned_df["release_year"].value_counts().idxmax()
    insights.append(f"The busiest release year in the dataset is {int(busiest_year)}.")

    added_by_year = cleaned_df.dropna(subset=["year_added"]).groupby("year_added").size()
    if len(added_by_year) >= 3:
        recent = added_by_year.sort_index().tail(5)
        x = recent.index.to_numpy(dtype=float)
        y = recent.values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        next_year = int(x.max() + 1)
        second_year = next_year + 1
        pred_one = max(0, slope * next_year + intercept)
        pred_two = max(0, slope * second_year + intercept)
        insights.append(
            f"Simple trend prediction estimates about {pred_one:.0f} additions in {next_year} and {pred_two:.0f} in {second_year}."
        )

    insights.append("What this means: Netflix catalog decisions can be guided by country, genre, and release-year patterns.")
    return insights[:6]


def run_netflix_project(project_root: Path | None = None) -> pd.DataFrame:
    project_root = project_root or Path(__file__).resolve().parents[1]
    processed_dir, figures_dir = ensure_output_dirs(project_root)

    raw_df = load_netflix_data(project_root)
    print("\nData info:")
    raw_df.info()
    print("\nData description:")
    print(raw_df.describe(include="all"))

    cleaned_df = clean_and_engineer_netflix(raw_df)
    cleaned_df.to_csv(processed_dir / "netflix_titles_cleaned.csv", index=False)
    save_netflix_visuals(cleaned_df, figures_dir)

    insights = netflix_insights(cleaned_df)
    report = ["# Netflix Titles: Insights and Interpretation", ""]
    report.extend([f"{i}. {insight}" for i, insight in enumerate(insights, start=1)])
    (project_root / "reports" / "netflix_insights.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Saved cleaned Netflix data to {processed_dir / 'netflix_titles_cleaned.csv'}")
    return cleaned_df


if __name__ == "__main__":
    run_netflix_project()
