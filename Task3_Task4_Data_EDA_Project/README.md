# Task 3 and Task 4 Data Preparation + EDA Project

This folder contains two complete student projects:

1. **Task 3:** Telecom Customer Churn Data Preparation and EDA
2. **Task 4:** Netflix Titles Exploratory Analysis and Insights

Both projects are organized step by step with notebooks, reusable scripts, cleaned CSV output folders, figures, and report templates.

## Folder Structure

```text
Task3_Task4_Data_EDA_Project/
  data/
    raw/                  # Put Kaggle CSV files here
    processed/            # Cleaned CSV files are saved here
  notebooks/              # Student notebooks
  reports/
    figures/              # Generated visualizations
    comparison_slide_deck_outline.md
  src/                    # Reusable Python scripts
  run_all.py              # Runs both projects after raw CSVs are added
  requirements.txt
```

## Dataset Setup

Download both Kaggle datasets manually and place the CSV files in `data/raw/`.

Expected filenames:

- Telecom churn: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Netflix titles: `netflix_titles.csv`

If your filenames are slightly different, the scripts will try to auto-detect files containing `churn` or `netflix` in the name.

## How To Run

Install libraries:

```bash
pip install -r requirements.txt
```

Run both projects:

```bash
python run_all.py
```

Or run individually:

```bash
python src/churn_project.py
python src/netflix_project.py
```

## Deliverables Created

After running the scripts, these files are produced:

- `data/processed/customer_churn_cleaned.csv`
- `data/processed/netflix_titles_cleaned.csv`
- `reports/churn_insights.md`
- `reports/netflix_insights.md`
- Figures inside `reports/figures/`

## Notebook Workflow

Use these notebooks for step-wise classroom work:

- `notebooks/01_customer_churn_preparation_eda.ipynb`
- `notebooks/02_netflix_titles_eda_insights.ipynb`

The scripts and notebooks follow the same logic, so students can learn interactively in notebooks and still run the full pipeline from Python files.

