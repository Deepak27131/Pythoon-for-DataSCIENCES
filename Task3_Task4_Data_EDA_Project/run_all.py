from pathlib import Path

from src.churn_project import run_churn_project
from src.netflix_project import run_netflix_project


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print("Running Task 3: Customer Churn Data Preparation and EDA")
    run_churn_project(project_root)
    print("\nRunning Task 4: Netflix Titles EDA and Insights")
    run_netflix_project(project_root)
    print("\nDone. Check data/processed and reports/figures for outputs.")


if __name__ == "__main__":
    main()

