from pathlib import Path


def find_csv(raw_dir: Path, preferred_name: str, keyword: str) -> Path:
    preferred = raw_dir / preferred_name
    if preferred.exists():
        return preferred

    matches = sorted(raw_dir.glob(f"*{keyword}*.csv"))
    if matches:
        return matches[0]

    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        f"Could not find {preferred_name}. Place it in {raw_dir} or use a CSV filename containing '{keyword}'."
    )


def ensure_output_dirs(project_root: Path) -> tuple[Path, Path]:
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "reports" / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, figures_dir

