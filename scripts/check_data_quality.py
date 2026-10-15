from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "Russia_World_Bank_2010_2024.csv"

OUTPUT_DIR = PROJECT_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

QUALITY_CSV = TABLE_DIR / "data_quality.csv"
QUALITY_REPORT = REPORT_DIR / "data_quality_report.txt"


# ============================================================
# SETTINGS
# ============================================================

IDENTIFIER_COLUMNS = {
    "Year",
    "Country",
    "Country_Code",
}

EXPECTED_START_YEAR = 2010
EXPECTED_END_YEAR = 2024

# Variables that logically must remain between 0 and 100
PERCENTAGE_LIMITED_VARIABLES = {
    "Basic_drinking_water",
    "Electricity_access",
    "Internet_users",
    "Forest_area",
}

# NOTE:
# Secondary_school_enrollment is gross enrollment.
# Gross enrollment can legitimately exceed 100%, so we DO NOT
# apply a maximum of 100 to it.


# ============================================================
# FUNCTIONS
# ============================================================

def load_data():
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    print(f"\n[LOAD] Reading dataset:")
    print(f"       {DATA_FILE}")

    if not DATA_FILE.exists():
        print("[FATAL] Dataset does not exist.")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_FILE)

    except pd.errors.EmptyDataError:
        print("[FATAL] CSV file is empty.")
        sys.exit(1)

    except pd.errors.ParserError as exc:
        print(f"[FATAL] Could not parse CSV: {exc}")
        sys.exit(1)

    except OSError as exc:
        print(f"[FATAL] Could not open file: {exc}")
        sys.exit(1)

    print(f"[OK] Loaded {len(df)} rows and {len(df.columns)} columns.")

    return df


def create_output_directories():
    print("\n[SETUP] Creating output directories...")

    try:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        print(f"[FATAL] Could not create output directories: {exc}")
        sys.exit(1)

    print(f"[OK] Tables  -> {TABLE_DIR}")
    print(f"[OK] Reports -> {REPORT_DIR}")


def get_indicator_columns(df):
    return [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
    ]


def check_years(df):
    print("\n" + "-" * 70)
    print("CHECKING YEAR STRUCTURE")
    print("-" * 70)

    issues = []

    if "Year" not in df.columns:
        return ["Year column missing from dataset."]

    expected_years = set(
        range(EXPECTED_START_YEAR, EXPECTED_END_YEAR + 1)
    )

    actual_years = set(
        pd.to_numeric(df["Year"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    missing_years = sorted(expected_years - actual_years)
    unexpected_years = sorted(actual_years - expected_years)

    print(f"Expected years : {EXPECTED_START_YEAR}-{EXPECTED_END_YEAR}")
    print(f"Expected rows  : {len(expected_years)}")
    print(f"Actual rows    : {len(df)}")

    if missing_years:
        issues.append(
            f"Dataset missing entire years: {missing_years}"
        )
        print(f"[WARNING] Missing years: {missing_years}")
    else:
        print("[OK] All expected years are present.")

    if unexpected_years:
        issues.append(
            f"Unexpected years present: {unexpected_years}"
        )
        print(f"[WARNING] Unexpected years: {unexpected_years}")

    duplicates = df["Year"].duplicated().sum()

    if duplicates > 0:
        issues.append(
            f"{duplicates} duplicate year observations detected."
        )
        print(f"[WARNING] Duplicate years: {duplicates}")
    else:
        print("[OK] No duplicate years.")

    return issues


def check_indicator(df, column):
    print(f"\nChecking: {column}")

    numeric = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    original_non_missing = df[column].notna().sum()
    numeric_non_missing = numeric.notna().sum()

    conversion_failures = (
        original_non_missing - numeric_non_missing
    )

    n = int(numeric.notna().sum())
    missing = int(numeric.isna().sum())

    missing_years = (
        df.loc[numeric.isna(), "Year"]
        .astype(int)
        .tolist()
    )

    minimum = numeric.min()
    maximum = numeric.max()

    issues = []

    if conversion_failures > 0:
        issues.append(
            f"{conversion_failures} value(s) could not be converted "
            "to numeric."
        )

    if column in PERCENTAGE_LIMITED_VARIABLES:

        invalid = numeric[
            (numeric < 0) | (numeric > 100)
        ]

        if not invalid.empty:
            issues.append(
                f"{len(invalid)} observation(s) outside 0-100%."
            )

    if column == "Secondary_school_enrollment":

        negative_values = numeric[numeric < 0]

        if not negative_values.empty:
            issues.append(
                f"{len(negative_values)} negative enrollment value(s)."
            )

    if column == "Life_expectancy":

        invalid = numeric[
            (numeric <= 0) | (numeric > 120)
        ]

        if not invalid.empty:
            issues.append(
                f"{len(invalid)} implausible life expectancy value(s)."
            )

    if column == "GDP_per_capita":

        negative_values = numeric[numeric < 0]

        if not negative_values.empty:
            issues.append(
                f"{len(negative_values)} negative GDP per capita value(s)."
            )

    if "CO2" in column:

        negative_values = numeric[numeric < 0]

        if not negative_values.empty:
            issues.append(
                f"{len(negative_values)} negative CO2 value(s)."
            )

    print(f"    N              : {n}")
    print(f"    Missing        : {missing}")
    print(f"    Missing years  : {missing_years}")
    print(f"    Minimum        : {minimum}")
    print(f"    Maximum        : {maximum}")

    if issues:
        for issue in issues:
            print(f"    [WARNING] {issue}")
    else:
        print("    [OK] No obvious quality problems.")

    return {
        "Variable": column,
        "Potential_N": len(df),
        "Non_Missing_N": n,
        "Missing_N": missing,
        "Missing_Years": ", ".join(
            map(str, missing_years)
        ),
        "Minimum": minimum,
        "Maximum": maximum,
        "Issues": " | ".join(issues) if issues else "None",
    }


def main():
    create_output_directories()

    df = load_data()

    overall_issues = check_years(df)

    indicators = get_indicator_columns(df)

    print("\n" + "=" * 70)
    print("CHECKING INDICATORS")
    print("=" * 70)

    quality_results = []

    for column in indicators:
        result = check_indicator(df, column)
        quality_results.append(result)

    quality_df = pd.DataFrame(quality_results)

    try:
        quality_df.to_csv(
            QUALITY_CSV,
            index=False,
        )

        print(f"\n[OK] Quality table saved:")
        print(f"     {QUALITY_CSV}")

    except OSError as exc:
        print(f"[ERROR] Could not save CSV: {exc}")

    try:
        with QUALITY_REPORT.open(
            "w",
            encoding="utf-8",
        ) as report:

            report.write(
                "RUSSIA WORLD BANK DATA QUALITY REPORT\n"
            )
            report.write("=" * 60 + "\n\n")

            report.write(
                f"Dataset: {DATA_FILE.name}\n"
            )
            report.write(
                f"Period: {EXPECTED_START_YEAR}-"
                f"{EXPECTED_END_YEAR}\n"
            )
            report.write(
                f"Rows: {len(df)}\n\n"
            )

            report.write("OVERALL ISSUES\n")
            report.write("-" * 60 + "\n")

            if overall_issues:
                for issue in overall_issues:
                    report.write(f"- {issue}\n")
            else:
                report.write("None detected.\n")

            report.write("\nINDICATOR SUMMARY\n")
            report.write("-" * 60 + "\n")

            report.write(
                quality_df.to_string(index=False)
            )

        print(f"[OK] Text report saved:")
        print(f"     {QUALITY_REPORT}")

    except OSError as exc:
        print(f"[ERROR] Could not save report: {exc}")

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()