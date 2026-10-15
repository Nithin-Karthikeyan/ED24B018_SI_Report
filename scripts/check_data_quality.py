"""
Function: Validate dataset structure, detect missing values, implausible ranges,
unit inconsistencies, and unusual year-to-year changes.

Inputs:
- data/Russia_World_Bank_2010_2024.csv
- data/indicator_metadata.csv (optional, for unit checks)

Outputs:
- output/tables/data_quality.csv
- output/tables/unexpected_changes.csv
- output/reports/data_quality_report.txt
"""

# ============================================================
# check_data_quality.py
#
# CHANGES
# ============================================================
# 1. Added expected-column validation.
# 2. Added identifier-column validation.
# 3. Added year-ordering validation.
# 4. Added duplicate-year and missing-year checks.
# 5. Added non-numeric value detection for indicator columns.
# 6. Retained variable-specific impossible/range checks.
# 7. Added unit-consistency validation using indicator_metadata.csv
#    when metadata are available.
# 8. Reworked unexpected-change detection to avoid over-flagging:
#       - uses a robust MAD-based statistical screen;
#       - ALSO requires a variable-specific practical threshold;
#       - only compares genuinely consecutive years;
#       - uses percentage change for GDP per capita and CO2 per capita;
#       - uses absolute/unit changes for the remaining indicators.
# 9. Added the screening metric and practical threshold to
#    unexpected_changes.csv for transparency.
# 10. Flagged observations remain REVIEW items only. They are not
#     automatically treated as errors or removed from the dataset.
#
# NOTE:
# A change is flagged only when BOTH conditions are met:
#     1. |modified z-score| > MAD_Z_THRESHOLD
#     2. |change| exceeds the variable-specific practical threshold
#
# This prevents small but statistically unusual movements in nearly
# constant series from being incorrectly presented as major anomalies.
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "Russia_World_Bank_2010_2024.csv"
)

METADATA_FILE = (
    PROJECT_DIR
    / "data"
    / "indicator_metadata.csv"
)

OUTPUT_DIR = PROJECT_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

QUALITY_CSV = TABLE_DIR / "data_quality.csv"
UNEXPECTED_CHANGES_CSV = TABLE_DIR / "unexpected_changes.csv"
QUALITY_REPORT = REPORT_DIR / "data_quality_report.txt"


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_START_YEAR = 2010
EXPECTED_END_YEAR = 2024

IDENTIFIER_COLUMNS = {
    "Year",
    "Country",
    "Country_Code",
}

EXPECTED_INDICATOR_COLUMNS = {
    "GDP_per_capita",
    "GDP_growth",
    "Life_expectancy",
    "Basic_drinking_water",
    "Secondary_school_enrollment",
    "Electricity_access",
    "Internet_users",
    "CO2_per_capita_excl_LULUCF",
    "Forest_area",
}

EXPECTED_COLUMNS = IDENTIFIER_COLUMNS | EXPECTED_INDICATOR_COLUMNS

# Variables that logically must remain between 0 and 100.
PERCENTAGE_LIMITED_VARIABLES = {
    "Basic_drinking_water",
    "Electricity_access",
    "Internet_users",
    "Forest_area",
}

# Secondary_school_enrollment is gross enrollment, so values above
# 100% can be legitimate and are not automatically rejected.

EXPECTED_UNITS = {
    "GDP_per_capita": "constant 2015 US$ per person",
    "GDP_growth": "annual %",
    "Life_expectancy": "years",
    "Basic_drinking_water": "% of population",
    "Secondary_school_enrollment": "% gross",
    "Electricity_access": "% of population",
    "Internet_users": "% of population",
    "CO2_per_capita_excl_LULUCF": "t CO2e per capita",
    "Forest_area": "% of land area",
}

# Robust statistical threshold for annual-change screening.
MAD_Z_THRESHOLD = 3.5

# Practical thresholds prevent the MAD test from over-flagging small
# movements in nearly constant indicators.
#
# metric = "absolute": threshold is in the indicator's own change unit.
# metric = "percent":  threshold is year-to-year percentage change.
CHANGE_RULES = {
    "GDP_per_capita": {
        "metric": "percent",
        "threshold": 10.0,
        "threshold_unit": "% year-to-year",
    },
    "GDP_growth": {
        "metric": "absolute",
        "threshold": 5.0,
        "threshold_unit": "percentage points",
    },
    "Life_expectancy": {
        "metric": "absolute",
        "threshold": 1.0,
        "threshold_unit": "years",
    },
    "Basic_drinking_water": {
        "metric": "absolute",
        "threshold": 2.0,
        "threshold_unit": "percentage points",
    },
    "Secondary_school_enrollment": {
        "metric": "absolute",
        "threshold": 5.0,
        "threshold_unit": "percentage points",
    },
    "Electricity_access": {
        "metric": "absolute",
        "threshold": 5.0,
        "threshold_unit": "percentage points",
    },
    "Internet_users": {
        "metric": "absolute",
        "threshold": 10.0,
        "threshold_unit": "percentage points",
    },
    "CO2_per_capita_excl_LULUCF": {
        "metric": "percent",
        "threshold": 10.0,
        "threshold_unit": "% year-to-year",
    },
    "Forest_area": {
        "metric": "absolute",
        "threshold": 1.0,
        "threshold_unit": "percentage points",
    },
}


# ============================================================
# FUNCTIONS
# ============================================================

def load_data():
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    print("\n[LOAD] Reading dataset:")
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


# ============================================================
# DATASET STRUCTURE
# ============================================================

def check_columns(df):
    print("\n" + "-" * 70)
    print("CHECKING DATASET COLUMNS")
    print("-" * 70)

    issues = []
    actual_columns = set(df.columns)

    missing_columns = sorted(EXPECTED_COLUMNS - actual_columns)
    unexpected_columns = sorted(actual_columns - EXPECTED_COLUMNS)

    if missing_columns:
        message = f"Expected columns missing: {missing_columns}"
        issues.append(message)
        print(f"[WARNING] {message}")
    else:
        print("[OK] All expected columns are present.")

    if unexpected_columns:
        message = f"Unexpected columns present: {unexpected_columns}"
        issues.append(message)
        print(f"[WARNING] {message}")
    else:
        print("[OK] No unexpected columns detected.")

    return issues


# ============================================================
# YEAR STRUCTURE
# ============================================================

def check_years(df):
    print("\n" + "-" * 70)
    print("CHECKING YEAR STRUCTURE")
    print("-" * 70)

    issues = []

    if "Year" not in df.columns:
        return ["Year column missing from dataset."]

    numeric_years = pd.to_numeric(df["Year"], errors="coerce")

    if numeric_years.isna().any():
        count = int(numeric_years.isna().sum())
        message = f"{count} Year value(s) could not be interpreted as numeric."
        issues.append(message)
        print(f"[WARNING] {message}")

    valid_years = numeric_years.dropna().astype(int)

    expected_years = set(
        range(EXPECTED_START_YEAR, EXPECTED_END_YEAR + 1)
    )
    actual_years = set(valid_years.tolist())

    missing_years = sorted(expected_years - actual_years)
    unexpected_years = sorted(actual_years - expected_years)

    print(f"Expected years : {EXPECTED_START_YEAR}-{EXPECTED_END_YEAR}")
    print(f"Expected rows  : {len(expected_years)}")
    print(f"Actual rows    : {len(df)}")

    if missing_years:
        message = f"Dataset missing entire years: {missing_years}"
        issues.append(message)
        print(f"[WARNING] Missing years: {missing_years}")
    else:
        print("[OK] All expected years are present.")

    if unexpected_years:
        message = f"Unexpected years present: {unexpected_years}"
        issues.append(message)
        print(f"[WARNING] Unexpected years: {unexpected_years}")

    duplicate_mask = numeric_years.duplicated(keep=False) & numeric_years.notna()
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        duplicate_years = sorted(
            numeric_years[duplicate_mask]
            .astype(int)
            .unique()
            .tolist()
        )
        message = f"Duplicate year observations detected for: {duplicate_years}"
        issues.append(message)
        print(f"[WARNING] {message}")
    else:
        print("[OK] No duplicate years.")

    if valid_years.is_monotonic_increasing:
        print("[OK] Years are in chronological order.")
    else:
        message = "Rows are not sorted in chronological year order."
        issues.append(message)
        print(f"[WARNING] {message}")

    return issues


# ============================================================
# INDICATOR CHECKS
# ============================================================

def check_indicator(df, column):
    print(f"\nChecking: {column}")

    numeric = pd.to_numeric(df[column], errors="coerce")

    original_non_missing = int(df[column].notna().sum())
    numeric_non_missing = int(numeric.notna().sum())
    conversion_failures = original_non_missing - numeric_non_missing

    n = int(numeric.notna().sum())
    missing = int(numeric.isna().sum())

    if "Year" in df.columns:
        year_values = pd.to_numeric(df["Year"], errors="coerce")
        missing_years = (
            year_values[numeric.isna()]
            .dropna()
            .astype(int)
            .tolist()
        )
    else:
        missing_years = []

    minimum = numeric.min()
    maximum = numeric.max()
    issues = []

    if conversion_failures > 0:
        issues.append(
            f"{conversion_failures} value(s) could not be converted to numeric."
        )

    if column in PERCENTAGE_LIMITED_VARIABLES:
        invalid = numeric[(numeric < 0) | (numeric > 100)]
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
        invalid = numeric[(numeric <= 0) | (numeric > 120)]
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

    if column == "CO2_per_capita_excl_LULUCF":
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
        "Missing_Years": ", ".join(map(str, missing_years)),
        "Minimum": minimum,
        "Maximum": maximum,
        "Issues": " | ".join(issues) if issues else "None",
    }


# ============================================================
# UNIT CONSISTENCY
# ============================================================

def normalise_unit(unit):
    """Normalise unit text before comparison."""

    if pd.isna(unit):
        return ""

    return (
        str(unit)
        .strip()
        .lower()
        .replace("₂", "2")
        .replace("co₂", "co2")
    )


def check_units():
    print("\n" + "-" * 70)
    print("CHECKING UNIT CONSISTENCY")
    print("-" * 70)

    issues = []

    if not METADATA_FILE.exists():
        print("[SKIPPED] Metadata file not found:")
        print(f"          {METADATA_FILE}")
        print(
            "[INFO] Unit consistency cannot be verified from "
            "numeric observations alone."
        )
        return {
            "status": "SKIPPED",
            "issues": [
                "Unit consistency check skipped because "
                "indicator_metadata.csv was not available."
            ],
        }

    try:
        metadata = pd.read_csv(METADATA_FILE)
    except Exception as exc:
        message = f"Could not read metadata file: {exc}"
        print(f"[WARNING] {message}")
        return {
            "status": "WARNING",
            "issues": [message],
        }

    required_metadata_columns = {"Variable", "Unit"}
    missing = required_metadata_columns - set(metadata.columns)

    if missing:
        message = f"Metadata file missing required columns: {sorted(missing)}"
        print(f"[WARNING] {message}")
        return {
            "status": "WARNING",
            "issues": [message],
        }

    for variable, expected_unit in EXPECTED_UNITS.items():
        row = metadata[metadata["Variable"] == variable]

        if row.empty:
            message = f"{variable}: metadata entry not found."
            issues.append(message)
            print(f"[WARNING] {message}")
            continue

        actual_unit = row.iloc[0]["Unit"]
        expected_normalised = normalise_unit(expected_unit)
        actual_normalised = normalise_unit(actual_unit)

        if expected_normalised != actual_normalised:
            message = (
                f"{variable}: expected unit '{expected_unit}', "
                f"found '{actual_unit}'."
            )
            issues.append(message)
            print(f"[WARNING] {message}")
        else:
            print(f"[OK] {variable}: {actual_unit}")

    if issues:
        return {
            "status": "WARNING",
            "issues": issues,
        }

    print("[OK] All available indicator units match expected units.")
    return {
        "status": "PASS",
        "issues": [],
    }


# ============================================================
# UNEXPECTED YEAR-TO-YEAR CHANGES
# ============================================================

def _build_change_metric(series_df, metric):
    """
    Build the change metric used for anomaly screening.

    absolute -> current value - previous value
    percent  -> percentage change relative to previous value
    """

    series_df["Change"] = (
        series_df["Value"]
        - series_df["Previous_Value"]
    )

    series_df["Percent_Change"] = np.where(
        series_df["Previous_Value"].abs() > 0,
        (
            series_df["Change"]
            / series_df["Previous_Value"].abs()
        ) * 100.0,
        np.nan,
    )

    if metric == "percent":
        return series_df["Percent_Change"]

    return series_df["Change"]


def detect_unexpected_changes(df, indicator_columns):
    print("\n" + "-" * 70)
    print("CHECKING UNEXPECTED YEAR-TO-YEAR CHANGES")
    print("-" * 70)
    print(
        "[INFO] A change is flagged only if it is BOTH statistically "
        "unusual and practically large."
    )

    results = []

    if "Year" not in df.columns:
        print("[SKIPPED] Year column unavailable.")
        return pd.DataFrame()

    working_df = df.copy()
    working_df["Year"] = pd.to_numeric(
        working_df["Year"],
        errors="coerce",
    )
    working_df = (
        working_df
        .dropna(subset=["Year"])
        .sort_values("Year")
    )

    for column in indicator_columns:
        rule = CHANGE_RULES.get(
            column,
            {
                "metric": "absolute",
                "threshold": 0.0,
                "threshold_unit": "indicator units",
            },
        )

        metric = rule["metric"]
        practical_threshold = float(rule["threshold"])
        threshold_unit = rule["threshold_unit"]

        numeric = pd.to_numeric(
            working_df[column],
            errors="coerce",
        )

        series_df = pd.DataFrame({
            "Year": working_df["Year"],
            "Value": numeric,
        })

        series_df = (
            series_df
            .dropna()
            .sort_values("Year")
        )

        if len(series_df) < 4:
            print(f"[SKIPPED] {column}: too few observations.")
            continue

        series_df["Previous_Year"] = series_df["Year"].shift(1)
        series_df["Previous_Value"] = series_df["Value"].shift(1)
        series_df["Year_Gap"] = (
            series_df["Year"]
            - series_df["Previous_Year"]
        )

        # Only compare true year-to-year movements. If an intermediate
        # observation is missing, do not treat a multi-year movement as
        # a one-year jump.
        series_df = series_df[
            series_df["Year_Gap"] == 1
        ].copy()

        if series_df.empty:
            print(f"[SKIPPED] {column}: no consecutive-year pairs available.")
            continue

        screening_value = _build_change_metric(
            series_df,
            metric,
        )
        series_df["Screening_Value"] = screening_value

        valid_screening = series_df["Screening_Value"].dropna()

        if valid_screening.empty:
            print(f"[SKIPPED] {column}: no valid change metric available.")
            continue

        median_change = valid_screening.median()
        mad = (
            valid_screening
            .sub(median_change)
            .abs()
            .median()
        )

        series_df["Modified_Z_Score"] = np.nan

        if np.isclose(mad, 0.0):
            # If the normal change pattern is essentially constant,
            # treat deviations from that pattern as statistically unusual.
            statistically_unusual = (
                series_df["Screening_Value"]
                .sub(median_change)
                .abs()
                > 1e-12
            )
        else:
            modified_z = (
                0.6745
                * (
                    series_df["Screening_Value"]
                    - median_change
                )
                / mad
            )
            series_df["Modified_Z_Score"] = modified_z
            statistically_unusual = (
                modified_z.abs() > MAD_Z_THRESHOLD
            )

        practically_large = (
            series_df["Screening_Value"].abs()
            > practical_threshold
        )

        flags = (
            statistically_unusual
            & practically_large
        )

        flagged = series_df[flags].copy()

        metric_name = (
            "percentage change"
            if metric == "percent"
            else "absolute change"
        )

        if flagged.empty:
            print(
                f"[OK] {column}: no changes met both screening criteria "
                f"({metric_name}, practical threshold = "
                f"{practical_threshold:g} {threshold_unit})."
            )
            continue

        print(
            f"[REVIEW] {column}: {len(flagged)} change(s) flagged "
            f"({metric_name}, threshold = "
            f"{practical_threshold:g} {threshold_unit})."
        )

        for _, row in flagged.iterrows():
            previous_year = int(row["Previous_Year"])
            current_year = int(row["Year"])

            print(
                "    "
                f"{previous_year} -> {current_year}: "
                f"{row['Previous_Value']} -> {row['Value']} "
                f"(raw change = {row['Change']}, "
                f"screening value = {row['Screening_Value']})"
            )

            results.append({
                "Variable": column,
                "Previous_Year": previous_year,
                "Year": current_year,
                "Previous_Value": row["Previous_Value"],
                "Value": row["Value"],
                "Change": row["Change"],
                "Percent_Change": row["Percent_Change"],
                "Screening_Metric": metric_name,
                "Screening_Value": row["Screening_Value"],
                "Practical_Threshold": practical_threshold,
                "Threshold_Unit": threshold_unit,
                "Modified_Z_Score": row["Modified_Z_Score"],
                "Status": "REVIEW",
            })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    create_output_directories()
    df = load_data()

    overall_issues = []

    column_issues = check_columns(df)
    overall_issues.extend(column_issues)

    year_issues = check_years(df)
    overall_issues.extend(year_issues)

    indicators = [
        column
        for column in get_indicator_columns(df)
        if column in EXPECTED_INDICATOR_COLUMNS
    ]

    print("\n" + "=" * 70)
    print("CHECKING INDICATORS")
    print("=" * 70)

    quality_results = []

    for column in indicators:
        result = check_indicator(df, column)
        quality_results.append(result)

    quality_df = pd.DataFrame(quality_results)

    unit_result = check_units()

    unexpected_changes_df = detect_unexpected_changes(
        df,
        indicators,
    )

    # --------------------------------------------------------
    # Save quality summary CSV
    # --------------------------------------------------------

    try:
        quality_df.to_csv(
            QUALITY_CSV,
            index=False,
        )
        print("\n[OK] Quality table saved:")
        print(f"     {QUALITY_CSV}")
    except OSError as exc:
        print(f"[ERROR] Could not save CSV: {exc}")

    # --------------------------------------------------------
    # Save unusual changes CSV
    # --------------------------------------------------------

    unexpected_change_columns = [
        "Variable",
        "Previous_Year",
        "Year",
        "Previous_Value",
        "Value",
        "Change",
        "Percent_Change",
        "Screening_Metric",
        "Screening_Value",
        "Practical_Threshold",
        "Threshold_Unit",
        "Modified_Z_Score",
        "Status",
    ]

    try:
        if unexpected_changes_df.empty:
            pd.DataFrame(
                columns=unexpected_change_columns
            ).to_csv(
                UNEXPECTED_CHANGES_CSV,
                index=False,
            )
        else:
            unexpected_changes_df.to_csv(
                UNEXPECTED_CHANGES_CSV,
                index=False,
            )

        print("[OK] Unexpected-change table saved:")
        print(f"     {UNEXPECTED_CHANGES_CSV}")
    except OSError as exc:
        print(
            "[ERROR] Could not save unexpected-change CSV: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    try:
        with QUALITY_REPORT.open(
            "w",
            encoding="utf-8",
        ) as report:

            report.write(
                "RUSSIA WORLD BANK DATA QUALITY REPORT\n"
            )
            report.write("=" * 60 + "\n\n")

            report.write(f"Dataset: {DATA_FILE.name}\n")
            report.write(
                f"Period: {EXPECTED_START_YEAR}-{EXPECTED_END_YEAR}\n"
            )
            report.write(f"Rows: {len(df)}\n")
            report.write(f"Columns: {len(df.columns)}\n\n")

            report.write("DATASET STRUCTURE ISSUES\n")
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
            report.write("\n")

            report.write("\nUNIT CONSISTENCY CHECK\n")
            report.write("-" * 60 + "\n")
            report.write(f"Status: {unit_result['status']}\n")

            if unit_result["issues"]:
                for issue in unit_result["issues"]:
                    report.write(f"- {issue}\n")
            else:
                report.write(
                    "All units match expected metadata.\n"
                )

            report.write("\nUNEXPECTED YEAR-TO-YEAR CHANGES\n")
            report.write("-" * 60 + "\n")
            report.write(
                "Only consecutive-year pairs are examined. A change is "
                "flagged only when it is both statistically unusual under "
                "the MAD-based screen and larger than the variable-specific "
                "practical threshold. Flagged values are retained and are "
                "review items, not automatic data errors.\n\n"
            )

            if unexpected_changes_df.empty:
                report.write(
                    "No annual changes met both unexpected-change "
                    "screening criteria.\n"
                )
            else:
                report.write(
                    unexpected_changes_df.to_string(index=False)
                )
                report.write("\n")

        print("[OK] Text report saved:")
        print(f"     {QUALITY_REPORT}")

    except OSError as exc:
        print(f"[ERROR] Could not save report: {exc}")

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
