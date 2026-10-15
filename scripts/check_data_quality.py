"""
Function: Validate dataset structure, detect missing values, implausible ranges, and unusual year-to-year changes.
Inputs: data/Russia_World_Bank_2010_2024.csv (optional: data/indicator_metadata.csv for unit checks).
Outputs:
- output/tables/data_quality.csv: Per-indicator quality metrics (N, missing, min, max, issues).
- output/tables/unexpected_changes.csv: Flagged observations with unusual year-to-year shifts (MAD-based).
- output/reports/data_quality_report.txt: Human-readable summary of all checks.
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
# 8. Added robust year-to-year unexpected-change detection using
#    first differences and Median Absolute Deviation (MAD).
# 9. Added unexpected_changes.csv output for observations that
#    require manual investigation.
# 10. Added clearer PASS / WARNING / SKIPPED information to the
#     generated data-quality report.
#
# NOTE:
# Unexpected changes are only FLAGGED for investigation.
# They are not automatically treated as incorrect observations.
#
# Unit validation requires metadata produced during extraction.
# If data/indicator_metadata.csv does not exist, the unit check
# is reported as skipped rather than falsely marked as passed.
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

# Optional metadata file.
# extract_data.py can later be modified to generate this.
METADATA_FILE = (
    PROJECT_DIR
    / "data"
    / "indicator_metadata.csv"
)

OUTPUT_DIR = PROJECT_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

QUALITY_CSV = TABLE_DIR / "data_quality.csv"
UNEXPECTED_CHANGES_CSV = (
    TABLE_DIR
    / "unexpected_changes.csv"
)

QUALITY_REPORT = (
    REPORT_DIR
    / "data_quality_report.txt"
)


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


EXPECTED_COLUMNS = (
    IDENTIFIER_COLUMNS
    | EXPECTED_INDICATOR_COLUMNS
)


# Variables that logically must remain between 0 and 100.
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


# Expected World Bank units.
#
# These are checked only when metadata are available.
EXPECTED_UNITS = {
    "GDP_per_capita":
        "constant 2015 US$ per person",

    "GDP_growth":
        "annual %",

    "Life_expectancy":
        "years",

    "Basic_drinking_water":
        "% of population",

    "Secondary_school_enrollment":
        "% gross",

    "Electricity_access":
        "% of population",

    "Internet_users":
        "% of population",

    "CO2_per_capita_excl_LULUCF":
        "t CO2e per capita",

    "Forest_area":
        "% of land area",
}


# Threshold used for robust unusual-change detection.
#
# A modified z-score above 3.5 is commonly used as an
# outlier-screening rule.
MAD_Z_THRESHOLD = 3.5


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
        print(
            f"[FATAL] Could not parse CSV: {exc}"
        )
        sys.exit(1)

    except OSError as exc:
        print(
            f"[FATAL] Could not open file: {exc}"
        )
        sys.exit(1)

    print(
        f"[OK] Loaded {len(df)} rows and "
        f"{len(df.columns)} columns."
    )

    return df


def create_output_directories():
    print(
        "\n[SETUP] Creating output directories..."
    )

    try:
        TABLE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as exc:
        print(
            "[FATAL] Could not create output "
            f"directories: {exc}"
        )
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

    missing_columns = sorted(
        EXPECTED_COLUMNS - actual_columns
    )

    unexpected_columns = sorted(
        actual_columns - EXPECTED_COLUMNS
    )

    if missing_columns:
        message = (
            "Expected columns missing: "
            f"{missing_columns}"
        )

        issues.append(message)

        print(
            f"[WARNING] {message}"
        )

    else:
        print(
            "[OK] All expected columns are present."
        )

    if unexpected_columns:
        message = (
            "Unexpected columns present: "
            f"{unexpected_columns}"
        )

        issues.append(message)

        print(
            f"[WARNING] {message}"
        )

    else:
        print(
            "[OK] No unexpected columns detected."
        )

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
        return [
            "Year column missing from dataset."
        ]

    numeric_years = pd.to_numeric(
        df["Year"],
        errors="coerce",
    )

    if numeric_years.isna().any():
        count = int(
            numeric_years.isna().sum()
        )

        message = (
            f"{count} Year value(s) could not "
            "be interpreted as numeric."
        )

        issues.append(message)

        print(
            f"[WARNING] {message}"
        )

    valid_years = (
        numeric_years
        .dropna()
        .astype(int)
    )

    expected_years = set(
        range(
            EXPECTED_START_YEAR,
            EXPECTED_END_YEAR + 1,
        )
    )

    actual_years = set(
        valid_years.tolist()
    )

    missing_years = sorted(
        expected_years - actual_years
    )

    unexpected_years = sorted(
        actual_years - expected_years
    )

    print(
        f"Expected years : "
        f"{EXPECTED_START_YEAR}-"
        f"{EXPECTED_END_YEAR}"
    )

    print(
        f"Expected rows  : "
        f"{len(expected_years)}"
    )

    print(
        f"Actual rows    : "
        f"{len(df)}"
    )

    if missing_years:

        message = (
            "Dataset missing entire years: "
            f"{missing_years}"
        )

        issues.append(message)

        print(
            f"[WARNING] Missing years: "
            f"{missing_years}"
        )

    else:
        print(
            "[OK] All expected years are present."
        )

    if unexpected_years:

        message = (
            "Unexpected years present: "
            f"{unexpected_years}"
        )

        issues.append(message)

        print(
            f"[WARNING] Unexpected years: "
            f"{unexpected_years}"
        )

    duplicate_mask = (
        numeric_years
        .duplicated(
            keep=False
        )
        & numeric_years.notna()
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        duplicate_years = sorted(
            numeric_years[
                duplicate_mask
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        message = (
            "Duplicate year observations detected "
            f"for: {duplicate_years}"
        )

        issues.append(message)

        print(
            f"[WARNING] {message}"
        )

    else:
        print(
            "[OK] No duplicate years."
        )

    # Check chronological ordering.
    if valid_years.is_monotonic_increasing:
        print(
            "[OK] Years are in chronological order."
        )

    else:

        message = (
            "Rows are not sorted in chronological "
            "year order."
        )

        issues.append(message)

        print(
            f"[WARNING] {message}"
        )

    return issues


# ============================================================
# INDICATOR CHECKS
# ============================================================

def check_indicator(df, column):
    print(
        f"\nChecking: {column}"
    )

    numeric = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    original_non_missing = int(
        df[column]
        .notna()
        .sum()
    )

    numeric_non_missing = int(
        numeric
        .notna()
        .sum()
    )

    conversion_failures = (
        original_non_missing
        - numeric_non_missing
    )

    n = int(
        numeric.notna().sum()
    )

    missing = int(
        numeric.isna().sum()
    )

    if "Year" in df.columns:

        year_values = pd.to_numeric(
            df["Year"],
            errors="coerce",
        )

        missing_years = (
            year_values[
                numeric.isna()
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

    else:
        missing_years = []

    minimum = numeric.min()
    maximum = numeric.max()

    issues = []

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    if conversion_failures > 0:

        issues.append(
            f"{conversion_failures} value(s) "
            "could not be converted to numeric."
        )

    # --------------------------------------------------------
    # Percentage-limited indicators
    # --------------------------------------------------------

    if column in PERCENTAGE_LIMITED_VARIABLES:

        invalid = numeric[
            (numeric < 0)
            | (numeric > 100)
        ]

        if not invalid.empty:

            issues.append(
                f"{len(invalid)} observation(s) "
                "outside 0-100%."
            )

    # --------------------------------------------------------
    # Gross secondary enrollment
    # --------------------------------------------------------

    if column == "Secondary_school_enrollment":

        negative_values = numeric[
            numeric < 0
        ]

        if not negative_values.empty:

            issues.append(
                f"{len(negative_values)} negative "
                "enrollment value(s)."
            )

    # --------------------------------------------------------
    # Life expectancy
    # --------------------------------------------------------

    if column == "Life_expectancy":

        invalid = numeric[
            (numeric <= 0)
            | (numeric > 120)
        ]

        if not invalid.empty:

            issues.append(
                f"{len(invalid)} implausible life "
                "expectancy value(s)."
            )

    # --------------------------------------------------------
    # GDP per capita
    # --------------------------------------------------------

    if column == "GDP_per_capita":

        negative_values = numeric[
            numeric < 0
        ]

        if not negative_values.empty:

            issues.append(
                f"{len(negative_values)} negative "
                "GDP per capita value(s)."
            )

    # --------------------------------------------------------
    # CO2 excluding LULUCF per capita
    # --------------------------------------------------------

    if (
        column
        == "CO2_per_capita_excl_LULUCF"
    ):

        negative_values = numeric[
            numeric < 0
        ]

        if not negative_values.empty:

            issues.append(
                f"{len(negative_values)} negative "
                "CO2 value(s)."
            )

    print(
        f"    N              : {n}"
    )

    print(
        f"    Missing        : {missing}"
    )

    print(
        f"    Missing years  : "
        f"{missing_years}"
    )

    print(
        f"    Minimum        : "
        f"{minimum}"
    )

    print(
        f"    Maximum        : "
        f"{maximum}"
    )

    if issues:

        for issue in issues:
            print(
                f"    [WARNING] {issue}"
            )

    else:
        print(
            "    [OK] No obvious "
            "quality problems."
        )

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
        "Issues":
            " | ".join(issues)
            if issues
            else "None",
    }


# ============================================================
# UNIT CONSISTENCY
# ============================================================

def normalise_unit(unit):
    """
    Normalise unit text before comparison so minor
    formatting/capitalisation differences do not cause
    false mismatches.
    """

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

        print(
            "[SKIPPED] Metadata file not found:"
        )

        print(
            f"          {METADATA_FILE}"
        )

        print(
            "[INFO] Unit consistency cannot be "
            "verified from numeric observations alone."
        )

        return {
            "status": "SKIPPED",
            "issues": [
                "Unit consistency check skipped because "
                "indicator_metadata.csv was not available."
            ],
        }

    try:
        metadata = pd.read_csv(
            METADATA_FILE
        )

    except Exception as exc:

        message = (
            "Could not read metadata file: "
            f"{exc}"
        )

        print(
            f"[WARNING] {message}"
        )

        return {
            "status": "WARNING",
            "issues": [message],
        }

    required_metadata_columns = {
        "Variable",
        "Unit",
    }

    missing = (
        required_metadata_columns
        - set(metadata.columns)
    )

    if missing:

        message = (
            "Metadata file missing required "
            f"columns: {sorted(missing)}"
        )

        print(
            f"[WARNING] {message}"
        )

        return {
            "status": "WARNING",
            "issues": [message],
        }

    for variable, expected_unit \
            in EXPECTED_UNITS.items():

        row = metadata[
            metadata["Variable"]
            == variable
        ]

        if row.empty:

            message = (
                f"{variable}: metadata entry "
                "not found."
            )

            issues.append(message)

            print(
                f"[WARNING] {message}"
            )

            continue

        actual_unit = (
            row.iloc[0]["Unit"]
        )

        expected_normalised = (
            normalise_unit(
                expected_unit
            )
        )

        actual_normalised = (
            normalise_unit(
                actual_unit
            )
        )

        if (
            expected_normalised
            != actual_normalised
        ):

            message = (
                f"{variable}: expected unit "
                f"'{expected_unit}', "
                f"found '{actual_unit}'."
            )

            issues.append(message)

            print(
                f"[WARNING] {message}"
            )

        else:

            print(
                f"[OK] {variable}: "
                f"{actual_unit}"
            )

    if issues:

        return {
            "status": "WARNING",
            "issues": issues,
        }

    print(
        "[OK] All available indicator units "
        "match expected units."
    )

    return {
        "status": "PASS",
        "issues": [],
    }


# ============================================================
# UNEXPECTED YEAR-TO-YEAR CHANGES
# ============================================================

def detect_unexpected_changes(
    df,
    indicator_columns,
):
    print("\n" + "-" * 70)
    print("CHECKING UNEXPECTED YEAR-TO-YEAR CHANGES")
    print("-" * 70)

    results = []

    if "Year" not in df.columns:

        print(
            "[SKIPPED] Year column unavailable."
        )

        return pd.DataFrame()

    working_df = df.copy()

    working_df["Year"] = pd.to_numeric(
        working_df["Year"],
        errors="coerce",
    )

    working_df = (
        working_df
        .dropna(
            subset=["Year"]
        )
        .sort_values("Year")
    )

    for column in indicator_columns:

        numeric = pd.to_numeric(
            working_df[column],
            errors="coerce",
        )

        series_df = pd.DataFrame({
            "Year":
                working_df["Year"],
            "Value":
                numeric,
        })

        series_df = (
            series_df
            .dropna()
            .sort_values("Year")
        )

        # Need enough observations to identify
        # unusual differences sensibly.
        if len(series_df) < 4:

            print(
                f"[SKIPPED] {column}: "
                "too few observations."
            )

            continue

        series_df["Previous_Year"] = (
            series_df["Year"].shift(1)
        )

        series_df["Previous_Value"] = (
            series_df["Value"].shift(1)
        )

        series_df["Change"] = (
            series_df["Value"]
            - series_df["Previous_Value"]
        )

        changes = (
            series_df["Change"]
            .dropna()
        )

        if changes.empty:
            continue

        median_change = (
            changes.median()
        )

        absolute_deviation = (
            changes
            - median_change
        ).abs()

        mad = (
            absolute_deviation
            .median()
        )

        # If MAD is zero, most annual changes are
        # identical. Flag any changes that differ from
        # that repeated pattern.
        if mad == 0:

            flags = (
                series_df["Change"]
                .notna()
                & (
                    series_df["Change"]
                    != median_change
                )
            )

            series_df[
                "Modified_Z_Score"
            ] = np.nan

        else:

            modified_z = (
                0.6745
                * (
                    series_df["Change"]
                    - median_change
                )
                / mad
            )

            series_df[
                "Modified_Z_Score"
            ] = modified_z

            flags = (
                modified_z.abs()
                > MAD_Z_THRESHOLD
            )

        flagged = series_df[
            flags
        ].copy()

        if flagged.empty:

            print(
                f"[OK] {column}: "
                "no unusually large annual "
                "changes detected."
            )

            continue

        print(
            f"[REVIEW] {column}: "
            f"{len(flagged)} change(s) flagged."
        )

        for _, row in flagged.iterrows():

            previous_year = int(
                row["Previous_Year"]
            )

            current_year = int(
                row["Year"]
            )

            print(
                "    "
                f"{previous_year} -> "
                f"{current_year}: "
                f"{row['Previous_Value']} -> "
                f"{row['Value']} "
                f"(change = {row['Change']})"
            )

            results.append({
                "Variable":
                    column,

                "Previous_Year":
                    previous_year,

                "Year":
                    current_year,

                "Previous_Value":
                    row[
                        "Previous_Value"
                    ],

                "Value":
                    row[
                        "Value"
                    ],

                "Change":
                    row[
                        "Change"
                    ],

                "Modified_Z_Score":
                    row[
                        "Modified_Z_Score"
                    ],

                "Status":
                    "REVIEW",
            })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    create_output_directories()

    df = load_data()

    overall_issues = []

    # --------------------------------------------------------
    # Dataset structure
    # --------------------------------------------------------

    column_issues = check_columns(
        df
    )

    overall_issues.extend(
        column_issues
    )

    # --------------------------------------------------------
    # Year structure
    # --------------------------------------------------------

    year_issues = check_years(
        df
    )

    overall_issues.extend(
        year_issues
    )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    indicators = [
        column
        for column in get_indicator_columns(
            df
        )
        if column in EXPECTED_INDICATOR_COLUMNS
    ]

    print("\n" + "=" * 70)
    print("CHECKING INDICATORS")
    print("=" * 70)

    quality_results = []

    for column in indicators:

        result = check_indicator(
            df,
            column,
        )

        quality_results.append(
            result
        )

    quality_df = pd.DataFrame(
        quality_results
    )

    # --------------------------------------------------------
    # Units
    # --------------------------------------------------------

    unit_result = check_units()

    # --------------------------------------------------------
    # Unexpected changes
    # --------------------------------------------------------

    unexpected_changes_df = (
        detect_unexpected_changes(
            df,
            indicators,
        )
    )

    # --------------------------------------------------------
    # Save quality summary CSV
    # --------------------------------------------------------

    try:

        quality_df.to_csv(
            QUALITY_CSV,
            index=False,
        )

        print(
            "\n[OK] Quality table saved:"
        )

        print(
            f"     {QUALITY_CSV}"
        )

    except OSError as exc:

        print(
            f"[ERROR] Could not save CSV: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # Save unusual changes CSV
    # --------------------------------------------------------

    try:

        if unexpected_changes_df.empty:

            empty_columns = [
                "Variable",
                "Previous_Year",
                "Year",
                "Previous_Value",
                "Value",
                "Change",
                "Modified_Z_Score",
                "Status",
            ]

            pd.DataFrame(
                columns=empty_columns
            ).to_csv(
                UNEXPECTED_CHANGES_CSV,
                index=False,
            )

        else:

            unexpected_changes_df.to_csv(
                UNEXPECTED_CHANGES_CSV,
                index=False,
            )

        print(
            "[OK] Unexpected-change table saved:"
        )

        print(
            f"     {UNEXPECTED_CHANGES_CSV}"
        )

    except OSError as exc:

        print(
            "[ERROR] Could not save "
            "unexpected-change CSV: "
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
                "RUSSIA WORLD BANK "
                "DATA QUALITY REPORT\n"
            )

            report.write(
                "=" * 60 + "\n\n"
            )

            report.write(
                f"Dataset: "
                f"{DATA_FILE.name}\n"
            )

            report.write(
                f"Period: "
                f"{EXPECTED_START_YEAR}-"
                f"{EXPECTED_END_YEAR}\n"
            )

            report.write(
                f"Rows: {len(df)}\n"
            )

            report.write(
                f"Columns: "
                f"{len(df.columns)}\n\n"
            )

            # --------------------------------------------
            # Overall issues
            # --------------------------------------------

            report.write(
                "DATASET STRUCTURE ISSUES\n"
            )

            report.write(
                "-" * 60 + "\n"
            )

            if overall_issues:

                for issue in overall_issues:

                    report.write(
                        f"- {issue}\n"
                    )

            else:

                report.write(
                    "None detected.\n"
                )

            # --------------------------------------------
            # Indicator summary
            # --------------------------------------------

            report.write(
                "\nINDICATOR SUMMARY\n"
            )

            report.write(
                "-" * 60 + "\n"
            )

            report.write(
                quality_df.to_string(
                    index=False
                )
            )

            report.write(
                "\n"
            )

            # --------------------------------------------
            # Unit check
            # --------------------------------------------

            report.write(
                "\nUNIT CONSISTENCY CHECK\n"
            )

            report.write(
                "-" * 60 + "\n"
            )

            report.write(
                f"Status: "
                f"{unit_result['status']}\n"
            )

            if unit_result["issues"]:

                for issue in (
                    unit_result["issues"]
                ):

                    report.write(
                        f"- {issue}\n"
                    )

            else:

                report.write(
                    "All units match expected "
                    "metadata.\n"
                )

            # --------------------------------------------
            # Unexpected changes
            # --------------------------------------------

            report.write(
                "\nUNEXPECTED YEAR-TO-YEAR "
                "CHANGES\n"
            )

            report.write(
                "-" * 60 + "\n"
            )

            report.write(
                "These observations are flagged "
                "for review only and are not "
                "automatically treated as "
                "incorrect.\n\n"
            )

            if unexpected_changes_df.empty:

                report.write(
                    "No unexpected annual changes "
                    "were flagged by the MAD-based "
                    "screening rule.\n"
                )

            else:

                report.write(
                    unexpected_changes_df
                    .to_string(
                        index=False
                    )
                )

                report.write(
                    "\n"
                )

        print(
            "[OK] Text report saved:"
        )

        print(
            f"     {QUALITY_REPORT}"
        )

    except OSError as exc:

        print(
            f"[ERROR] Could not save report: "
            f"{exc}"
        )

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()