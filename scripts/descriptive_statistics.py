"""
Function: Compute descriptive statistics for all indicators (mean, median, variance, std dev, IQR, CV, etc.).
Inputs: data/Russia_World_Bank_2010_2024.csv
Outputs:
- output/tables/descriptive_statistics.csv: Full stats table (N, mean, median, min, max, range, variance, std, Q1, Q3, IQR, CV%).
- output/tables/variability_statistics.csv: Subset focused on dispersion measures (variance, std, CV, range, IQR).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "Russia_World_Bank_2010_2024.csv"

OUTPUT_DIR = PROJECT_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "tables"

DESCRIPTIVE_FILE = (
    TABLE_DIR / "descriptive_statistics.csv"
)

VARIABILITY_FILE = (
    TABLE_DIR / "variability_statistics.csv"
)


IDENTIFIER_COLUMNS = {
    "Year",
    "Country",
    "Country_Code",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    print(f"\n[LOAD] Reading:")
    print(f"       {DATA_FILE}")

    if not DATA_FILE.exists():
        print("[FATAL] Dataset not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_FILE)

    except Exception as exc:
        print(f"[FATAL] Could not load dataset: {exc}")
        sys.exit(1)

    print(
        f"[OK] Dataset loaded: "
        f"{len(df)} rows x {len(df.columns)} columns"
    )

    return df


def get_indicator_columns(df):
    return [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
    ]


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def calculate_statistics(series):
    """
    Calculate assignment-required descriptive statistics.

    Variance and standard deviation use ddof=1:
    sample variance/sample standard deviation.
    """

    data = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if data.empty:
        return None

    n = len(data)

    mean = data.mean()
    median = data.median()

    minimum = data.min()
    maximum = data.max()

    data_range = maximum - minimum

    variance = data.var(ddof=1)
    std_dev = data.std(ddof=1)

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    iqr = q3 - q1

    # CV is usually expressed as %
    # Avoid dividing by zero.
    if mean != 0:
        cv = (std_dev / abs(mean)) * 100
    else:
        cv = np.nan

    return {
        "N": n,
        "Mean": mean,
        "Median": median,
        "Minimum": minimum,
        "Maximum": maximum,
        "Range": data_range,
        "Variance": variance,
        "Standard_Deviation": std_dev,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Coefficient_of_Variation_Percent": cv,
    }


def main():
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    indicators = get_indicator_columns(df)

    print(f"\n[INFO] Indicators detected: {len(indicators)}")

    descriptive_rows = []

    print("\n" + "=" * 70)
    print("CALCULATING STATISTICS")
    print("=" * 70)

    for index, variable in enumerate(
        indicators,
        start=1,
    ):

        print(
            f"\n[{index}/{len(indicators)}] "
            f"{variable}"
        )

        stats = calculate_statistics(
            df[variable]
        )

        if stats is None:
            print(
                "    [WARNING] No valid observations."
            )
            continue

        print(f"    N       = {stats['N']}")
        print(f"    Mean    = {stats['Mean']:.4f}")
        print(f"    Median  = {stats['Median']:.4f}")
        print(
            f"    Std Dev = "
            f"{stats['Standard_Deviation']:.4f}"
        )

        descriptive_rows.append({
            "Variable": variable,
            **stats,
        })

    descriptive_df = pd.DataFrame(
        descriptive_rows
    )

    # Keep full precision internally.
    # CSV will preserve the actual float values.
    try:
        descriptive_df.to_csv(
            DESCRIPTIVE_FILE,
            index=False,
        )

        print("\n[OK] Descriptive statistics saved:")
        print(f"     {DESCRIPTIVE_FILE}")

    except OSError as exc:
        print(
            f"[ERROR] Could not save descriptive table: {exc}"
        )

    # Separate variability table for assignment section 5
    variability_columns = [
        "Variable",
        "N",
        "Variance",
        "Standard_Deviation",
        "Coefficient_of_Variation_Percent",
        "Range",
        "IQR",
    ]

    variability_df = descriptive_df[
        variability_columns
    ].copy()

    try:
        variability_df.to_csv(
            VARIABILITY_FILE,
            index=False,
        )

        print("\n[OK] Variability statistics saved:")
        print(f"     {VARIABILITY_FILE}")

    except OSError as exc:
        print(
            f"[ERROR] Could not save variability table: {exc}"
        )

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS TABLE")
    print("=" * 70)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(
            descriptive_df.round(4).to_string(
                index=False
            )
        )

    print("\nDone.")


if __name__ == "__main__":
    main()