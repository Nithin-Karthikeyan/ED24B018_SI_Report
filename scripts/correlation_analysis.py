"""
Function: Calculate covariance and Pearson correlation for the three required
assignment variable pairs.

Input:
- data/Russia_World_Bank_2010_2024.csv

Output:
- output/tables/correlation_results.csv

The output contains:
- paired non-missing N
- sample covariance (ddof=1)
- Pearson correlation coefficient r
- two-sided Pearson correlation p-value
"""

from pathlib import Path
import sys

import pandas as pd
from scipy.stats import pearsonr


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "Russia_World_Bank_2010_2024.csv"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "tables"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "correlation_results.csv"
)


# ============================================================
# REQUIRED PAIRS
# ============================================================

CORRELATION_PAIRS = [
    (
        "GDP_per_capita",
        "Life_expectancy",
    ),
    (
        "GDP_per_capita",
        "Electricity_access",
    ),
    (
        "GDP_per_capita",
        "CO2_per_capita_excl_LULUCF",
    ),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("COVARIANCE AND CORRELATION ANALYSIS")
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


# ============================================================
# CALCULATIONS
# ============================================================

def calculate_pair(df, x_var, y_var):
    data = df[
        [x_var, y_var]
    ].copy()

    data[x_var] = pd.to_numeric(
        data[x_var],
        errors="coerce",
    )

    data[y_var] = pd.to_numeric(
        data[y_var],
        errors="coerce",
    )

    data = data.dropna()

    n = len(data)

    if n < 3:
        raise ValueError(
            f"Not enough paired observations for "
            f"{x_var} and {y_var}."
        )

    # Sample covariance: denominator n - 1.
    covariance = data[x_var].cov(
        data[y_var]
    )

    # Pearson r and two-sided significance test.
    result = pearsonr(
        data[x_var],
        data[y_var],
    )

    return {
        "Variable_X": x_var,
        "Variable_Y": y_var,
        "Paired_N": n,
        "Covariance": covariance,
        "Pearson_r": result.statistic,
        "Pearson_p_value": result.pvalue,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    rows = []

    print("\n" + "=" * 70)
    print("CALCULATING REQUIRED PAIRS")
    print("=" * 70)

    for x_var, y_var in CORRELATION_PAIRS:

        print(
            f"\n[PAIR] {x_var} x {y_var}"
        )

        result = calculate_pair(
            df,
            x_var,
            y_var,
        )

        rows.append(result)

        print(
            f"    Paired N   = "
            f"{result['Paired_N']}"
        )
        print(
            f"    Covariance = "
            f"{result['Covariance']:.6f}"
        )
        print(
            f"    Pearson r  = "
            f"{result['Pearson_r']:.6f}"
        )
        print(
            f"    p-value    = "
            f"{result['Pearson_p_value']:.8g}"
        )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n[OK] Correlation results saved:")
    print(f"     {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
