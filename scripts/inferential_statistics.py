"""
Function: Reproduce the confidence intervals and hypothesis tests used in
Sections 6 and 7 of the report.

Input:
- data/Russia_World_Bank_2010_2024.csv

Outputs:
- output/tables/confidence_intervals.csv
- output/tables/hypothesis_tests.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import (
    spearmanr,
    t,
    ttest_1samp,
)


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

CI_OUTPUT_FILE = (
    OUTPUT_DIR
    / "confidence_intervals.csv"
)

TEST_OUTPUT_FILE = (
    OUTPUT_DIR
    / "hypothesis_tests.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("INFERENTIAL STATISTICS")
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
# CONFIDENCE INTERVAL
# ============================================================

def mean_t_confidence_interval(series, confidence=0.95):
    data = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    n = len(data)
    dfree = n - 1

    mean = data.mean()
    std_dev = data.std(ddof=1)
    standard_error = std_dev / np.sqrt(n)

    alpha = 1.0 - confidence

    t_critical = t.ppf(
        1.0 - alpha / 2.0,
        dfree,
    )

    margin_error = (
        t_critical
        * standard_error
    )

    lower = mean - margin_error
    upper = mean + margin_error

    return {
        "N": n,
        "Mean": mean,
        "Standard_Deviation": std_dev,
        "Degrees_of_Freedom": dfree,
        "Standard_Error": standard_error,
        "t_Critical": t_critical,
        "Margin_of_Error": margin_error,
        "CI_Lower": lower,
        "CI_Upper": upper,
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

    # --------------------------------------------------------
    # SECTION 6: 95% CONFIDENCE INTERVALS
    # --------------------------------------------------------

    ci_variables = [
        "GDP_growth",
        "CO2_per_capita_excl_LULUCF",
    ]

    ci_rows = []

    print("\n" + "=" * 70)
    print("95% CONFIDENCE INTERVALS")
    print("=" * 70)

    for variable in ci_variables:

        result = mean_t_confidence_interval(
            df[variable],
            confidence=0.95,
        )

        ci_rows.append({
            "Variable": variable,
            "Confidence_Level": 0.95,
            **result,
        })

        print(f"\n[CI] {variable}")
        print(
            f"    Mean = "
            f"{result['Mean']:.8f}"
        )
        print(
            f"    SE   = "
            f"{result['Standard_Error']:.8f}"
        )
        print(
            f"    CI   = "
            f"[{result['CI_Lower']:.8f}, "
            f"{result['CI_Upper']:.8f}]"
        )

    pd.DataFrame(
        ci_rows
    ).to_csv(
        CI_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SECTION 7: HYPOTHESIS TEST 1
    # GDP growth mean against zero
    # --------------------------------------------------------

    gdp_growth = pd.to_numeric(
        df["GDP_growth"],
        errors="coerce",
    ).dropna()

    gdp_test = ttest_1samp(
        gdp_growth,
        popmean=0.0,
        alternative="two-sided",
    )

    # --------------------------------------------------------
    # SECTION 7: HYPOTHESIS TEST 2
    # Positive monotonic association: Year vs Internet users
    # --------------------------------------------------------

    internet_data = df[
        ["Year", "Internet_users"]
    ].copy()

    internet_data["Year"] = pd.to_numeric(
        internet_data["Year"],
        errors="coerce",
    )

    internet_data["Internet_users"] = pd.to_numeric(
        internet_data["Internet_users"],
        errors="coerce",
    )

    internet_data = internet_data.dropna()

    spearman_test = spearmanr(
        internet_data["Year"],
        internet_data["Internet_users"],
        alternative="greater",
    )

    test_rows = [
        {
            "Test": (
                "One-sample t-test: "
                "GDP growth mean vs 0"
            ),
            "N": len(gdp_growth),
            "Alternative": "two-sided",
            "Test_Statistic": gdp_test.statistic,
            "p_value": gdp_test.pvalue,
            "Alpha": 0.05,
            "Decision": (
                "Reject H0"
                if gdp_test.pvalue < 0.05
                else "Fail to reject H0"
            ),
        },
        {
            "Test": (
                "Spearman rank test: "
                "Year vs Internet users"
            ),
            "N": len(internet_data),
            "Alternative": "greater",
            "Test_Statistic": spearman_test.statistic,
            "p_value": spearman_test.pvalue,
            "Alpha": 0.05,
            "Decision": (
                "Reject H0"
                if spearman_test.pvalue < 0.05
                else "Fail to reject H0"
            ),
        },
    ]

    pd.DataFrame(
        test_rows
    ).to_csv(
        TEST_OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    for row in test_rows:
        print(f"\n[TEST] {row['Test']}")
        print(
            f"    Statistic = "
            f"{row['Test_Statistic']:.8f}"
        )
        print(
            f"    p-value   = "
            f"{row['p_value']:.8g}"
        )
        print(
            f"    Decision  = "
            f"{row['Decision']}"
        )

    print("\n[OK] Saved:")
    print(f"     {CI_OUTPUT_FILE}")
    print(f"     {TEST_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
