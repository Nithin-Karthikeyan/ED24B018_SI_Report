from pathlib import Path
import sys

import matplotlib

# Use a non-interactive backend.
# This avoids requiring Tcl/Tk when plots are only being saved to files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "Russia_World_Bank_2010_2024.csv"

OUTPUT_DIR = PROJECT_DIR / "output"
IMAGE_DIR = OUTPUT_DIR / "images"

TIME_SERIES_DIR = IMAGE_DIR / "time_series"
HISTOGRAM_DIR = IMAGE_DIR / "histograms"
BOXPLOT_DIR = IMAGE_DIR / "boxplots"
SCATTER_DIR = IMAGE_DIR / "scatterplots"


# ============================================================
# DISPLAY INFORMATION
# ============================================================

VARIABLE_INFO = {

    "GDP_per_capita": {
        "title": "GDP per capita",
        "unit": "Constant 2015 US$ per person",
    },

    "GDP_growth": {
        "title": "GDP growth",
        "unit": "Annual %",
    },

    "Life_expectancy": {
        "title": "Life expectancy at birth",
        "unit": "Years",
    },

    "Basic_drinking_water": {
        "title": "Population using basic drinking-water services",
        "unit": "% of population",
    },

    "Secondary_school_enrollment": {
        "title": "Secondary school enrollment",
        "unit": "% gross",
    },

    "Electricity_access": {
        "title": "Access to electricity",
        "unit": "% of population",
    },

    "Internet_users": {
        "title": "Individuals using the Internet",
        "unit": "% of population",
    },

    "CO2_per_capita": {
        "title": "CO2 emissions per capita",
        "unit": "Metric tons per capita",
    },

    "CO2_per_capita_excl_LULUCF": {
        "title": (
            "CO2 emissions excluding LULUCF per capita"
        ),
        "unit": "t CO2e per capita",
    },

    "Forest_area": {
        "title": "Forest area",
        "unit": "% of land area",
    },
}


# These can be changed after we inspect the distributions.
HISTOGRAM_VARIABLES = [
    "GDP_growth",
    "Internet_users",
]

BOXPLOT_VARIABLES = [
    "GDP_growth",
    "Internet_users",
]


# ============================================================
# FUNCTIONS
# ============================================================

def load_data():
    print("=" * 70)
    print("GRAPH GENERATOR")
    print("=" * 70)

    print(f"\n[LOAD] Reading:")
    print(f"       {DATA_FILE}")

    if not DATA_FILE.exists():
        print("[FATAL] Dataset does not exist.")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_FILE)

    except Exception as exc:
        print(f"[FATAL] Could not load CSV: {exc}")
        sys.exit(1)

    print(f"[OK] Loaded {len(df)} annual observations.")

    return df


def create_directories():

    directories = [
        TIME_SERIES_DIR,
        HISTOGRAM_DIR,
        BOXPLOT_DIR,
        SCATTER_DIR,
    ]

    print("\n[SETUP] Creating graph directories...")

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        print(f"[OK] {directory}")


def get_indicator_columns(df):

    identifiers = {
        "Year",
        "Country",
        "Country_Code",
    }

    return [
        column
        for column in df.columns
        if column not in identifiers
    ]


def get_title(variable):

    if variable in VARIABLE_INFO:
        return VARIABLE_INFO[variable]["title"]

    return variable.replace("_", " ")


def get_unit(variable):

    if variable in VARIABLE_INFO:
        return VARIABLE_INFO[variable]["unit"]

    return "Value"


def add_source_note(fig):
    fig.text(
        0.5,
        0.01,
        "Source: World Bank, World Development Indicators (WDI).",
        ha="center",
        fontsize=8,
    )


def save_figure(fig, path):

    try:
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"    [SAVED] {path.name}")

    except OSError as exc:
        print(
            f"    [ERROR] Could not save {path.name}: {exc}"
        )

    finally:
        plt.close(fig)


# ============================================================
# TIME SERIES
# ============================================================

def plot_time_series(df, variable):

    data = df[
        ["Year", variable]
    ].dropna()

    if data.empty:
        print(
            f"    [SKIP] {variable}: no observations."
        )
        return

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        data["Year"],
        data[variable],
        marker="o",
    )

    ax.set_title(
        f"{get_title(variable)} in Russian Federation, "
        "2010-2024"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(get_unit(variable))

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.set_xticks(
        range(2010, 2025)
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    add_source_note(fig)

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    path = (
        TIME_SERIES_DIR
        / f"{variable}_time_series.png"
    )

    save_figure(fig, path)


# ============================================================
# HISTOGRAMS
# ============================================================

def plot_histogram(df, variable):

    data = pd.to_numeric(
        df[variable],
        errors="coerce",
    ).dropna()

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        data,
        bins="auto",
        edgecolor="black",
    )

    ax.set_title(
        f"Distribution of {get_title(variable)} "
        "in Russian Federation, 2010-2024"
    )

    ax.set_xlabel(get_unit(variable))
    ax.set_ylabel("Frequency")

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    add_source_note(fig)

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    path = (
        HISTOGRAM_DIR
        / f"{variable}_histogram.png"
    )

    save_figure(fig, path)


# ============================================================
# BOXPLOTS
# ============================================================

def plot_boxplot(df, variable):

    data = pd.to_numeric(
        df[variable],
        errors="coerce",
    ).dropna()

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        data,
        vert=True,
    )

    ax.set_title(
        f"Boxplot of {get_title(variable)} "
        "in Russian Federation, 2010-2024"
    )

    ax.set_ylabel(get_unit(variable))

    ax.set_xticks([1])
    ax.set_xticklabels(
        [get_title(variable)]
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    add_source_note(fig)

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    path = (
        BOXPLOT_DIR
        / f"{variable}_boxplot.png"
    )

    save_figure(fig, path)


# ============================================================
# SCATTERPLOTS
# ============================================================

def plot_scatter(
    df,
    x_variable,
    y_variable,
):

    if (
        x_variable not in df.columns
        or y_variable not in df.columns
    ):
        print(
            f"    [SKIP] Missing columns: "
            f"{x_variable}, {y_variable}"
        )
        return

    data = df[
        [
            "Year",
            x_variable,
            y_variable,
        ]
    ].dropna()

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        data[x_variable],
        data[y_variable],
    )

    # Add year labels so individual observations
    # can be investigated easily.
    for _, row in data.iterrows():

        ax.annotate(
            str(int(row["Year"])),
            (
                row[x_variable],
                row[y_variable],
            ),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_title(
        f"{get_title(x_variable)} vs "
        f"{get_title(y_variable)}"
    )

    ax.set_xlabel(
        f"{get_title(x_variable)} "
        f"({get_unit(x_variable)})"
    )

    ax.set_ylabel(
        f"{get_title(y_variable)} "
        f"({get_unit(y_variable)})"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    add_source_note(fig)

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    filename = (
        f"{x_variable}_vs_"
        f"{y_variable}.png"
    )

    path = SCATTER_DIR / filename

    save_figure(fig, path)


def detect_co2_column(df):

    possible_columns = [
        "CO2_per_capita",
        "CO2_per_capita_excl_LULUCF",
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    df = load_data()

    indicators = get_indicator_columns(df)

    print("\n" + "=" * 70)
    print("TIME-SERIES GRAPHS")
    print("=" * 70)

    for index, variable in enumerate(
        indicators,
        start=1,
    ):
        print(
            f"[{index}/{len(indicators)}] "
            f"{variable}"
        )

        plot_time_series(
            df,
            variable,
        )

    print("\n" + "=" * 70)
    print("HISTOGRAMS")
    print("=" * 70)

    for variable in HISTOGRAM_VARIABLES:

        if variable not in df.columns:
            print(
                f"[SKIP] {variable} not found."
            )
            continue

        print(f"[PLOT] {variable}")

        plot_histogram(
            df,
            variable,
        )

    print("\n" + "=" * 70)
    print("BOXPLOTS")
    print("=" * 70)

    for variable in BOXPLOT_VARIABLES:

        if variable not in df.columns:
            print(
                f"[SKIP] {variable} not found."
            )
            continue

        print(f"[PLOT] {variable}")

        plot_boxplot(
            df,
            variable,
        )

    print("\n" + "=" * 70)
    print("REQUIRED SCATTERPLOTS")
    print("=" * 70)

    co2_column = detect_co2_column(df)

    scatter_pairs = [
        (
            "GDP_per_capita",
            "Life_expectancy",
        ),
        (
            "GDP_per_capita",
            "Electricity_access",
        ),
    ]

    if co2_column:
        scatter_pairs.append(
            (
                "GDP_per_capita",
                co2_column,
            )
        )
    else:
        print(
            "[WARNING] No CO2 variable detected."
        )

    for x_variable, y_variable in scatter_pairs:

        print(
            f"[PLOT] {x_variable} vs {y_variable}"
        )

        plot_scatter(
            df,
            x_variable,
            y_variable,
        )

    print("\n" + "=" * 70)
    print("GRAPH GENERATION COMPLETE")
    print("=" * 70)

    print("\nImages saved under:")
    print(f"    {IMAGE_DIR}")


if __name__ == "__main__":
    main()