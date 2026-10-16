# Nithin | ED24B018 | Statistical Inference Report — Russia

**Course:** HS5703 — Statistical Inference  
**Student:** K. Nithin (ED24B018)  
**Country:** Russian Federation (`RUS`)  
**Period:** 2010–2024  
**Data source:** World Bank World Development Indicators (WDI)

Repository: https://github.com/Nithin-Karthikeyan/ED24B018_SI_Report

This repository contains the data, Python scripts, generated tables, figures, and reproducible calculations used for the Statistical Inference assignment on Russia.

The analysis uses nine World Bank indicators. For CO2 emissions, the instructor-approved substitute `EN.GHG.CO2.PC.CE.AR5` (CO2 emissions excluding LULUCF per capita) is used because the originally specified series is unavailable for Russia over the required period.

---

## Repository Structure

```text
ED24B018_SI_Report/
├── data/
│   ├── raw_world_bank/                  # Raw World Bank API responses + metadata
│   ├── Russia_World_Bank_2010_2024.csv # Main analysis dataset
│   ├── Russia_Data_Coverage.csv         # Observation/missing-data coverage
│   └── indicator_metadata.csv           # Indicator definitions, units and metadata
│
├── scripts/
│   ├── extract_data.py                  # Download and construct the dataset
│   ├── check_data_quality.py            # Validate data and flag unusual changes
│   ├── descriptive_statistics.py        # Descriptive and variability statistics
│   ├── plot_graphs.py                   # Generate all report figures
│   ├── inferential_statistics.py        # Confidence intervals + hypothesis tests
│   └── correlation_analysis.py          # Covariance + Pearson correlations
│
├── output/
│   ├── tables/                          # Numerical outputs used in the report
│   ├── reports/                         # Data-quality text report
│   └── images/
│       ├── time_series/                 # Nine time-series plots
│       ├── histograms/                  # Selected histograms
│       ├── boxplots/                    # Selected boxplots
│       └── scatterplots/                # Required correlation scatterplots
│
├── src/
│   └── ed24b018_si_report/              # Minimal Python package scaffold
│
├── pyproject.toml                       # Project metadata and dependencies
├── uv.lock                              # Reproducible dependency lock file
└── README.md                            # Repository guide
```

### Folder Summary

- **`data/`** — cleaned analysis data, coverage information, metadata, and untouched API responses.
- **`scripts/`** — all Python code used to produce the statistical results and plots.
- **`output/tables/`** — machine-generated CSV tables used for numerical results in the report.
- **`output/reports/`** — detailed human-readable data-quality output.
- **`output/images/`** — all figures used in the graphical analysis.
- **`src/`** — package scaffold; the statistical workflow is currently run directly from `scripts/`.

---

## Scripts

### `scripts/extract_data.py`

**Function:**  
Downloads the required World Bank indicators for Russia for 2010–2024, stores untouched API responses, constructs the analysis dataset, and retrieves indicator metadata.

**Input:**  
- World Bank Indicators API
- No local data file required

**Outputs:**
- `data/Russia_World_Bank_2010_2024.csv`
- `data/Russia_Data_Coverage.csv`
- `data/indicator_metadata.csv`
- `data/raw_world_bank/*.json`

---

### `scripts/check_data_quality.py`

**Function:**  
Checks dataset structure, missing values, numeric validity, plausible ranges, unit consistency, duplicate/missing years, and unusual year-to-year changes. Unexpected changes are screened using a MAD-based modified z-score together with practical variable-specific thresholds.

Flagged observations are retained as **review items** and are not automatically removed.

**Inputs:**
- `data/Russia_World_Bank_2010_2024.csv`
- `data/indicator_metadata.csv`

**Outputs:**
- `output/tables/data_quality.csv`
- `output/tables/unexpected_changes.csv`
- `output/reports/data_quality_report.txt`

---

### `scripts/descriptive_statistics.py`

**Function:**  
Calculates descriptive statistics for all nine indicators and produces the dispersion measures used in the variance and heterogeneity analysis.

Statistics include:
- N
- mean and median
- minimum and maximum
- range
- variance and standard deviation
- Q1 and Q3
- IQR
- coefficient of variation (CV)

**Input:**
- `data/Russia_World_Bank_2010_2024.csv`

**Outputs:**
- `output/tables/descriptive_statistics.csv`
- `output/tables/variability_statistics.csv`

---

### `scripts/plot_graphs.py`

**Function:**  
Generates all figures used in the graphical analysis.

**Input:**
- `data/Russia_World_Bank_2010_2024.csv`

**Outputs:**
- `output/images/time_series/*.png`
- `output/images/histograms/*.png`
- `output/images/boxplots/*.png`
- `output/images/scatterplots/*.png`

Current generated figures include:
- 9 time-series plots
- 2 histograms
- 2 boxplots
- 3 required correlation scatterplots

---

### `scripts/inferential_statistics.py`

**Function:**  
Reproduces the inferential calculations used in Sections 6 and 7 of the report.

It calculates:
- 95% one-sample Student's t confidence intervals
- one-sample t-test for mean GDP growth against zero
- Spearman rank test for the positive association between Year and Internet use

**Input:**
- `data/Russia_World_Bank_2010_2024.csv`

**Outputs:**
- `output/tables/confidence_intervals.csv`
- `output/tables/hypothesis_tests.csv`

---

### `scripts/correlation_analysis.py`

**Function:**  
Calculates covariance and Pearson correlation for the three required pairs:

1. GDP per capita × Life expectancy
2. GDP per capita × Electricity access
3. GDP per capita × CO2 emissions excluding LULUCF per capita

It reports paired N, sample covariance, Pearson's `r`, and the two-sided Pearson correlation p-value.

**Input:**
- `data/Russia_World_Bank_2010_2024.csv`

**Output:**
- `output/tables/correlation_results.csv`

---

## Generated Outputs

### `output/tables/`

| File | Contents |
|---|---|
| `descriptive_statistics.csv` | Full-precision descriptive statistics for all nine indicators |
| `variability_statistics.csv` | Variance, SD, CV, range and IQR |
| `data_quality.csv` | Coverage, missingness, minima/maxima and detected issues |
| `unexpected_changes.csv` | Annual changes flagged for review by the quality screen |
| `confidence_intervals.csv` | 95% t-confidence intervals used in Section 6 |
| `hypothesis_tests.csv` | Test statistics, p-values and decisions used in Section 7 |
| `correlation_results.csv` | Paired N, covariance, Pearson r and p-values used in Section 8 |

All calculations retain full numerical precision in these files. Values are rounded only when displayed in the report.

### `output/reports/`

`data_quality_report.txt` gives a readable summary of:
- dataset structure
- observation coverage
- unit checks
- missing data
- unusual annual changes requiring review

### `output/images/`

The four subfolders contain the figures used directly in the report:
- `time_series/`
- `histograms/`
- `boxplots/`
- `scatterplots/`

---

## Running the Analysis

The project uses `uv` for dependency management.

Install/synchronise dependencies:

```bash
uv sync
```

Run the workflow in order:

```bash
uv run python scripts/extract_data.py
uv run python scripts/check_data_quality.py
uv run python scripts/descriptive_statistics.py
uv run python scripts/plot_graphs.py
uv run python scripts/inferential_statistics.py
uv run python scripts/correlation_analysis.py
```

The main Python dependencies are:
- `pandas`
- `matplotlib`
- `scipy`

---

## Report-to-Code Mapping

| Report component | Reproducible source |
|---|---|
| Data construction and metadata | `extract_data.py` |
| Data-quality checks | `check_data_quality.py` |
| Descriptive statistics | `descriptive_statistics.py` |
| Graphical analysis | `plot_graphs.py` |
| Variance and heterogeneity | `descriptive_statistics.py` |
| Confidence intervals | `inferential_statistics.py` |
| Hypothesis tests | `inferential_statistics.py` |
| Covariance and correlation | `correlation_analysis.py` |
| Confounding interpretation | Interpretation of the statistical results; no causal model is fitted |

---

## Important Current Consistency Check

The current `extract_data.py` has the insertion of `Country` and `Country_Code` commented out. Therefore the present analysis CSV contains `Year` plus the nine indicator columns, while `check_data_quality.py` expects `Country` and `Country_Code` and currently flags them as missing in `data_quality_report.txt`.

Before final submission, either:
1. re-enable the `Country` and `Country_Code` columns in `extract_data.py` and rerun the workflow, or
2. make the report and quality-check expectations consistent with the final submitted dataset.

The assignment specification expects `Year`, `Country`, `Country Code`, and the nine indicators, so option 1 is the cleaner final setup.