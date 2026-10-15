from pathlib import Path
import json
import sys

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

COUNTRY_CODE = "RUS"
COUNTRY_NAME = "Russian Federation"

START_YEAR = 2010
END_YEAR = 2024

API_BASE_URL = "https://api.worldbank.org/v2"

PROJECT_DIR = Path.home() / "ED24B018_SI_Report"
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_world_bank"

OUTPUT_CSV = DATA_DIR / "Russia_World_Bank_2010_2024.csv"
COVERAGE_CSV = DATA_DIR / "Russia_Data_Coverage.csv"

REQUEST_TIMEOUT = 20


# Required indicators from the Statistical Inference assignment
INDICATORS = {
    "GDP_per_capita": {
        "code": "NY.GDP.PCAP.KD",
        "description": "GDP per capita (constant 2015 US$)",
        "unit": "constant 2015 US$ per person",
    },
    "GDP_growth": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "description": "GDP growth",
        "unit": "annual %",
    },
    "Life_expectancy": {
        "code": "SP.DYN.LE00.IN",
        "description": "Life expectancy at birth",
        "unit": "years",
    },
    "Basic_drinking_water": {
        "code": "SH.H2O.BASW.ZS",
        "description": "People using at least basic drinking-water services",
        "unit": "% of population",
    },
    "Secondary_school_enrollment": {
        "code": "SE.SEC.ENRR",
        "description": "School enrollment, secondary",
        "unit": "% gross",
    },
    "Electricity_access": {
        "code": "EG.ELC.ACCS.ZS",
        "description": "Access to electricity",
        "unit": "% of population",
    },
    "Internet_users": {
        "code": "IT.NET.USER.ZS",
        "description": "Individuals using the Internet",
        "unit": "% of population",
    },
    "CO2_per_capita_excl_LULUCF": {
    "code": "EN.GHG.CO2.PC.CE.AR5",
    "description": "Carbon dioxide (CO2) emissions excluding LULUCF per capita",
    "unit": "t CO2e per capita",
    },
    "Forest_area": {
        "code": "AG.LND.FRST.ZS",
        "description": "Forest area",
        "unit": "% of land area",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_directories():
    """Create output directories if they do not already exist."""

    print("\n[SETUP] Checking project directories...")

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        print(f"[OK] Data directory:     {DATA_DIR}")
        print(f"[OK] Raw data directory: {RAW_DATA_DIR}")

    except OSError as exc:
        print(f"[FATAL] Could not create data directories: {exc}")
        sys.exit(1)


def create_session():
    """
    Create a requests session with automatic retry behaviour.

    Retries are attempted for temporary server-side errors such as
    HTTP 429, 500, 502, 503 and 504.
    """

    print("\n[SETUP] Configuring World Bank API connection...")

    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "ED24B018-Statistical-Inference-Assignment/1.0"
    })

    print("[OK] API session configured with retry support.")

    return session


def save_raw_response(code, response_data):
    """
    Save the complete World Bank API response.

    This preserves the raw downloaded data separately from the cleaned
    analysis dataset, which is useful for reproducibility.
    """

    raw_file = RAW_DATA_DIR / f"RUS_{code}_{START_YEAR}_{END_YEAR}.json"

    try:
        with raw_file.open("w", encoding="utf-8") as file:
            json.dump(response_data, file, indent=2, ensure_ascii=False)

        print(f"    [RAW] Saved API response -> {raw_file.name}")

    except OSError as exc:
        print(f"    [WARNING] Could not save raw API response: {exc}")


def fetch_indicator(session, variable_name, indicator_info):
    """
    Download one indicator from the World Bank API.

    Returns:
        dict: Mapping {year: value}

    Raises:
        RuntimeError if the request or API response cannot be processed.
    """

    code = indicator_info["code"]

    url = (
        f"{API_BASE_URL}/country/{COUNTRY_CODE}/indicator/{code}"
        f"?format=json"
        f"&date={START_YEAR}:{END_YEAR}"
        f"&per_page=100"
    )

    print(f"    API URL: {url}")

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        # Converts HTTP error responses such as 404 / 500 into exceptions.
        response.raise_for_status()

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Request timed out after {REQUEST_TIMEOUT} seconds."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to the World Bank API. "
            "Check the internet connection."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"World Bank API returned HTTP error: {exc}"
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Unexpected network/request error: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Decode JSON
    # --------------------------------------------------------

    try:
        api_data = response.json()

    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(
            "World Bank returned a response that could not be decoded as JSON."
        ) from exc

    # Save untouched response for reproducibility
    save_raw_response(code, api_data)

    # --------------------------------------------------------
    # Validate API structure
    # --------------------------------------------------------

    if not isinstance(api_data, list):
        raise RuntimeError(
            "Unexpected World Bank API response format: expected a list."
        )

    if len(api_data) < 2:
        raise RuntimeError(
            "World Bank API response is incomplete."
        )

    metadata = api_data[0]
    observations = api_data[1]

    if observations is None:
        raise RuntimeError(
            f"No observations returned for indicator {code}."
        )

    if not isinstance(observations, list):
        raise RuntimeError(
            f"Unexpected observation structure for indicator {code}."
        )

    if len(observations) == 0:
        raise RuntimeError(
            f"World Bank returned zero observations for indicator {code}."
        )

    if isinstance(metadata, dict):
        total = metadata.get("total", "unknown")
        print(f"    API reports {total} observations.")

    # --------------------------------------------------------
    # Extract year/value pairs
    # --------------------------------------------------------

    values = {}

    for observation in observations:

        try:
            year_raw = observation.get("date")
            value_raw = observation.get("value")

            if year_raw is None:
                print("    [WARNING] Observation had no year. Skipping.")
                continue

            year = int(year_raw)

            if not START_YEAR <= year <= END_YEAR:
                continue

            # Missing World Bank observations are deliberately retained
            # as None rather than interpolated.
            if value_raw is None:
                values[year] = None
                continue

            value = float(value_raw)

            values[year] = value

        except (TypeError, ValueError, AttributeError) as exc:
            print(
                "    [WARNING] Could not parse observation "
                f"{observation}: {exc}"
            )

    return values


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("WORLD BANK DATA DOWNLOADER")
    print("Statistical Inference Assignment")
    print("=" * 70)

    print(f"\nCountry : {COUNTRY_NAME} ({COUNTRY_CODE})")
    print(f"Period  : {START_YEAR}-{END_YEAR}")
    print(f"Variables: {len(INDICATORS)}")

    create_directories()

    session = create_session()

    # One row per year
    df = pd.DataFrame({
        "Year": range(START_YEAR, END_YEAR + 1)
    })

    # Not exactly needed, so commented out for now
    # df.insert(1, "Country", COUNTRY_NAME)
    # df.insert(2, "Country_Code", COUNTRY_CODE)

    coverage_records = []
    failed_indicators = []

    print("\n" + "=" * 70)
    print("DOWNLOADING INDICATORS")
    print("=" * 70)

    total_indicators = len(INDICATORS)

    for index, (variable_name, info) in enumerate(
        INDICATORS.items(),
        start=1,
    ):

        print("\n" + "-" * 70)
        print(
            f"[{index}/{total_indicators}] "
            f"{info['description']}"
        )
        print(f"    Variable : {variable_name}")
        print(f"    Code     : {info['code']}")
        print(f"    Unit     : {info['unit']}")

        try:
            values = fetch_indicator(
                session=session,
                variable_name=variable_name,
                indicator_info=info,
            )

            df[variable_name] = df["Year"].map(values)

            non_missing = int(df[variable_name].notna().sum())
            missing = int(df[variable_name].isna().sum())

            missing_years = (
                df.loc[
                    df[variable_name].isna(),
                    "Year"
                ]
                .astype(int)
                .tolist()
            )

            print(f"    [OK] Non-missing observations : {non_missing}")
            print(f"    [OK] Missing observations     : {missing}")

            if missing_years:
                print(
                    "    [INFO] Missing years           : "
                    + ", ".join(map(str, missing_years))
                )
            else:
                print("    [INFO] Missing years           : None")

            coverage_records.append({
                "Variable": variable_name,
                "Indicator_Name": info["description"],
                "World_Bank_Code": info["code"],
                "Unit": info["unit"],
                "Potential_Observations": END_YEAR - START_YEAR + 1,
                "Non_Missing": non_missing,
                "Missing": missing,
                "Missing_Years": ", ".join(map(str, missing_years)),
                "Status": "OK",
            })

        except RuntimeError as exc:

            print(f"    [ERROR] Failed to download indicator.")
            print(f"    [ERROR] Reason: {exc}")

            # Keep the column in the final dataset so that failure is visible
            df[variable_name] = pd.NA

            failed_indicators.append(variable_name)

            coverage_records.append({
                "Variable": variable_name,
                "Indicator_Name": info["description"],
                "World_Bank_Code": info["code"],
                "Unit": info["unit"],
                "Potential_Observations": END_YEAR - START_YEAR + 1,
                "Non_Missing": 0,
                "Missing": END_YEAR - START_YEAR + 1,
                "Missing_Years": "DOWNLOAD FAILED",
                "Status": "DOWNLOAD FAILED",
            })


    # ========================================================
    # DATA QUALITY CHECKS
    # ========================================================

    print("\n" + "=" * 70)
    print("BASIC DATA QUALITY CHECK")
    print("=" * 70)

    expected_rows = END_YEAR - START_YEAR + 1

    print(f"\nExpected years : {expected_rows}")
    print(f"Dataset rows   : {len(df)}")

    if len(df) == expected_rows:
        print("[OK] Correct number of annual observations.")
    else:
        print("[WARNING] Unexpected number of rows.")

    duplicate_years = df["Year"].duplicated().sum()

    if duplicate_years == 0:
        print("[OK] No duplicate years.")
    else:
        print(f"[WARNING] Found {duplicate_years} duplicate years.")

    print("\nMissing observations by variable:")

    for variable_name in INDICATORS:
        missing = int(df[variable_name].isna().sum())
        print(f"    {variable_name:<32} : {missing}")


    # ========================================================
    # SAVE CLEAN DATASET
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)

    try:
        df.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8",
        )

        print(f"[OK] Clean dataset saved to:")
        print(f"     {OUTPUT_CSV}")

    except (OSError, PermissionError) as exc:
        print(f"[FATAL] Could not save main CSV: {exc}")
        sys.exit(1)


    # ========================================================
    # SAVE COVERAGE REPORT
    # ========================================================

    coverage_df = pd.DataFrame(coverage_records)

    try:
        coverage_df.to_csv(
            COVERAGE_CSV,
            index=False,
            encoding="utf-8",
        )

        print(f"\n[OK] Data coverage report saved to:")
        print(f"     {COVERAGE_CSV}")

    except (OSError, PermissionError) as exc:
        print(f"[WARNING] Could not save coverage report: {exc}")


    # ========================================================
    # DISPLAY FINAL DATASET
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL CLEAN DATASET")
    print("=" * 70)

    print(df.to_string(index=False))


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    successful = total_indicators - len(failed_indicators)

    print(f"Successful indicators : {successful}/{total_indicators}")
    print(f"Failed indicators     : {len(failed_indicators)}")

    if failed_indicators:
        print("\nFailed variables:")

        for variable in failed_indicators:
            print(f"    - {variable}")

        print(
            "\n[WARNING] Dataset was saved, but some indicators "
            "could not be downloaded."
        )

    else:
        print("\n[SUCCESS] All World Bank indicators downloaded successfully.")

    print("\nFiles created:")

    print(f"    Clean dataset   : {OUTPUT_CSV}")
    print(f"    Coverage report : {COVERAGE_CSV}")
    print(f"    Raw API data    : {RAW_DATA_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\n[STOPPED] Program interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        # Final fallback for unexpected programming/runtime errors.
        print("\n[FATAL] Unexpected error occurred.")
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        sys.exit(1)