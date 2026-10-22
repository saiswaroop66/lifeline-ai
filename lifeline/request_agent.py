"""
LifeLine - Blood Request Agent
------------------------------
Features:
- Groq-powered request parsing with local fallback
- Blood group, component, units, and location extraction
- SQLite search using data/lifeline.db
- Safe handling of text, missing quantities, and availability statuses
- Works with the existing inventory table

Important:
Database results are saved snapshots, not live inventory.
This tool does not determine clinical compatibility or authorize transfusions.
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "lifeline.db"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
).strip()


# ============================================================
# NORMALIZATION
# ============================================================

BLOOD_GROUPS = {
    "A+": ["a+", "a positive", "a pos", "a plus"],
    "A-": ["a-", "a negative", "a neg", "a minus"],
    "B+": ["b+", "b positive", "b pos", "b plus"],
    "B-": ["b-", "b negative", "b neg", "b minus"],
    "AB+": ["ab+", "ab positive", "ab pos", "ab plus"],
    "AB-": ["ab-", "ab negative", "ab neg", "ab minus"],
    "O+": ["o+", "o positive", "o pos", "o plus"],
    "O-": ["o-", "o negative", "o neg", "o minus"],
}

COMPONENTS = {
    "whole blood": [
        "whole blood",
        "whole-blood",
        "wholeblood",
    ],
    "packed red blood cells": [
        "packed red blood cells",
        "packed red cells",
        "red blood cells",
        "red cells",
        "prbc",
        "rbc",
    ],
    "plasma": [
        "plasma",
        "fresh frozen plasma",
        "ffp",
    ],
    "platelets": [
        "platelets",
        "platelet",
    ],
    "cryoprecipitate": [
        "cryoprecipitate",
        "cryo",
    ],
}


def normalize_text(value: Any) -> str:
    """Convert a value into normalized lowercase text."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_blood_group(value: Any) -> Optional[str]:
    """Normalize common blood-group expressions."""
    text = normalize_text(value)

    if not text:
        return None

    compact = re.sub(r"\s+", "", text)

    direct = {
        "a+": "A+",
        "a-": "A-",
        "b+": "B+",
        "b-": "B-",
        "ab+": "AB+",
        "ab-": "AB-",
        "o+": "O+",
        "o-": "O-",
    }

    if compact in direct:
        return direct[compact]

    for group, aliases in BLOOD_GROUPS.items():
        for alias in aliases:
            pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return group

    return None


def normalize_component(value: Any) -> Optional[str]:
    """Normalize component names."""
    text = normalize_text(value)

    if not text:
        return None

    # Prefer the longest alias first to avoid partial matches.
    aliases = []

    for canonical, variants in COMPONENTS.items():
        for alias in variants:
            aliases.append((alias, canonical))

    aliases.sort(key=lambda item: len(item[0]), reverse=True)

    for alias, canonical in aliases:
        if alias in text:
            return canonical

    return None


def parse_units(value: Any) -> Optional[float]:
    """
    Extract a numeric quantity.
    Returns None when the quantity is unknown.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None

    text = normalize_text(value)

    if not text:
        return None

    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    for word, number in word_numbers.items():
        if re.search(r"\b" + word + r"\b", text):
            return float(number)

    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


def normalize_status(value: Any) -> str:
    """Normalize availability status without guessing unknown values."""
    text = normalize_text(value)

    if not text:
        return "Unknown"

    unavailable_values = {
        "unavailable",
        "not available",
        "out of stock",
        "not in stock",
        "no stock",
        "zero",
        "0",
        "false",
        "no",
    }

    available_values = {
        "available",
        "in stock",
        "stock available",
        "true",
        "yes",
    }

    unknown_values = {
        "unknown",
        "not reported",
        "not known",
        "pending",
        "n/a",
        "na",
        "none",
        "null",
    }

    if text in unavailable_values:
        return "Unavailable"

    if text in available_values:
        return "Available"

    if text in unknown_values:
        return "Unknown"

    # Avoid interpreting arbitrary text as confirmed availability.
    return "Unknown"


def extract_location(text: str) -> Optional[str]:
    """Conservatively extract common location phrases."""
    patterns = [
        r"\b(?:in|near|around|at)\s+(visakhapatnam|vizag)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            location = match.group(1).lower()

            if location in {"vizag", "visakhapatnam"}:
                return "Visakhapatnam"

            return location.title()

    return None


# ============================================================
# LOCAL REQUEST PARSER
# ============================================================

def _parse_blood_request_rules(message: str) -> Dict[str, Any]:
    """Fallback parser that does not require an API key."""
    original = str(message or "").strip()
    text = normalize_text(original)

    component = normalize_component(text)
    blood_group = normalize_blood_group(text)
    location = extract_location(text)

    units = None

    quantity_patterns = [
        r"\b(\d+(?:\.\d+)?)\s*(?:units?|bags?|bottles?)\b",
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s*(?:units?|bags?|bottles?)\b",
    ]

    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    for pattern in quantity_patterns:
        match = re.search(pattern, text)

        if match:
            raw_quantity = match.group(1)

            if raw_quantity in word_numbers:
                units = float(word_numbers[raw_quantity])
            else:
                units = parse_units(raw_quantity)

            break

    missing = []

    if component is None:
        missing.append("component")

    if blood_group is None:
        missing.append("blood_group")

    return {
        "original_request": original,
        "component": component,
        "blood_group": blood_group,
        "units": units,
        "location": location,
        "missing": missing,
        "valid": len(missing) == 0,
        "source": "rules_fallback",
    }


# ============================================================
# GROQ REQUEST PARSER
# ============================================================

def _get_groq_client():
    """Return a Groq client when configuration is available."""
    if not GROQ_API_KEY or Groq is None:
        return None

    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None


def parse_blood_request(message: str) -> Dict[str, Any]:
    """
    Parse a natural-language request with Groq.
    Falls back to local rules if Groq fails.
    """
    original = str(message or "").strip()

    if not original:
        return {
            "original_request": "",
            "component": None,
            "blood_group": None,
            "units": None,
            "location": None,
            "missing": ["component", "blood_group"],
            "valid": False,
            "source": "rules_fallback",
            "parser_note": "The request was empty.",
        }

    client = _get_groq_client()

    if client is None:
        result = _parse_blood_request_rules(original)
        result["parser_note"] = (
            "Groq is not configured or the groq package is unavailable."
        )
        return result

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
You extract structured information from blood-bank search requests.

Return only a JSON object with these fields:
{
  "component": string or null,
  "blood_group": string or null,
  "units": number or null,
  "location": string or null
}

Allowed canonical components:
- whole blood
- packed red blood cells
- plasma
- platelets
- cryoprecipitate

Allowed blood groups:
A+, A-, B+, B-, AB+, AB-, O+, O-

Rules:
- Do not invent missing information.
- "three bags" means quantity 3.
- "B positive" means B+.
- "O negative" means O-.
- "Vizag" means Visakhapatnam.
- If the user does not specify a quantity, return null.
- Do not make clinical compatibility decisions.
- Do not claim that blood is available.
""".strip(),
                },
                {
                    "role": "user",
                    "content": original,
                },
            ],
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Validate and normalize model output.
        component = normalize_component(data.get("component"))
        blood_group = normalize_blood_group(data.get("blood_group"))
        units = parse_units(data.get("units"))

        location_value = data.get("location")
        location = (
            str(location_value).strip()
            if location_value is not None
            and str(location_value).strip()
            else None
        )

        if location and normalize_text(location) == "vizag":
            location = "Visakhapatnam"

        missing = []

        if component is None:
            missing.append("component")

        if blood_group is None:
            missing.append("blood_group")

        return {
            "original_request": original,
            "component": component,
            "blood_group": blood_group,
            "units": units,
            "location": location,
            "missing": missing,
            "valid": len(missing) == 0,
            "source": "groq",
        }

    except Exception as exc:
        result = _parse_blood_request_rules(original)
        result["parser_note"] = (
            f"Groq request failed; using local parser. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return result


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_connection():
    """Open the LifeLine SQLite database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Check that data/lifeline.db exists."
        )

    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def load_inventory() -> pd.DataFrame:
    """Read inventory rows from the existing database."""
    with get_connection() as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
            connection,
        )["name"].tolist()

        if "inventory" not in tables:
            raise RuntimeError(
                "The database does not contain an 'inventory' table."
            )

        df = pd.read_sql_query(
            "SELECT * FROM inventory",
            connection,
        )

    return df


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list,
) -> Optional[str]:
    """Return the first matching column name."""
    lower_map = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def normalize_available_series(series: pd.Series) -> pd.Series:
    """
    Convert status strings to a proper Boolean Series.

    This is the fix for the Arrow large_string AND bool error.
    Only explicitly reported available statuses become True.
    """
    normalized = series.astype("string").str.strip().str.lower()

    true_values = {
        "true",
        "1",
        "yes",
        "available",
        "in stock",
        "stock available",
    }

    return normalized.isin(true_values).astype(bool)


# ============================================================
# INVENTORY SEARCH
# ============================================================

def search_blood_records(
    request: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Search saved inventory records by requested component/group.

    Returns matching explicitly available records and records
    with lower quantities. Unknown quantity/status is not treated
    as confirmed availability.
    """
    if not request.get("valid"):
        return {
            "successful": False,
            "error": "Request is missing a component or blood group.",
            "matches": pd.DataFrame(),
            "lower_quantity_records": pd.DataFrame(),
            "unknown_records": pd.DataFrame(),
            "records_checked": 0,
        }

    try:
        df = load_inventory()

        if df.empty:
            return {
                "successful": True,
                "error": None,
                "matches": pd.DataFrame(),
                "lower_quantity_records": pd.DataFrame(),
                "unknown_records": pd.DataFrame(),
                "records_checked": 0,
            }

        component_col = first_existing_column(
            df,
            ["component", "blood_component", "product"],
        )

        group_col = first_existing_column(
            df,
            ["blood_group", "blood_type", "group"],
        )

        status_col = first_existing_column(
            df,
            [
                "reported_status",
                "availability_status",
                "status",
                "availability",
            ],
        )

        units_col = first_existing_column(
            df,
            [
                "units_listed",
                "units",
                "quantity",
                "units_available",
                "available_units",
                "stock",
            ],
        )

        location_col = first_existing_column(
            df,
            [
                "facility_name",
                "blood_bank_name",
                "hospital_name",
                "name",
            ],
        )

        if component_col is None or group_col is None:
            return {
                "successful": False,
                "error": (
                    "Could not find component and blood-group columns "
                    f"in inventory. Columns found: {list(df.columns)}"
                ),
                "matches": pd.DataFrame(),
                "lower_quantity_records": pd.DataFrame(),
                "unknown_records": pd.DataFrame(),
                "records_checked": len(df),
            }

        # Normalize data into helper columns.
        df = df.copy()

        df["_normalized_component"] = df[component_col].map(
            normalize_component
        )

        df["_normalized_group"] = df[group_col].map(
            normalize_blood_group
        )

        component = normalize_component(request.get("component"))
        blood_group = normalize_blood_group(
            request.get("blood_group")
        )

        mask = (
            df["_normalized_component"].eq(component)
            & df["_normalized_group"].eq(blood_group)
        )

        location = request.get("location")

        if location and location_col:
            location_text = normalize_text(location)

            location_mask = df[location_col].map(
                lambda value: location_text in normalize_text(value)
                or (
                    location_text == "visakhapatnam"
                    and "vizag" in normalize_text(value)
                )
            )

            # Only apply the location filter if it finds records.
            if (mask & location_mask).any():
                mask = mask & location_mask

        matches = df.loc[mask].copy()

        if status_col:
            matches["_normalized_status"] = matches[status_col].map(
                normalize_status
            )
        else:
            matches["_normalized_status"] = "Unknown"

        # Always use a Boolean dtype for availability.
        matches["_reported_available"] = (
            matches["_normalized_status"].eq("Available").astype(bool)
        )

        if units_col:
            matches["_numeric_units"] = pd.to_numeric(
                matches[units_col]
                .astype("string")
                .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
                errors="coerce",
            )
        else:
            matches["_numeric_units"] = float("nan")

        requested_units = request.get("units")

        if requested_units is not None:
            requested_units = float(requested_units)

            enough_units_mask = (
                matches["_reported_available"]
                & matches["_numeric_units"].notna()
                & (matches["_numeric_units"] >= requested_units)
            )

            lower_units_mask = (
                matches["_reported_available"]
                & matches["_numeric_units"].notna()
                & (matches["_numeric_units"] < requested_units)
            )

            unknown_mask = (
                ~matches["_reported_available"]
                | matches["_numeric_units"].isna()
            )

            enough_units = matches.loc[enough_units_mask].copy()
            lower_quantity = matches.loc[lower_units_mask].copy()
            unknown_records = matches.loc[unknown_mask].copy()

        else:
            enough_units_mask = matches["_reported_available"]

            enough_units = matches.loc[enough_units_mask].copy()

            lower_quantity = pd.DataFrame(
                columns=matches.columns
            )

            unknown_records = matches.loc[
                ~matches["_reported_available"]
                | matches["_numeric_units"].isna()
            ].copy()

        return {
            "successful": True,
            "error": None,
            "matches": enough_units,
            "lower_quantity_records": lower_quantity,
            "unknown_records": unknown_records,
            "records_checked": len(df),
            "matching_records": len(matches),
        }

    except Exception as exc:
        return {
            "successful": False,
            "error": f"{type(exc).__name__}: {exc}",
            "matches": pd.DataFrame(),
            "lower_quantity_records": pd.DataFrame(),
            "unknown_records": pd.DataFrame(),
            "records_checked": 0,
        }


# ============================================================
# COMPLETE REQUEST WORKFLOW
# ============================================================

def process_blood_request(message: str) -> Dict[str, Any]:
    """Parse a request and search the database."""
    parsed = parse_blood_request(message)

    if not parsed.get("valid"):
        return {
            "parsed_request": parsed,
            "search_result": {
                "successful": False,
                "error": "Please specify both a blood component and blood group.",
                "matches": pd.DataFrame(),
                "lower_quantity_records": pd.DataFrame(),
                "unknown_records": pd.DataFrame(),
                "records_checked": 0,
            },
        }

    result = search_blood_records(parsed)

    return {
        "parsed_request": parsed,
        "search_result": result,
    }


# ============================================================
# COMMAND-LINE TESTS
# ============================================================

def print_request_result(message: str) -> None:
    """Print a readable test result."""
    print("\n" + "=" * 65)
    print(f"Request: {message}")

    outcome = process_blood_request(message)

    parsed = outcome["parsed_request"]
    result = outcome["search_result"]

    print("\nParsed request:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    print("\nSearch successful:", result.get("successful"))
    print("Error:", result.get("error"))
    print("Records checked:", result.get("records_checked", 0))

    if not result.get("successful"):
        return

    print("Matching component/group records:", result.get("matching_records", 0))
    print(
        "Records meeting requested quantity:",
        len(result.get("matches", [])),
    )
    print(
        "Records with lower reported quantity:",
        len(result.get("lower_quantity_records", [])),
    )
    print(
        "Records with unknown/unconfirmed quantity or status:",
        len(result.get("unknown_records", [])),
    )

    matches = result.get("matches")

    if matches is not None and not matches.empty:
        print("\nRecords explicitly marked available with sufficient listed quantity:")
        visible_columns = [
            column
            for column in [
                "facility_name",
                "component",
                "blood_group",
                "units_listed",
                "reported_status",
                "last_updated",
            ]
            if column in matches.columns
        ]

        print(matches[visible_columns].to_string(index=False))

    print(
        "\nReminder: These are saved records, not live inventory. "
        "Confirm current stock directly with the facility."
    )


if __name__ == "__main__":
    examples = [
        "I need 2 units of B positive whole blood in Vizag",
        "Need three bags of O negative plasma",
    ]

    for example in examples:
        print_request_result(example)