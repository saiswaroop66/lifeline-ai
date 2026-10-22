
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from database import (
    initialize_database,
    get_inventory_records,
    get_blood_banks,
)

try:
    from request_agent import parse_blood_request
except Exception:
    parse_blood_request = None


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LifeLine | Blood Bank Search",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VERIFICATION_LOG = DATA_DIR / "verification_log.csv"

DEFAULT_AREA = "Visakhapatnam"

BLOOD_GROUPS = [
    "All", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"
]

COMPONENTS = [
    "All",
    "whole blood",
    "red blood cells",
    "platelets",
    "plasma",
    "cryoprecipitate",
]

STATUS_OPTIONS = [
    "All",
    "Available",
    "Unavailable",
    "Unknown",
    "Quantity recorded",
    "Zero recorded",
]


# ============================================================
# 2. NORMALIZATION HELPERS
# ============================================================

def clean(value):
    """Safely convert a value into trimmed text."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def norm_col(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


def find_column(df, candidates):
    lookup = {norm_col(col): col for col in df.columns}

    for candidate in candidates:
        key = norm_col(candidate)
        if key in lookup:
            return lookup[key]

    return None


def normalize_group(value):
    """Normalize blood-group labels such as 'O positive' and 'O+'."""
    text = clean(value).upper()
    text = re.sub(r"\s+", "", text)

    text = text.replace("POSITIVE", "+")
    text = text.replace("NEGATIVE", "-")

    aliases = {
        "APOS": "A+",
        "ANEG": "A-",
        "BPOS": "B+",
        "BNEG": "B-",
        "ABPOS": "AB+",
        "ABNEG": "AB-",
        "OPOS": "O+",
        "ONEG": "O-",
    }

    return aliases.get(text, text)


# Compatibility alias for code that calls normalize_blood_group().
def normalize_blood_group(value):
    return normalize_group(value)


def normalize_component(value):
    """Map common component abbreviations to consistent labels."""
    text = clean(value).lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if "cryoprecipitate" in text or text == "cryo":
        return "cryoprecipitate"

    if "whole blood" in text or text in {"wholeblood", "wb"}:
        return "whole blood"

    if (
        "platelet" in text
        or text in {"rdp", "sdp"}
        or "single donor platelet" in text
        or "random donor platelet" in text
    ):
        return "platelets"

    if "plasma" in text or text in {"ffp", "fresh frozen plasma"}:
        return "plasma"

    if any(
        term in text
        for term in (
            "red blood cell",
            "packed red cell",
            "prbc",
            "rbc",
        )
    ):
        return "red blood cells"

    return text


def normalize_status(value, units):
    raw = clean(value).lower()

    if raw in {
        "available",
        "in stock",
        "in-stock",
        "available now",
    }:
        return "Available"

    if raw in {
        "unavailable",
        "out of stock",
        "out-of-stock",
    }:
        return "Unavailable"

    if raw in {
        "unknown",
        "not known",
        "not reported",
        "n/a",
        "na",
        "not available",
        "",
    }:
        if pd.notna(units):
            if units > 0:
                return "Quantity recorded"
            if units == 0:
                return "Zero recorded"

        return "Unknown"

    if pd.notna(units):
        if units > 0:
            return "Quantity recorded"
        if units == 0:
            return "Zero recorded"

    return "Unknown"


def text_matches(series, query):
    query = clean(query)

    if not query:
        return pd.Series(True, index=series.index)

    return series.fillna("").astype(str).str.contains(
        query,
        case=False,
        regex=False,
        na=False,
    )


def numeric_units(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        str(value).replace(",", ""),
    )

    if not match:
        return None

    try:
        number = float(match.group())
        return number if number >= 0 else None
    except ValueError:
        return None


# ============================================================
# 3. NATURAL-LANGUAGE REQUEST PARSER
# ============================================================

def parse_fallback(query):
    """Local parser used when the AI parser is unavailable or incomplete."""
    text = clean(query).lower()

    group_patterns = [
        (r"\bab\s*(?:positive|\+)\b", "AB+"),
        (r"\bab\s*(?:negative|-)\b", "AB-"),
        (r"\ba\s*(?:positive|\+)\b", "A+"),
        (r"\ba\s*(?:negative|-)\b", "A-"),
        (r"\bb\s*(?:positive|\+)\b", "B+"),
        (r"\bb\s*(?:negative|-)\b", "B-"),
        (r"\bo\s*(?:positive|\+)\b", "O+"),
        (r"\bo\s*(?:negative|-)\b", "O-"),
        (r"\bab\+\b", "AB+"),
        (r"\bab-\b", "AB-"),
        (r"\ba\+\b", "A+"),
        (r"\ba-\b", "A-"),
        (r"\bb\+\b", "B+"),
        (r"\bb-\b", "B-"),
        (r"\bo\+\b", "O+"),
        (r"\bo-\b", "O-"),
    ]

    group = ""
    for pattern, candidate in group_patterns:
        if re.search(pattern, text):
            group = candidate
            break

    component = ""
    component_patterns = [
        (r"\bcryoprecipitate\b|\bcryo\b", "cryoprecipitate"),
        (r"\bwhole\s*blood\b|\bwb\b", "whole blood"),
        (
            r"\bred\s*blood\s*cells?\b|\bprbc\b|\brbc\b",
            "red blood cells",
        ),
        (
            r"\bplatelets?\b|\bsdp\b|\brdp\b",
            "platelets",
        ),
        (
            r"\bfresh\s*frozen\s*plasma\b|\bplasma\b|\bffp\b",
            "plasma",
        ),
    ]

    for pattern, candidate in component_patterns:
        if re.search(pattern, text):
            component = candidate
            break

    number_words = {
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

    units = None

    quantity_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*"
        r"(?:units?|bags?|bottles?)\b",
        text,
    )

    if quantity_match:
        try:
            units = float(quantity_match.group(1))
        except ValueError:
            units = None
    else:
        for word, number in number_words.items():
            if re.search(
                r"\b" + word + r"\s*(?:units?|bags?|bottles?)\b",
                text,
            ):
                units = float(number)
                break

    location = ""

    if "visakhapatnam" in text or "vizag" in text:
        location = DEFAULT_AREA
    elif "hyderabad" in text:
        location = "Hyderabad"
    elif "vijayawada" in text:
        location = "Vijayawada"

    missing = []

    if not group:
        missing.append("blood_group")

    if not component:
        missing.append("component")

    if units is None:
        missing.append("units")

    return {
        "original_request": query,
        "blood_group": group or None,
        "component": component or None,
        "units": units,
        "location": location or None,
        "missing": missing,
        "valid": not missing,
        "source": "fallback",
    }


def parse_request(query):
    """Use the configured request agent, with a safe local fallback."""
    fallback = parse_fallback(query)

    if parse_blood_request is not None:
        try:
            parsed = parse_blood_request(query)

            if isinstance(parsed, dict):
                parsed = parsed.copy()

                aliases = {
                    "blood_group": ("blood_group", "group", "blood_type"),
                    "component": ("component", "blood_component"),
                    "units": ("units", "requested_units", "quantity"),
                    "location": ("location", "city"),
                }

                for target, keys in aliases.items():
                    value = None

                    for key in keys:
                        if parsed.get(key) not in (None, ""):
                            value = parsed[key]
                            break

                    if value in (None, ""):
                        value = fallback.get(target)

                    parsed[target] = value

                parsed["blood_group"] = normalize_group(
                    parsed.get("blood_group")
                )
                parsed["component"] = normalize_component(
                    parsed.get("component")
                )

                try:
                    if parsed.get("units") not in (None, ""):
                        parsed["units"] = float(parsed["units"])
                    else:
                        parsed["units"] = None
                except (ValueError, TypeError):
                    parsed["units"] = fallback.get("units")

                parsed["location"] = clean(parsed.get("location"))

                parsed["missing"] = [
                    key
                    for key in ("blood_group", "component", "units")
                    if parsed.get(key) in (None, "")
                ]
                parsed["valid"] = not parsed["missing"]

                return parsed

        except Exception:
            pass

    return fallback


# ============================================================
# 4. DATABASE LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    initialize_database()

    inventory = get_inventory_records()
    banks = get_blood_banks()

    output_columns = [
        "Facility ID",
        "Blood Bank",
        "Blood Group",
        "Component",
        "Listed Units",
        "Source Status",
        "Recorded Status",
        "Location",
        "Address",
        "Phone",
        "Email",
        "Snapshot Date",
        "Search Text",
        "Source File",
        "Imported At",
    ]

    if inventory is None or inventory.empty:
        return pd.DataFrame(columns=output_columns), banks

    aliases = {
        "Facility ID": ["Facility ID", "facility_id", "id"],
        "Blood Bank": [
            "Blood Bank", "facility_name", "facility",
            "blood_bank", "name",
        ],
        "Blood Group": [
            "Blood Group", "blood_group", "blood_type",
        ],
        "Component": [
            "Component", "component", "blood_component",
        ],
        "Listed Units": [
            "Listed Units", "units_listed", "units",
            "quantity", "available_units",
        ],
        "Source Status": [
            "Source Status", "reported_status", "status",
        ],
        "Location": ["Location", "location", "area", "city"],
        "Address": ["Address", "address"],
        "Phone": ["Phone", "phone", "contact_number"],
        "Email": ["Email", "email"],
        "Snapshot Date": [
            "Snapshot Date", "last_updated", "last_update",
        ],
        "Source File": ["Source File", "source_file"],
        "Imported At": ["Imported At", "imported_at"],
    }

    prepared = pd.DataFrame(index=inventory.index)

    for target, candidates in aliases.items():
        column = find_column(inventory, candidates)
        prepared[target] = inventory[column] if column else ""

    prepared["Blood Group"] = prepared["Blood Group"].apply(
        normalize_group
    )
    prepared["Component"] = prepared["Component"].apply(
        normalize_component
    )

    prepared["Listed Units"] = pd.to_numeric(
        prepared["Listed Units"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )

    prepared["Recorded Status"] = [
        normalize_status(status, units)
        for status, units in zip(
            prepared["Source Status"],
            prepared["Listed Units"],
        )
    ]

    text_columns = [
        "Facility ID",
        "Blood Bank",
        "Location",
        "Address",
        "Phone",
        "Email",
        "Snapshot Date",
        "Source File",
        "Imported At",
        "Source Status",
    ]

    for column in text_columns:
        prepared[column] = (
            prepared[column].fillna("").astype(str).str.strip()
        )

    prepared["Location"] = prepared["Location"].where(
        prepared["Location"].ne(""),
        prepared["Address"],
    )

    prepared["Search Text"] = (
        prepared["Blood Bank"]
        + " "
        + prepared["Facility ID"]
        + " "
        + prepared["Location"]
        + " "
        + prepared["Address"]
    ).str.lower()

    for column in output_columns:
        if column not in prepared.columns:
            prepared[column] = ""

    return prepared[output_columns].reset_index(drop=True), banks


# ============================================================
# 5. AGENT 1 — EXACT-UNIT MATCHING
# ============================================================

def latest_record_per_facility(records):
    """Use one latest record per facility to avoid double-counting."""
    if records.empty:
        return records.copy()

    work = records.copy()

    work["_snapshot_sort"] = work["Snapshot Date"].fillna("").astype(str)
    work["_import_sort"] = work["Imported At"].fillna("").astype(str)

    work["_facility_key"] = (
        work["Facility ID"].fillna("").astype(str).str.strip()
    )

    work["_facility_key"] = work["_facility_key"].where(
        work["_facility_key"].ne(""),
        work["Blood Bank"].fillna("").astype(str).str.strip(),
    )

    work = work.sort_values(
        ["_snapshot_sort", "_import_sort"],
        kind="stable",
    )

    work = work.drop_duplicates("_facility_key", keep="last")

    return work.drop(
        columns=["_snapshot_sort", "_import_sort", "_facility_key"]
    )


def calculate_exact_match(
    inventory_df,
    blood_group,
    component,
    requested_units,
):
    group = normalize_group(blood_group)
    comp = normalize_component(component)
    requested = float(requested_units)

    matches = inventory_df[
        (inventory_df["Blood Group"] == group)
        & (inventory_df["Component"] == comp)
    ].copy()

    matches = latest_record_per_facility(matches)

    matches["_numeric_units"] = matches["Listed Units"].apply(numeric_units)

    # Only explicit "Available" records are eligible for proposed allocation.
    eligible = matches[
        (matches["Recorded Status"] == "Available")
        & matches["_numeric_units"].notna()
        & (matches["_numeric_units"] > 0)
    ].copy()

    eligible = eligible.sort_values(
        "_numeric_units",
        ascending=False,
    )

    allocations = []
    remaining = requested

    for _, row in eligible.iterrows():
        if remaining <= 0:
            break

        listed = float(row["_numeric_units"])
        proposed = min(listed, remaining)

        if proposed <= 0:
            continue

        remaining = max(remaining - proposed, 0)

        allocations.append({
            "Facility ID": clean(row["Facility ID"]),
            "Blood Bank": clean(row["Blood Bank"]) or "Unnamed facility",
            "Listed Units": listed,
            "Proposed Units": proposed,
            "Remaining Shortage": remaining,
            "Phone": clean(row["Phone"]),
            "Address": clean(row["Address"]),
            "Snapshot Date": clean(row["Snapshot Date"]),
            "Source Status": clean(row["Source Status"]),
        })

    proposed_total = requested - remaining

    unknown_count = (
        int(matches["_numeric_units"].isna().sum())
        if not matches.empty
        else 0
    )

    return {
        "requested": requested,
        "proposed": proposed_total,
        "shortage": remaining,
        "allocation": pd.DataFrame(allocations),
        "matching_records": matches.drop(
            columns=["_numeric_units"],
            errors="ignore",
        ),
        "unknown_count": unknown_count,
        "eligible_facilities": len(eligible),
    }


# ============================================================
# 6. AGENT 2 — MULTI-BANK FULFILLMENT
# ============================================================

def plan_fulfillment(
    inventory_df,
    blood_group,
    component,
    requested_units,
):
    result = calculate_exact_match(
        inventory_df,
        blood_group,
        component,
        requested_units,
    )

    if result["proposed"] <= 0:
        status = "no_candidates"
        message = (
            "No matching saved records explicitly marked Available "
            "with a usable quantity were found."
        )
    elif result["shortage"] > 0:
        status = "partial"
        message = (
            "The saved records support only a partial proposed allocation."
        )
    else:
        status = "proposed"
        message = (
            "The saved records suggest a proposed allocation across "
            "the listed facilities."
        )

    return {
        **result,
        "status": status,
        "message": message,
        "requires_human_approval": True,
        "is_live": False,
    }


# ============================================================
# 7. AGENT 3 — HUMAN VERIFICATION LOG
# ============================================================

VERIFICATION_COLUMNS = [
    "timestamp",
    "facility_id",
    "facility_name",
    "phone",
    "blood_group",
    "component",
    "requested_units",
    "reported_units",
    "outcome",
    "staff_name_or_reference",
    "notes",
    "recorded_by",
]


def load_verification_log():
    if not VERIFICATION_LOG.exists():
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)

    try:
        df = pd.read_csv(VERIFICATION_LOG)

        for column in VERIFICATION_COLUMNS:
            if column not in df.columns:
                df[column] = ""

        return df[VERIFICATION_COLUMNS]

    except Exception:
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)


def save_verification(record):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log = load_verification_log()
    log = pd.concat(
        [log, pd.DataFrame([record])],
        ignore_index=True,
    )

    log.to_csv(VERIFICATION_LOG, index=False)


# ============================================================
# 8. APP HEADER AND DATABASE
# ============================================================

st.title("🩸 LifeLine")
st.subheader("AI-powered Blood Bank Search & Fulfillment")

st.caption(
    "Search saved blood-bank records, plan exact-unit allocations, "
    "and record human verification outcomes."
)

st.warning(
    "Snapshot data only — not live or guaranteed stock. Confirm availability "
    "directly with the blood centre before travelling. Unknown quantity is not "
    "zero. Allocation is a planning aid only; clinical compatibility and blood "
    "release must be decided by authorized clinical and blood-bank staff."
)

try:
    inventory_df, banks_df = load_data()
except Exception as exc:
    st.error("LifeLine could not read its existing SQLite database.")
    st.code(str(exc))
    st.info("Check database.py and ensure data/lifeline.db is populated.")
    st.stop()

if inventory_df.empty:
    st.warning("The SQLite inventory table is empty.")
    st.info("Run your existing import_csv.py script, then refresh the app.")
    st.stop()


# ============================================================
# 9. SEARCH FORMS
# ============================================================

with st.form("lifeline_natural_search"):
    natural_query = st.text_input(
        "Describe your blood request",
        placeholder=(
            "Example: I need 2 units of AB+ cryoprecipitate in Visakhapatnam"
        ),
    )

    natural_clicked = st.form_submit_button(
        "🔎 Search using description",
        type="primary",
        use_container_width=True,
    )


with st.sidebar:
    st.title("🩸 LifeLine")
    st.caption("VISAKHAPATNAM BLOOD NETWORK")
    st.divider()
    st.header("Search filters")

    with st.form("lifeline_filter_form"):
        keyword = st.text_input("Centre name / facility ID")

        location = st.text_input(
            "City / area",
            value=DEFAULT_AREA,
        )

        group_filter = st.selectbox("Blood group", BLOOD_GROUPS)
        component_filter = st.selectbox("Blood component", COMPONENTS)
        status_filter = st.selectbox("Recorded status", STATUS_OPTIONS)

        minimum_units = st.number_input(
            "Minimum listed units",
            min_value=0,
            max_value=100000,
            value=0,
            step=1,
        )

        include_unknown = st.checkbox(
            "Include unknown quantities",
            value=True,
        )

        filter_clicked = st.form_submit_button(
            "Apply filters",
            type="primary",
            use_container_width=True,
        )


# ============================================================
# 10. SESSION STATE
# ============================================================

if "lifeline_query" not in st.session_state:
    st.session_state.lifeline_query = ""

if "lifeline_parsed" not in st.session_state:
    st.session_state.lifeline_parsed = {}

if "lifeline_mode" not in st.session_state:
    st.session_state.lifeline_mode = ""

if "lifeline_filters" not in st.session_state:
    st.session_state.lifeline_filters = {}


if natural_clicked:
    if not natural_query.strip():
        st.warning("Enter a blood requirement first.")
    else:
        st.session_state.lifeline_query = natural_query.strip()
        st.session_state.lifeline_parsed = parse_request(
            natural_query.strip()
        )
        st.session_state.lifeline_mode = "natural"


if filter_clicked:
    st.session_state.lifeline_mode = "filters"
    st.session_state.lifeline_filters = {
        "keyword": keyword.strip(),
        "location": location.strip(),
        "group": group_filter,
        "component": component_filter,
        "status": status_filter,
        "minimum_units": int(minimum_units),
        "include_unknown": include_unknown,
    }


if st.sidebar.button("Reset search", use_container_width=True):
    st.session_state.lifeline_mode = ""
    st.session_state.lifeline_query = ""
    st.session_state.lifeline_parsed = {}
    st.session_state.lifeline_filters = {}
    st.rerun()


# ============================================================
# 11. APPLY REQUEST OR FILTERS
# ============================================================

mode = st.session_state.lifeline_mode

# Do not show all hospitals on startup.
results = (
    inventory_df.copy()
    if mode in {"natural", "filters"}
    else inventory_df.iloc[0:0].copy()
)

parsed = {}
requested_group = ""
requested_component = ""
requested_units = None
requested_location = ""

if mode == "natural":
    parsed = st.session_state.lifeline_parsed

    requested_group = normalize_group(parsed.get("blood_group") or "")
    requested_component = normalize_component(parsed.get("component") or "")
    requested_units = parsed.get("units")
    requested_location = clean(parsed.get("location"))

    st.header("Your request")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blood group", requested_group or "Not specified")
    c2.metric("Component", requested_component or "Not specified")
    c3.metric(
        "Units requested",
        requested_units if requested_units is not None else "Not specified",
    )
    c4.metric("Location", requested_location or "Not specified")

    if parsed.get("source") == "groq":
        st.success("Request parsed using Groq AI.")
    else:
        st.info("Request parsed using the available local parser.")

    if not requested_group or not requested_component:
        st.warning(
            "Include both a blood group and component, for example "
            "'2 units of AB+ cryoprecipitate'."
        )
        results = results.iloc[0:0].copy()

    elif requested_units is None:
        st.warning(
            "Include an exact quantity, for example '2 units'. "
            "Facilities will appear after you specify the units required."
        )
        results = results.iloc[0:0].copy()

    else:
        try:
            requested_units = float(requested_units)
            if requested_units <= 0:
                raise ValueError
        except (TypeError, ValueError):
            st.error("Requested units must be a positive number.")
            requested_units = None
            results = results.iloc[0:0].copy()

        if requested_units is not None:
            results = results[
                (results["Blood Group"] == requested_group)
                & (results["Component"] == requested_component)
            ].copy()

            # Apply location before planning. Never silently ignore it.
            if requested_location:
                location_mask = (
                    results["Location"].astype(str).str.contains(
                        requested_location,
                        case=False,
                        regex=False,
                        na=False,
                    )
                    | results["Address"].astype(str).str.contains(
                        requested_location,
                        case=False,
                        regex=False,
                        na=False,
                    )
                )
                results = results[location_mask].copy()

elif mode == "filters":
    filters = st.session_state.lifeline_filters

    requested_group = filters.get("group", "All")
    requested_component = filters.get("component", "All")
    requested_location = filters.get("location", "")

    if requested_group != "All":
        results = results[
            results["Blood Group"] == normalize_group(requested_group)
        ]

    if requested_component != "All":
        results = results[
            results["Component"] == normalize_component(requested_component)
        ]

    if requested_location:
        location_mask = (
            results["Location"].astype(str).str.contains(
                requested_location,
                case=False,
                regex=False,
                na=False,
            )
            | results["Address"].astype(str).str.contains(
                requested_location,
                case=False,
                regex=False,
                na=False,
            )
        )
        results = results[location_mask].copy()

    if filters.get("keyword"):
        results = results[
            text_matches(results["Search Text"], filters["keyword"])
        ]

    if filters.get("status", "All") != "All":
        results = results[
            results["Recorded Status"] == filters["status"]
        ]

    minimum = filters.get("minimum_units", 0)

    if filters.get("include_unknown", True):
        results = results[
            results["Listed Units"].isna()
            | (results["Listed Units"] >= minimum)
        ]
    else:
        results = results[
            results["Listed Units"].notna()
            & (results["Listed Units"] >= minimum)
        ]

else:
    st.info(
        "Enter a blood requirement above or apply the sidebar filters. "
        "Hospital records remain hidden until you search."
    )


# ============================================================
# 12. AGENTS 1 + 2 — EXACT MATCH AND ALLOCATION
# ============================================================

plan = None

if mode == "natural" and requested_units is not None:
    if requested_group and requested_component:
        st.divider()
        st.header("🤖 Exact match & multi-bank fulfillment")

        # results already contains only the requested group, component,
        # and location, so the plan cannot silently use unrelated facilities.
        plan = plan_fulfillment(
            results,
            requested_group,
            requested_component,
            requested_units,
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Units requested", f"{plan['requested']:g}")
        k2.metric("Proposed units", f"{plan['proposed']:g}")
        k3.metric("Remaining shortage", f"{plan['shortage']:g}")

        st.write(plan["message"])

        if plan["status"] == "proposed":
            st.success(
                "Saved records suggest enough listed quantity for a proposed allocation."
            )
        elif plan["status"] == "partial":
            st.warning(
                "The proposed allocation is partial. The remaining shortage "
                "is shown above."
            )
        else:
            st.warning(
                "No eligible saved records were found for this proposed allocation."
            )

        allocation = plan["allocation"]

        if not allocation.empty:
            st.subheader("Proposed allocation by facility")

            display_columns = [
                "Blood Bank",
                "Listed Units",
                "Proposed Units",
                "Remaining Shortage",
            ]

            allocation_display = allocation[display_columns].copy()

            total_row = pd.DataFrame([{
                "Blood Bank": "TOTAL",
                "Listed Units": allocation["Listed Units"].sum(),
                "Proposed Units": allocation["Proposed Units"].sum(),
                "Remaining Shortage": plan["shortage"],
            }])

            allocation_display = pd.concat(
                [allocation_display, total_row],
                ignore_index=True,
            )

            st.dataframe(
                allocation_display,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "This is a proposed allocation from saved records, not a "
                "reservation. Confirm current availability directly with "
                "each facility."
            )

            # Only show facilities contributing to the proposed allocation.
            allocation_ids = {
                clean(value)
                for value in allocation["Facility ID"]
                if clean(value)
            }

            allocation_names = {
                clean(value)
                for value in allocation["Blood Bank"]
                if clean(value)
            }

            results = results[
                results["Facility ID"].map(clean).isin(allocation_ids)
                | results["Blood Bank"].map(clean).isin(allocation_names)
            ].copy()

            results = results.drop_duplicates(
                subset=["Facility ID", "Blood Bank"],
                keep="first",
            )

        else:
            results = results.iloc[0:0].copy()
            st.info(
                "No facility can contribute units from eligible saved records."
            )

        if plan["unknown_count"]:
            st.caption(
                f"{plan['unknown_count']} matching record(s) have unknown "
                "quantities. They are not counted as available units."
            )


# ============================================================
# 13. MATCHING BLOOD BANKS
# ============================================================

if mode in {"natural", "filters"}:
    st.divider()
    st.header("🏥 Matching Blood Banks")

    facility_count = (
        results["Facility ID"].replace("", pd.NA).nunique()
        if not results.empty
        else 0
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Matching records", len(results))
    m2.metric("Facilities represented", int(facility_count))
    m3.metric(
        "Known quantities",
        int(results["Listed Units"].notna().sum())
        if not results.empty
        else 0,
    )

    if results.empty:
        st.warning(
            "No facilities matched your request or contributed to the plan. "
            "Try a different location or request."
        )

    else:
        for index, (_, row) in enumerate(
            results.reset_index(drop=True).iterrows(),
            start=1,
        ):
            bank_name = clean(row["Blood Bank"]) or "Unnamed facility"
            facility_id = clean(row["Facility ID"]) or "Not provided"
            units = row["Listed Units"]

            if pd.isna(units):
                units_label = "Unknown"
            elif float(units).is_integer():
                units_label = str(int(units))
            else:
                units_label = str(units)

            with st.container(border=True):
                left, right = st.columns([5, 1])

                left.subheader(f"🏥 {bank_name}")
                left.caption(f"Facility ID: {facility_id}")
                right.caption(f"Record #{index}")

                c1, c2, c3 = st.columns(3)

                c1.write("**Blood group**")
                c1.write(clean(row["Blood Group"]) or "Unknown")

                c2.write("**Component**")
                c2.write(clean(row["Component"]) or "Unknown")

                c3.write("**Listed units**")
                c3.write(units_label)

                c4, c5, c6 = st.columns(3)

                c4.write("**Recorded status**")
                c4.write(clean(row["Recorded Status"]) or "Unknown")

                c5.write("**Location**")
                c5.write(
                    clean(row["Location"])
                    or clean(row["Address"])
                    or "Not provided"
                )

                c6.write("**Snapshot date**")
                c6.write(clean(row["Snapshot Date"]) or "Not provided")

                with st.expander("Contact details and source"):
                    st.write(
                        f"**Phone:** {clean(row['Phone']) or 'Not provided'}"
                    )
                    st.write(
                        f"**Email:** {clean(row['Email']) or 'Not provided'}"
                    )
                    st.write(
                        f"**Address:** {clean(row['Address']) or 'Not provided'}"
                    )
                    st.write(
                        f"**Source file:** "
                        f"{clean(row['Source File']) or 'Not provided'}"
                    )
                    st.write(
                        f"**Imported at:** "
                        f"{clean(row['Imported At']) or 'Not provided'}"
                    )


# ============================================================
# 14. AGENT 3 — HUMAN AVAILABILITY VERIFICATION
# ============================================================

st.divider()
st.header("📝 Human availability verification")

st.caption(
    "Use this form after a person contacts a blood bank. It records human "
    "reports only; LifeLine does not automatically call facilities."
)

candidate_df = inventory_df.copy()

if mode == "natural" and requested_group and requested_component:
    candidate_df = candidate_df[
        (candidate_df["Blood Group"] == requested_group)
        & (candidate_df["Component"] == requested_component)
    ].copy()

candidate_df = candidate_df.drop_duplicates(
    subset=["Facility ID", "Blood Bank"],
    keep="first",
)

facility_options = []
facility_lookup = {}

for _, row in candidate_df.iterrows():
    label = (
        f"{clean(row['Blood Bank']) or 'Unnamed facility'}"
        f" | {clean(row['Facility ID']) or 'No ID'}"
    )

    if label not in facility_lookup:
        facility_lookup[label] = row
        facility_options.append(label)


with st.form("verification_record_form"):
    selected_facility = st.selectbox(
        "Facility contacted",
        facility_options if facility_options else ["No facility records"],
    )

    available_groups = [g for g in BLOOD_GROUPS if g != "All"]
    group_index = (
        available_groups.index(requested_group)
        if requested_group in available_groups
        else 0
    )

    verify_group = st.selectbox(
        "Blood group checked",
        available_groups,
        index=group_index,
    )

    available_components = [c for c in COMPONENTS if c != "All"]
    component_index = (
        available_components.index(requested_component)
        if requested_component in available_components
        else 0
    )

    verify_component = st.selectbox(
        "Component checked",
        available_components,
        index=component_index,
    )

    default_units = 1

    if requested_units is not None:
        try:
            if int(requested_units) >= 1:
                default_units = int(requested_units)
        except (ValueError, TypeError, OverflowError):
            pass

    verify_requested = st.number_input(
        "Units requested",
        min_value=1,
        max_value=10000,
        value=default_units,
        step=1,
    )

    outcome_options = {
        "confirmed_by_facility": "Facility staff reported available",
        "partially_available": "Partially available",
        "unavailable": "Reported unavailable",
        "no_answer": "No answer",
        "callback_requested": "Callback requested",
        "details_unclear": "Details unclear",
    }

    verify_outcome = st.selectbox(
        "Reported outcome",
        list(outcome_options.keys()),
        format_func=lambda value: outcome_options[value],
    )

    reported_units = st.number_input(
        "Units reported by facility",
        min_value=0,
        max_value=10000,
        value=0,
        step=1,
    )

    staff_reference = st.text_input("Staff name / reference (optional)")
    notes = st.text_area("Notes (optional)")

    submitted_verification = st.form_submit_button(
        "Save verification record",
        type="primary",
        use_container_width=True,
    )


if submitted_verification:
    if (
        not facility_options
        or selected_facility not in facility_lookup
    ):
        st.error("No facility is available to attach this record.")

    else:
        facility = facility_lookup[selected_facility]

        record = {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "facility_id": clean(facility["Facility ID"]),
            "facility_name": clean(facility["Blood Bank"]),
            "phone": clean(facility["Phone"]),
            "blood_group": verify_group,
            "component": verify_component,
            "requested_units": int(verify_requested),
            "reported_units": int(reported_units),
            "outcome": verify_outcome,
            "staff_name_or_reference": staff_reference.strip(),
            "notes": notes.strip(),
            "recorded_by": "human_reported",
        }

        try:
            save_verification(record)

            st.success(
                "Verification record saved to data/verification_log.csv."
            )

            st.caption(
                "This human-reported record does not automatically update "
                "the inventory database."
            )

        except Exception as exc:
            st.error("Could not save verification record.")
            st.code(str(exc))


with st.expander("Recent verification history"):
    history = load_verification_log()

    if history.empty:
        st.write("No verification records have been saved yet.")

    else:
        st.dataframe(
            history.tail(100).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download verification history",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="lifeline_verification_log.csv",
            mime="text/csv",
        )


# ============================================================
# 15. EXPORT AND FOOTER
# ============================================================

st.divider()

st.download_button(
    "⬇️ Download current search results",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="lifeline_search_results.csv",
    mime="text/csv",
    use_container_width=True,
)

with st.expander("About LifeLine data and safety"):
    st.write(
        "LifeLine searches saved inventory records in its SQLite database. "
        "They are not a live stock feed. Proposed allocations are not "
        "reservations. Contact the blood centre to confirm current availability. "
        "The app does not determine clinical compatibility or authorize blood "
        "release. Verification entries are human-reported and do not "
        "automatically change inventory."
    )

st.caption("LifeLine • Blood-bank discovery for Visakhapatnam")
