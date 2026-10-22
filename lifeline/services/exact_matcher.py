
"""LifeLine exact-unit matching service."""

import math
import pandas as pd


def _number(value):
    try:
        if pd.isna(value):
            return None
        number = float(value)
        if math.isfinite(number) and number >= 0:
            return number
    except (TypeError, ValueError):
        pass
    return None


def calculate_exact_match(
    inventory_df,
    blood_group,
    component,
    requested_units,
):
    """Calculate a proposed allocation from saved inventory records."""

    try:
        requested = float(requested_units)
    except (TypeError, ValueError):
        raise ValueError("Requested units must be a positive whole number.")

    if (
        not math.isfinite(requested)
        or requested <= 0
        or not requested.is_integer()
    ):
        raise ValueError("Requested units must be a positive whole number.")

    requested = int(requested)

    required = {
        "Facility ID", "Blood Bank", "Blood Group",
        "Component", "Listed Units", "Recorded Status",
    }
    missing = required - set(inventory_df.columns)

    if missing:
        raise ValueError(
            "Missing inventory columns: " + ", ".join(sorted(missing))
        )

    df = inventory_df.copy()

    df = df[
        df["Blood Group"].astype(str).str.strip().str.upper().eq(
            str(blood_group).strip().upper()
        )
        & df["Component"].astype(str).str.strip().str.casefold().eq(
            str(component).strip().casefold()
        )
    ].copy()

    if df.empty:
        return _summary(requested, [], 0, 0)

    df["Facility ID"] = df["Facility ID"].fillna("").astype(str).str.strip()
    df = df[df["Facility ID"].ne("")]

    # Prefer the most recently recorded row for each facility.
    sort_cols = [
        col for col in ("Snapshot Date", "Imported At")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="first")

    df = df.drop_duplicates("Facility ID", keep="last")

    candidates = []
    unknown_count = 0

    for _, row in df.iterrows():
        status = str(row["Recorded Status"]).strip().casefold()
        quantity = _number(row["Listed Units"])

        if status != "available" or quantity is None:
            unknown_count += 1
            continue

        # Units are discrete; do not silently round fractional quantities.
        if not quantity.is_integer():
            unknown_count += 1
            continue

        if quantity <= 0:
            continue

        candidates.append({
            "facility_id": row["Facility ID"],
            "facility": str(row["Blood Bank"]).strip() or "Unnamed facility",
            "listed_units": int(quantity),
            "snapshot_date": str(row.get("Snapshot Date", "") or ""),
            "phone": str(row.get("Phone", "") or ""),
            "address": str(row.get("Address", "") or ""),
        })

    # Larger quantities first, to reduce the number of proposed facilities.
    candidates.sort(key=lambda item: item["listed_units"], reverse=True)

    allocation = []
    remaining = requested

    for candidate in candidates:
        if remaining == 0:
            break

        proposed = min(candidate["listed_units"], remaining)

        allocation.append({
            **candidate,
            "proposed_units": proposed,
        })
        remaining -= proposed

    return _summary(
        requested,
        allocation,
        requested - remaining,
        unknown_count,
    )


def _summary(requested, allocation, proposed_total, unknown_count):
    return {
        "requested_units": requested,
        "proposed_units": proposed_total,
        "remaining_shortage": requested - proposed_total,
        "fully_covered_by_listed_records": proposed_total >= requested,
        "facilities_in_proposal": len(allocation),
        "unknown_or_unconfirmed_records": unknown_count,
        "allocation": allocation,
        "warning": (
            "Quantities come from saved records, not live stock. "
            "Proposed allocations require facility confirmation."
        ),
    }
