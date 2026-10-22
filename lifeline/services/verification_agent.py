
"""LifeLine human-assisted blood-bank verification service."""

from datetime import datetime, timezone


VALID_OUTCOMES = {
    "confirmed_by_facility",
    "partially_available",
    "unavailable",
    "no_answer",
    "callback_requested",
    "details_unclear",
}


def prepare_call_script(
    facility_name,
    blood_group,
    component,
    requested_units,
):
    """Prepare a script for an authorized coordinator."""

    try:
        units = int(requested_units)
    except (TypeError, ValueError):
        raise ValueError("Requested units must be a positive whole number.")

    if units <= 0 or units != requested_units:
        raise ValueError("Requested units must be a positive whole number.")

    return {
        "facility": str(facility_name).strip() or "Blood bank",
        "script": (
            f"Hello, I am calling to verify a blood requirement. "
            f"Could your facility please confirm whether {units} units "
            f"of {blood_group} {component} are currently available? "
            f"Please confirm the component, quantity, and the time "
            f"this information is valid for. We will follow your "
            f"authorized procedures for any next steps."
        ),
        "status": "pending",
        "requires_human_call": True,
    }


def record_verification(
    facility_id,
    facility_name,
    blood_group,
    component,
    requested_units,
    outcome,
    reported_units=None,
    verified_by="",
    notes="",
):
    """Validate and return a verification record for later database storage."""

    outcome = str(outcome).strip().casefold()

    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            "Outcome must be one of: " + ", ".join(sorted(VALID_OUTCOMES))
        )

    try:
        requested = int(requested_units)
    except (TypeError, ValueError):
        raise ValueError("Requested units must be a positive whole number.")

    if requested <= 0 or requested != requested_units:
        raise ValueError("Requested units must be a positive whole number.")

    reported = None

    if reported_units is not None:
        try:
            reported = int(reported_units)
        except (TypeError, ValueError):
            raise ValueError("Reported units must be a non-negative whole number.")

        if reported < 0 or reported != reported_units:
            raise ValueError("Reported units must be a non-negative whole number.")

    if outcome in {"confirmed_by_facility", "partially_available"}:
        if reported is None:
            raise ValueError(
                "Enter the quantity reported by the facility for this outcome."
            )

    if outcome == "confirmed_by_facility" and reported < requested:
        raise ValueError(
            "Reported units are below the request. Use partially_available "
            "or another accurate outcome."
        )

    if outcome == "partially_available" and not (0 < reported < requested):
        raise ValueError(
            "Partially available requires a reported quantity between "
            "zero and the requested quantity."
        )

    return {
        "facility_id": str(facility_id).strip(),
        "facility_name": str(facility_name).strip(),
        "blood_group": str(blood_group).strip().upper(),
        "component": str(component).strip(),
        "requested_units": requested,
        "outcome": outcome,
        "reported_units": reported,
        "verified_by": str(verified_by).strip(),
        "notes": str(notes).strip(),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "human_reported",
        "database_saved": False,
        "warning": (
            "This is a human-reported verification record. It is not "
            "a reservation, clinical compatibility decision, or authorization "
            "to release blood."
        ),
    }
