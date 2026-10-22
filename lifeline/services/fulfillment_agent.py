
"""LifeLine multi-bank fulfillment planning service."""

from services.exact_matcher import calculate_exact_match


def plan_fulfillment(
    inventory_df,
    blood_group,
    component,
    requested_units,
):
    """Create a proposed multi-facility plan; do not book or reserve stock."""

    match = calculate_exact_match(
        inventory_df=inventory_df,
        blood_group=blood_group,
        component=component,
        requested_units=requested_units,
    )

    allocation = match["allocation"]

    if not allocation:
        status = "no_candidates"
        message = (
            "No eligible records explicitly marked Available "
            "with a usable quantity were found."
        )
    elif match["remaining_shortage"] > 0:
        status = "partial"
        message = (
            "The candidate records do not cover the entire request. "
            "The remaining quantity still needs to be sourced."
        )
    else:
        status = "proposed"
        message = (
            "The listed quantities could cover the request in this "
            "proposal. Confirm stock with every facility before relying on it."
        )

    return {
        "status": status,
        "message": message,
        "requested_units": match["requested_units"],
        "proposed_units": match["proposed_units"],
        "remaining_shortage": match["remaining_shortage"],
        "facilities_in_plan": match["facilities_in_proposal"],
        "allocation": allocation,
        "unknown_or_unconfirmed_records": (
            match["unknown_or_unconfirmed_records"]
        ),
        "requires_human_approval": True,
        "is_booking": False,
        "warning": match["warning"],
    }
