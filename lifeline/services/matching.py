
import pandas as pd


def find_matching_facilities(
    inventory: pd.DataFrame,
    blood_banks: pd.DataFrame,
    blood_group: str,
    required_units: int,
) -> pd.DataFrame:

    if required_units < 1:
        raise ValueError("Required units must be at least 1.")

    data = inventory.copy()

    data["blood_group"] = (
        data["blood_group"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    data["units_listed"] = pd.to_numeric(
        data["units_listed"],
        errors="coerce",
    )

    matches = data[
        (data["blood_group"] == blood_group.strip().upper())
        & (data["units_listed"] >= required_units)
    ].copy()

    if matches.empty:
        return matches

    # Join facility details using facility_id.
    details_columns = [
        column
        for column in [
            "facility_id",
            "facility_name",
            "address",
            "phone",
            "verification_status",
        ]
        if column in blood_banks.columns
    ]

    details = blood_banks[details_columns].drop_duplicates(
        subset=["facility_id"]
    )

    matches = matches.drop(
        columns=["facility_name", "address", "phone",
                 "verification_status"],
        errors="ignore",
    )

    matches = matches.merge(
        details,
        on="facility_id",
        how="left",
    )

    matches["last_updated"] = pd.to_datetime(
        matches["last_updated"],
        errors="coerce",
    )

    return matches.sort_values(
        "last_updated",
        ascending=False,
        na_position="last",
    )
