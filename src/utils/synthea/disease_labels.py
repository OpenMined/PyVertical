"""
Code to generate labels for disease prevalence
"""
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy


def get_diagnosis_date(conditions: pd.DataFrame, disease: str) -> pd.DataFrame:
    """
    Get a dataset of disease diagnosis dates from
    a dataset of conditions

    Args:
        observations [pandas.DataFrame]: a dataset of disease conditions
        disease [str]: the name of the disease for which diagnosis dates will be calculated

    Returns:
        pandas.DataFrame
            Dataframe of patient IDs and date of diagnosis for `disease`.
            If patient does not have the disease, value is NaN
    """
    # Initialise dataframe
    dates = pd.DataFrame({"PATIENT": conditions["PATIENT"].unique()})
    dates["DIAGNOSIS DATE"] = pd.NaT

    # Make sure dates in observations are datetimes
    conditions["DATE"] = pd.to_datetime(conditions["DATE"])

    for row in dates.iterrows():
        patient_disease_dates = conditions.loc[
            (conditions["PATIENT"] == row["PATIENT"])
            & (conditions["DESCRIPTION"] == disease),
            "DATE",
        ]

        # if there is at least one date, find the earliest
        if not patient_disease_dates.empty:
            row["DIAGNOSIS DATE"] = patient_disease_dates.min()

    return dates


def get_binary_labels_for_disease(disease_dates: pd.DataFrame) -> "numpy.ndarray":
    """
    Get an array of binary values for does not have disease (0)
    and has disease (1) from a dataset of diagnosis dates

    Args:
        disease_dates [pandas.DataFrame]: dataset of diagnosis dates

    Returns:
        numpy.ndarray
            Array of binary values for disease prevalence
    """
    return (
        disease_dates["DIAGNOSIS DATE"]
        .apply(lambda x: not pd.isna(x))
        .astype(int)
        .values
    )
