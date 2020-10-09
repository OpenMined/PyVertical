"""
Code to aid working with synthea observations data
"""
from typing import List

import pandas as pd


def pivot_observations(
    observations: pd.DataFrame, codes: List, type_names: List
) -> pd.DataFrame:
    """
    Turn observations table into a table of values for specified
    observation types

    Args:
        observations [pandas.DataFrame]: dataset of observations
        codes [list of strings]: list of observation codes to include in dataset
        type_names [list of string]: list of column names to use in the pivoted table.
            One for each code

    Returns:
        pandas.DataFrame
            Pivoted table
    """
    assert len(codes) == len(type_names)

    pd.DataFrame(columns=["PATIENT", "DATE"] + type_names)

    for _patient, _date in observations.groupby(["PATIENT", "DATE"]).index:
        patient_subdata = observations.loc[
            (observations["PATIENT"] == _patient) & (observations["DATE"] == _date)
        ]

        for code in codes:
            observation_subset = observations.loc[observations["CODE"] == code]
