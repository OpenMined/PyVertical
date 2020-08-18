"""
Create a dataset of has disease/does not have disease
for each patient in the study
"""
import argparse
from pathlib import Path

import pandas as pd


def main(args):
    data_folder = Path(__file__).resolve().parents[2] / "data" / "synthea"

    # Load conditions
    conditions = pd.read_csv(data_folder / "conditions.csv")

    # Dataset of all patients
    labels = pd.DataFrame({"PATIENT": conditions["PATIENT"].unique()})
    labels["HAS DISEASE"] = 0

    # Get unique patient IDs who have the disease of interest
    patients_with_disease = conditions.loc[
        conditions["DESCRIPTION"] == args.disease, "PATIENT"
    ].unique()
    labels.loc[labels["PATIENT"].isin(patients_with_disease), "HAS DISEASE"] = 1

    # Save file
    file_path = data_folder / "disease_labels.csv"
    print(f"Saving dataset of disease labels to {file_path}...")

    labels.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset of disease labels")
    parser.add_argument(
        "--disease", required=True, type=str, help="The name of the disease of interest"
    )

    args = parser.parse_args()

    main(args)
