"""
Input / Output utilities.
"""

from pathlib import Path
import pandas as pd


EXPECTED_FILES = {

    "multiclass":
        "multiclass_summary.csv",

    "binary":
        "binary_summary.csv",

    "optuna":
        "optuna_summary_multiclass.csv",

    "leakage":
        "leakage_summary.csv"
}


class DatasetManager:

    def __init__(self, input_folder):

        self.input_folder = Path(input_folder)

        self.datasets = {}

    def load(self):

        self.datasets = {}

        for key, filename in EXPECTED_FILES.items():

            file = self.input_folder / filename

            if file.exists():

                self.datasets[key] = pd.read_csv(file)

            else:

                print(f"[WARNING] {filename} not found.")

        return self.datasets

    def summary(self):

        print()

        print("=" * 70)

        print("DATASET SUMMARY")

        print("=" * 70)

        print()

        for name, df in self.datasets.items():

            print(f"{name}")

            print(f"Rows:      {df.shape[0]}")

            print(f"Columns:   {df.shape[1]}")

            print(f"Missing:   {df.isna().sum().sum()}")

            print()

    def get(self, name):

        return self.datasets[name]