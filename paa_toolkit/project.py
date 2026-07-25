from pathlib import Path
import pandas as pd


class Project:

    EXPECTED_DATASETS = {
        "multiclass": "optuna_summary_multiclass.csv",
        "binary": "binary_summary.csv",
        "leakage": "leakage_summary.csv",
    }

    def __init__(self, root):

        self.root = Path(root)

        self.input = self.root / "input"

        self.output = self.root / "output"

        self.figures = self.output / "figures"

        self.tables = self.output / "tables"

        self.reports = self.output / "reports"

        self.datasets = {}

    def create_folders(self):

        self.output.mkdir(exist_ok=True)

        self.figures.mkdir(exist_ok=True)

        self.tables.mkdir(exist_ok=True)

        self.reports.mkdir(exist_ok=True)

    def load(self):

        self.datasets = {}

        for key, filename in self.EXPECTED_DATASETS.items():

            file = self.input / filename

            if file.exists():

                self.datasets[key] = pd.read_csv(file)

            else:

                print(f"[WARNING] {filename} not found.")

    def summary(self):

        print("=" * 70)
        print("PROJECT")
        print("=" * 70)

        print(f"Root      : {self.root}")
        print(f"Input     : {self.input}")
        print(f"Output    : {self.output}")

        print()

        print("=" * 70)
        print("DATASETS")
        print("=" * 70)

        if not self.datasets:

            print("No datasets loaded.")

            return

        for name, df in self.datasets.items():

            print(f"{name}")

            print(f"Shape : {df.shape}")

            print(f"Missing : {df.isna().sum().sum()}")

            print()