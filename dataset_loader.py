from pathlib import Path
import pandas as pd
from models import DatasetRowChurn

class ChurnDataset:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.dataframe = pd.DataFrame()
        self.rows: list[DatasetRowChurn] = []

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Файл датасета не найден: {self.file_path}")

        self.dataframe = pd.read_csv(self.file_path)

        records = self.dataframe.to_dict(orient="records")

        self.rows = []

        for record in records:
            row = DatasetRowChurn(**record)
            self.rows.append(row)

        return self

    def preview(self, limit: int = 5) -> list[dict]:
        if self.dataframe.empty:
            self.load()

        limited_rows = self.rows[:limit]

        result = []

        for row in limited_rows:
            if hasattr(row, "model_dump"):
                result.append(row.model_dump())
            else:
                result.append(row.model_dump())

        return result

    def info(self) -> dict:
        if self.dataframe.empty:
            self.load()

        row_count = len(self.dataframe)
        column_count = len(self.dataframe.columns)

        feature_names = []

        for column in self.dataframe.columns:
            if column != "churn":
                feature_names.append(column)

        churn_counts = self.dataframe["churn"].value_counts().to_dict()

        churn_distribution = {}

        for churn_class, count in churn_counts.items():
            churn_distribution[str(churn_class)] = int(count)

        return {
            "row_count": row_count,
            "column_count": column_count,
            "feature_names": feature_names,
            "churn_distribution": churn_distribution
        }