from pathlib import Path
import pandas as pd
from models import DatasetRowChurn
from sklearn.model_selection import train_test_split

NUMERIC_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled"
]

CATEGORICAL_COLUMNS = [
     "region",
    "device_type",
    "payment_method"
]

TARGET_COLUMN = "churn"

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

    def _check_required_columns(self):

        required_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]

        missing_columns = []

        for column in required_columns:
            if column not in self.dataframe.columns:
                missing_columns.append(column)

        if missing_columns:
            raise ValueError(f"В датасете отсутствуют колонки: {missing_columns}")

    def _fill_missing_values(self):
        for column in NUMERIC_COLUMNS:
            if self.dataframe[column].isna().sum() > 0:
                median_value = self.dataframe[column].median()
                self.dataframe[column] = self.dataframe[column].fillna(median_value)

        for column in CATEGORICAL_COLUMNS:
            if self.dataframe[column].isna().sum() > 0:
                self.dataframe[column] = self.dataframe[column].fillna("unknown")

        if self.dataframe[TARGET_COLUMN].isna().sum() > 0:
            raise ValueError("В целевой колонке churn есть пропуски")

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
            "churn_distribution": churn_distribution,
            "numeric_features": NUMERIC_COLUMNS,
            "categorical_features": CATEGORICAL_COLUMNS,
            "target_column": TARGET_COLUMN
        }

    def prepare_data(self):
        if self.dataframe.empty:
            self.load()

        X = self.dataframe[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        y = self.dataframe[TARGET_COLUMN]

        return X, y

    def get_train_test_data(self, test_size: float = 0.2, random_state: int = 42):
        if self.dataframe.empty:
            self.load()

        if self.dataframe.empty:
            raise ValueError("Датасет пуст. Невозможно выполнить разбиение на train/test.")

        X, y = self.prepare_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_test_split_info(self, test_size: float = 0.2, random_state: int = 42) -> dict:

        X, y = self.prepare_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        train_churn_distribution = self._get_churn_distribution_from_series(y_train)
        test_churn_distribution = self._get_churn_distribution_from_series(y_test)

        return {
            "total_rows": len(self.dataframe),
            "test_size": test_size,
            "random_state": random_state,
            "features": {
                "numeric": NUMERIC_COLUMNS,
                "categorical": CATEGORICAL_COLUMNS,
                "target": TARGET_COLUMN
            },
            "train": {
                "rows": len(X_train),
                "columns": len(X_train.columns),
                "churn_distribution": train_churn_distribution
            },
            "test": {
                "rows": len(X_test),
                "columns": len(X_test.columns),
                "churn_distribution": test_churn_distribution
            }
        }

    def _get_churn_distribution(self, dataframe: pd.DataFrame) -> dict:

        churn_counts = dataframe[TARGET_COLUMN].value_counts().to_dict()

        churn_distribution = {}

        for churn_class, count in churn_counts.items():
            churn_distribution[str(churn_class)] = int(count)

        return churn_distribution

    def _get_churn_distribution_from_series(self, target: pd.Series) -> dict:

        churn_counts = target.value_counts().to_dict()

        churn_distribution = {}

        for churn_class, count in churn_counts.items():
            churn_distribution[str(churn_class)] = int(count)

        return churn_distribution