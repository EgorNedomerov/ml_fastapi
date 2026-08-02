from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataset_loader import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from pathlib import Path
import joblib

def train_churn_model(X_train, y_train) -> Pipeline:

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS)
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_churn_model(pipeline: Pipeline, X_test, y_test) -> dict:

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1), 4)
    }

def save_churn_model(pipeline: Pipeline, file_path: str, metadata: dict) -> None:

    model_file_path = Path(file_path)

    model_file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {
        "pipeline": pipeline,
        "metadata": metadata
    }

    joblib.dump(
        model_bundle,
        model_file_path
    )

def load_churn_model(file_path: str) -> dict | None:

    model_file_path = Path(file_path)

    if not model_file_path.exists():
        return None

    model_bundle = joblib.load(model_file_path)

    return model_bundle