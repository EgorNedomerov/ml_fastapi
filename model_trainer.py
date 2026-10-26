from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataset_loader import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

def create_churn_model(
    model_type: str,
    hyperparameters: dict | None = None,
    random_state: int = 42
):
    if hyperparameters is None:
        hyperparameters = {}

    normalized_model_type = model_type.lower().strip()

    if normalized_model_type == "logreg":
        used_hyperparameters = {
            "max_iter": 1000
        }

        used_hyperparameters.update(hyperparameters)

        try:
            model = LogisticRegression(**used_hyperparameters)

        except TypeError as error:
            raise ValueError(
                f"Некорректные hyperparameters для LogisticRegression: {error}"
            )

        return model, "LogisticRegression", normalized_model_type, used_hyperparameters

    if normalized_model_type == "random_forest":
        used_hyperparameters = {
            "n_estimators": 100,
            "random_state": random_state
        }

        used_hyperparameters.update(hyperparameters)

        try:
            model = RandomForestClassifier(**used_hyperparameters)

        except TypeError as error:
            raise ValueError(
                f"Некорректные hyperparameters для RandomForestClassifier: {error}"
            )

        return model, "RandomForestClassifier", normalized_model_type, used_hyperparameters

    raise ValueError(
        "Неподдерживаемый model_type. Допустимые значения: logreg, random_forest."
    )

def train_churn_model(X_train, y_train, model_type: str = "logreg", hyperparameters: dict | None = None, random_state: int = 42 ) -> tuple[Pipeline, str, str, dict]:

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

    model, model_name, normalized_model_type, used_hyperparameters = create_churn_model(
        model_type=model_type,
        hyperparameters=hyperparameters,
        random_state=random_state
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline, model_name, normalized_model_type, used_hyperparameters


def evaluate_churn_model(pipeline: Pipeline, X_test, y_test) -> dict:

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    roc_auc = None
    
    if hasattr(pipeline, "predict_proba") and len(set(y_test)) == 2:
        predicted_probabilities = pipeline.predict_proba(X_test)

        class_labels = list(pipeline.classes_)

        if 1 in class_labels:
            churn_class_index = class_labels.index(1)
            churn_probabilities = predicted_probabilities[:, churn_class_index]

            roc_auc = roc_auc_score(y_test, churn_probabilities)

    return {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None
    }

def save_churn_model(pipeline: Pipeline, file_path: str | Path, metadata: dict) -> None:

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

def load_churn_model(file_path: str | Path) -> dict | None:

    model_file_path = Path(file_path)

    if not model_file_path.exists():
        return None

    if model_file_path.stat().st_size == 0:
        return None
    
    try:
        model_bundle = joblib.load(model_file_path)

    except EOFError:
        return None

    return model_bundle