from fastapi import FastAPI, HTTPException, Query, Body
from models import FeatureVectorChurn, PredictionResponseChurn
from dataset_loader import ChurnDataset, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from model_trainer import evaluate_churn_model, train_churn_model, load_churn_model, save_churn_model
from datetime import datetime
from pathlib import Path
import pandas as pd

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE_PATH = BASE_DIR / "data" / "churn_dataset.csv"
MODEL_FILE_PATH = BASE_DIR / "saved_models" / "churn_model.joblib"

dataset = ChurnDataset(DATASET_FILE_PATH)

model_state = {
    "pipeline": None,
    "trained_at": None,
    "metrics": None,
    "model": None,
    "model_file_path": str(MODEL_FILE_PATH),
    "test_size": None,
    "random_state": None
}

loaded_model_bundle = load_churn_model(MODEL_FILE_PATH)

if loaded_model_bundle is not None:
    model_state["pipeline"] = loaded_model_bundle.get("pipeline")

    metadata = loaded_model_bundle.get("metadata", {})

    model_state["trained_at"] = metadata.get("trained_at")
    model_state["metrics"] = metadata.get("metrics")
    model_state["model"] = metadata.get("model")
    model_state["test_size"] = metadata.get("test_size")
    model_state["random_state"] = metadata.get("random_state")

def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()

@app.get("/")
def root():
    return {"message": "ml churn service is running"}

@app.post("/predict",
    response_model=PredictionResponseChurn,
    responses={
        200: {
            "description": "Успешное предсказание churn",
            "content": {
                "application/json": {
                    "example": {
                        "message": "prediction completed successfully",
                        "model": "LogisticRegression",
                        "count": 1,
                        "predictions": [
                            {
                                "input_data": {
                                    "monthly_fee": 49.99,
                                    "usage_hours": 120.5,
                                    "support_requests": 2,
                                    "account_age_months": 14,
                                    "failed_payments": 0,
                                    "region": "europe",
                                    "device_type": "mobile",
                                    "payment_method": "card",
                                    "autopay_enabled": 1
                                },
                                "predicted_churn": 0,
                                "churn_probability": 0.2187,
                                "class_probabilities": {
                                    "0": 0.7813,
                                    "1": 0.2187
                                }
                            }
                        ]
                    }
                }
            }
        },
        400: {
            "description": "Модель ещё не обучена"
        }
    }
)

def predict(features: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
    ...,
        openapi_examples={
            "single_client": {
                "summary": "Один клиент",
                "value": {
                    "monthly_fee": 49.99,
                    "usage_hours": 120.5,
                    "support_requests": 2,
                    "account_age_months": 14,
                    "failed_payments": 0,
                    "region": "europe",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 1
                }
            },
            "several_clients": {
                "summary": "Несколько клиентов",
                "value": [
                    {
                        "monthly_fee": 9.99,
                        "usage_hours": 18.4,
                        "support_requests": 0,
                        "account_age_months": 19,
                        "failed_payments": 0,
                        "region": "asia",
                        "device_type": "desktop",
                        "payment_method": "card",
                        "autopay_enabled": 0
                    },
                    {
                        "monthly_fee": 89.99,
                        "usage_hours": 3.2,
                        "support_requests": 5,
                        "account_age_months": 2,
                        "failed_payments": 2,
                        "region": "america",
                        "device_type": "mobile",
                        "payment_method": "paypal",
                        "autopay_enabled": 0
                    }
                ]
            }
        }
    )
):

    pipeline = model_state["pipeline"]

    if pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="Модель ещё не обучена. Сначала выполните POST /model/train."
        )
    
    if isinstance(features, list):
        clients = features
    else:
        clients = [features]

    input_records = []

    for client in clients:
        input_records.append(model_to_dict(client))

    input_dataframe = pd.DataFrame(input_records)
    input_dataframe = input_dataframe[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]

    predicted_classes = pipeline.predict(input_dataframe)
    predicted_probabilities = pipeline.predict_proba(input_dataframe)

    prediction_results = []

    for index, predicted_class in enumerate(predicted_classes):
        class_probabilities = {}

        for class_label, probability in zip(pipeline.classes_, predicted_probabilities[index]):
            class_probabilities[str(int(class_label))] = round(float(probability), 4)

        prediction_results.append(
            {
                "input_data": clients[index],
                "predicted_churn": int(predicted_class),
                "churn_probability": class_probabilities.get("1", 0.0),
                "class_probabilities": class_probabilities
            }
        )

    return {
        "message": "prediction completed successfully",
        "model": model_state["model"],
        "count": len(prediction_results),
        "predictions": prediction_results
    }

@app.get("/dataset/preview")
def dataset_preview(limit: int = Query(5, ge=1, le=100)):
    try:
        return dataset.preview(limit=limit)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.get("/dataset/info")
def dataset_info():
    try:
        return dataset.info()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
    
@app.get("/dataset/split-info")
def dataset_split_info(
    test_size: float = Query(0.2, ge=0.1, le=0.5),
    random_state: int = Query(42)
):
    return dataset.train_test_split_info(
        test_size=test_size,
        random_state=random_state
    )

@app.post("/model/train")
def model_train(
    test_size: float = Query(0.2, ge=0.1, le=0.5),
    random_state: int = Query(42)
):

    try:
        X_train, X_test, y_train, y_test = dataset.get_train_test_data(
            test_size=test_size,
            random_state=random_state
        )

        trained_pipeline = train_churn_model(X_train, y_train)

        metrics = evaluate_churn_model(
            pipeline=trained_pipeline,
            X_test=X_test,
            y_test=y_test
        )

        trained_at = datetime.now().isoformat(timespec="seconds")

        metadata = {
            "trained_at": trained_at,
            "metrics": metrics,
            "model": "LogisticRegression",
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test)
        }

        save_churn_model(pipeline=trained_pipeline, file_path=MODEL_FILE_PATH, metadata=metadata)

        model_state["pipeline"] = trained_pipeline
        model_state["trained_at"] = trained_at
        model_state["metrics"] = metrics
        model_state["model"] = "LogisticRegression"
        model_state["test_size"] = test_size
        model_state["random_state"] = random_state
        model_state["model_file_path"] = str(MODEL_FILE_PATH)

        return {
            "message": "model trained successfully",
            "model": "LogisticRegression",
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "metrics": metrics,
            "model_saved": True,
            "model_file_path": str(MODEL_FILE_PATH),
            "trained_at": trained_at
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.get("/model/status")
def model_status():
    is_trained = model_state["pipeline"] is not None

    return {
        "is_trained": is_trained,
        "model": model_state["model"],
        "trained_at": model_state["trained_at"],
        "metrics": model_state["metrics"],
        "model_file_path": model_state["model_file_path"],
        "test_size": model_state["test_size"],
        "random_state": model_state["random_state"]
    }