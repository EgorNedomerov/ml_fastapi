from fastapi import FastAPI, HTTPException, Query, Body, Request
from models import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn, ErrorResponseChurn
from dataset_loader import ChurnDataset, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, FEATURE_SCHEMA
from model_trainer import evaluate_churn_model, train_churn_model, load_churn_model, save_churn_model
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE_PATH = BASE_DIR / "data" / "churn_dataset.csv"
MODEL_FILE_PATH = BASE_DIR / "saved_models" / "churn_model.joblib"
TRAINING_HISTORY_FILE_PATH = BASE_DIR / "saved_models" / "training_history.json"

dataset = ChurnDataset(DATASET_FILE_PATH)

model_state = {
    "pipeline": None,
    "trained_at": None,
    "metrics": None,
    "model": None,
    "model_type": None,
    "hyperparameters": None,
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
    model_state["model_type"] = metadata.get("model_type")
    model_state["hyperparameters"] = metadata.get("hyperparameters")
    model_state["test_size"] = metadata.get("test_size")
    model_state["random_state"] = metadata.get("random_state")

def build_error_response(code: str, message: str, details=None) -> dict:
    return {"code": code,"message": message,"details": details}

def save_training_history(record: dict):
    if TRAINING_HISTORY_FILE_PATH.exists():
        with open(TRAINING_HISTORY_FILE_PATH, "r", encoding="utf-8") as file:
            history = json.load(file)
    else:
        history = []

    history.append(record)

    with open(TRAINING_HISTORY_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        error_content = build_error_response(
            code=exc.detail.get("code", f"HTTP_{exc.status_code}"),
            message=exc.detail.get("message", "Ошибка HTTP-запроса."),
            details=exc.detail.get("details")
        )
    else:
        error_content = build_error_response(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            details=None
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_content
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=build_error_response(
            code="VALIDATION_ERROR",
            message="Некорректный формат запроса или неверные типы значений.",
            details=exc.errors()
        )
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=build_error_response(
            code="DATA_PREPARATION_ERROR",
            message=str(exc),
            details=None
        )
    )

def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()

def build_prediction_dataframe(clients: list[FeatureVectorChurn]) -> pd.DataFrame:

    input_records = []

    for client in clients:
        input_records.append(model_to_dict(client))

    input_dataframe = pd.DataFrame(input_records)

    return input_dataframe[FEATURE_COLUMNS]

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
            "description": "Ошибка предсказания",
            "model": ErrorResponseChurn,
            "content": {
                "application/json": {
                    "examples": {
                        "model_not_trained": {
                            "summary": "Модель не обучена",
                            "value": {
                                "code": "MODEL_NOT_TRAINED",
                                "message": "Модель ещё не обучена. Сначала выполните POST /model/train.",
                                "details": None
                            }
                        },
                        "prediction_error": {
                            "summary": "Ошибка предсказания",
                            "value": {
                                "code": "PREDICTION_ERROR",
                                "message": "Ошибка при выполнении предсказания churn.",
                                "details": {
                                    "error": "Описание ошибки"
                                }
                            }
                        }
                    }
                }
            }
        },
        422: {
            "description": "Ошибка валидации входных данных",
            "model": ErrorResponseChurn,
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Некорректный формат запроса или неверные типы значений.",
                        "details": [
                            {
                                "loc": ["body", "monthly_fee"],
                                "msg": "Field required",
                                "type": "missing"
                            }
                        ]
                    }
                }
            }
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
            detail={
                "code": "MODEL_NOT_TRAINED",
                "message":"Модель ещё не обучена. Сначала выполните POST /model/train.",
                "details": None
            }
        )
    try:
        if isinstance(features, list):
            clients = features
        else:
            clients = [features]

        input_dataframe = build_prediction_dataframe(clients)

        predicted_classes = pipeline.predict(input_dataframe)
        predicted_probabilities = pipeline.predict_proba(input_dataframe)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "PREDICTION_ERROR",
                "message": "Ошибка при выполнении предсказания churn.",
                "details": {
                    "error": str(error)
                }
            }
        )
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

@app.post("/model/train",
          responses={
              200: {
        "description": "Модель успешно обучена"
    },
    400: {
        "description": "Ошибка конфигурации обучения",
        "model": ErrorResponseChurn,
        "content": {
            "application/json": {
                "examples": {
                    "unsupported_model_type": {
                        "summary": "Неподдерживаемый тип модели",
                        "value": {
                            "code": "TRAINING_CONFIG_ERROR",
                            "message": "Неподдерживаемый model_type. Допустимые значения: logreg, random_forest.",
                            "details": None
                        }
                    },
                    "wrong_hyperparameter": {
                        "summary": "Некорректный гиперпараметр",
                        "value": {
                            "code": "TRAINING_CONFIG_ERROR",
                            "message": "Некорректные hyperparameters для RandomForestClassifier: ...",
                            "details": None
                        }
                    }
                }
            }
        }
    },
    404: {
        "description": "Датасет не найден",
        "model": ErrorResponseChurn,
        "content": {
            "application/json": {
                "example": {
                    "code": "DATASET_NOT_FOUND",
                    "message": "Файл датасета не найден: ...",
                    "details": {
                        "dataset_path": "data/churn_dataset.csv"
                    }
                }
            }
        }
    },
    422: {
        "description": "Ошибка валидации TrainingConfigChurn",
        "model": ErrorResponseChurn
    },
    500: {
        "description": "Ошибка обучения модели",
        "model": ErrorResponseChurn
    }
}
)

def model_train(
     config: TrainingConfigChurn = Body(
        default=TrainingConfigChurn(),
        openapi_examples={
            "logistic_regression": {
                "summary": "Logistic Regression",
                "value": {
                    "model_type": "logreg",
                    "hyperparameters": {
                        "max_iter": 1000,
                        "class_weight": "balanced"
                    }
                }
            },
            "random_forest": {
                "summary": "Random Forest",
                "value": {
                    "model_type": "random_forest",
                    "hyperparameters": {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "random_state": 42
                    }
                }
            }
        }
    ),
    test_size: float = Query(0.2, ge=0.1, le=0.5),
    random_state: int = Query(42)
):

    try:
        X_train, X_test, y_train, y_test = dataset.get_train_test_data(
            test_size=test_size,
            random_state=random_state
        )

        (
        trained_pipeline,
        model_name,
        normalized_model_type,
        used_hyperparameters
        ) = train_churn_model(X_train, y_train, model_type=config.model_type, hyperparameters=config.hyperparameters,  random_state=random_state)
    
        metrics = evaluate_churn_model(
            pipeline=trained_pipeline,
            X_test=X_test,
            y_test=y_test
        )

        trained_at = datetime.now().isoformat(timespec="seconds")

        metadata = {
            "trained_at": trained_at,
            "metrics": metrics,
            "model": model_name,
            "model_type": normalized_model_type,
            "hyperparameters": used_hyperparameters,
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test)
        }

        save_churn_model(pipeline=trained_pipeline, file_path=MODEL_FILE_PATH, metadata=metadata)
        
        record = {
            "timestamp": trained_at,
            "model_type": normalized_model_type,
            "hyperparameters": used_hyperparameters,
            "metrics": metrics
        }
        save_training_history(record)

        model_state["pipeline"] = trained_pipeline
        model_state["trained_at"] = trained_at
        model_state["metrics"] = metrics
        model_state["model"] = model_name
        model_state["model_type"] = normalized_model_type
        model_state["hyperparameters"] = used_hyperparameters
        model_state["test_size"] = test_size
        model_state["random_state"] = random_state
        model_state["model_file_path"] = str(MODEL_FILE_PATH)

        return {
            "message": "model trained successfully",
            "model": model_name,
            "model_type": normalized_model_type,
            "hyperparameters": used_hyperparameters,
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
            detail={
                "code": "DATASET_NOT_FOUND",
                "message": str(error),
                "details": {
                    "dataset_path": str(DATASET_FILE_PATH)
                }
            }
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "TRAINING_CONFIG_ERROR",
                "message": str(error),
                "details": None
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MODEL_TRAINING_ERROR",
                "message": "Ошибка при обучении модели churn.",
                "details": {
                    "error": str(error)
                }
            }
        )

@app.get("/model/status")
def model_status():
    is_trained = model_state["pipeline"] is not None

    return {
        "is_trained": is_trained,
        "model": model_state["model"],
        "model_type": model_state["model_type"],
        "hyperparameters": model_state["hyperparameters"],
        "trained_at": model_state["trained_at"],
        "metrics": model_state["metrics"],
        "model_file_path": model_state["model_file_path"],
        "test_size": model_state["test_size"],
        "random_state": model_state["random_state"]
    }

@app.get("/model/schema")
def model_schema():
    return {
        "target": {
            "name": "churn",
            "type": "int",
            "description": "Целевая переменная: 1 — клиент ушёл, 0 — клиент не ушёл"
        },
        "features": {
            "all": FEATURE_COLUMNS,
            "numeric": NUMERIC_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS
        },
        "feature_schema": FEATURE_SCHEMA,
        "request_format": {
            "single_client": "POST /predict принимает один объект FeatureVectorChurn",
            "multiple_clients": "POST /predict принимает список объектов FeatureVectorChurn"
        }
    }
@app.get("/model/metrics")
def model_metrics(limit: int = Query(5, ge=1),model_type: str | None = None):
    
    if TRAINING_HISTORY_FILE_PATH.exists():
        with open(TRAINING_HISTORY_FILE_PATH, "r", encoding="utf-8") as file:
            history = json.load(file)
    else:
        history = []

    if model_type:
        history = [
            record for record in history
            if record["model_type"] == model_type
        ]

    return {
        "latest_training": history[-1] if history else None,
        "recent_history": history[-limit:],
        "total_records": len(history)
    }