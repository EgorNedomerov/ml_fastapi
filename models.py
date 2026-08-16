from typing import Any
from pydantic import BaseModel, Field, ConfigDict

class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

class DatasetRowChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int
    churn: int

class PredictionItemChurn(BaseModel):
    input_data: FeatureVectorChurn
    predicted_churn: int
    churn_probability: float
    class_probabilities: dict[str, float]


class PredictionResponseChurn(BaseModel):
    message: str
    model: str | None
    count: int
    predictions: list[PredictionItemChurn]

class TrainingConfigChurn(BaseModel):
    model_type: str = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

class ErrorResponseChurn(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | list[dict[str, Any]] | None = None