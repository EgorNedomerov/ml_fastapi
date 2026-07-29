from fastapi import FastAPI, HTTPException, Query
from models import FeatureVectorChurn
from dataset_loader import ChurnDataset

app = FastAPI()
dataset = ChurnDataset("data/churn_dataset.csv")

@app.get("/")
def root():
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features

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