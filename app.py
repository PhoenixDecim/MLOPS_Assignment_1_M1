from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()
MODELS = {
    "logistic_regression": "models/logistic_regression_model.joblib",
    "random_forest": "models/random_forest_model.joblib",
}
CLASS_LABELS = ["Setosa", "Versicolor", "Virginica"]
loaded_models = {name: joblib.load(path) for name, path in MODELS.items()}
scaler = joblib.load("models/standard_scaler.joblib")


class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    model_name: str


@app.get("/")
def read_root():
    return {"message": "Iris Prediction API"}


@app.post("/predict")
def predict(input_data: InputData):
    # Check if the model name is valid
    model_name = input_data.model_name
    if model_name not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail="Invalid model name. "
            f"Available models: {list(loaded_models.keys())}",
        )
    model = loaded_models[model_name]
    data = pd.DataFrame(
        [
            {
                "sepal.length": input_data.sepal_length,
                "sepal.width": input_data.sepal_width,
                "petal.length": input_data.petal_length,
                "petal.width": input_data.petal_width,
            }
        ]
    )
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    return {"model": model_name, "prediction": prediction}
