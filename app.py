from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import pickle


# =========================
# Load trained files
# =========================

with open("best_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("encoder.pkl", "rb") as encoder_file:
    encoders = pickle.load(encoder_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler_data = pickle.load(scaler_file)


# =========================
# FastAPI
# =========================

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# =========================
# Prediction function
# =========================

def make_prediction(input_data):

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, encoder in encoders.items():

        if col == "Churn":
            continue

        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Scale numerical columns
    numerical_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    input_df[numerical_cols] = scaler_data.transform(
        input_df[numerical_cols]
    )

    # Prediction
    prediction = loaded_model.predict(input_df)[0]

    probability = loaded_model.predict_proba(input_df)[0, 1]

    result = "Churn" if prediction == 1 else "No Churn"

    return result, float(probability)


# =========================
# Request model
# =========================

class PredictionRequest(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# =========================
# Home page
# =========================

@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None
        }
    )


# =========================
# Prediction API
# =========================

@app.post("/predict")
async def predict(data: PredictionRequest):

    input_data = data.model_dump()

    prediction, probability = make_prediction(input_data)

    return {
        "prediction": prediction,
        "probability": probability
    }