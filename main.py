from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
import shap

from app.database import create_database, save_prediction


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="AI Traffic Accident Risk Prediction API",
    description="Machine Learning API for predicting accident severity",
    version="1.0.0"
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_PATH = r"C:\Users\HP\OneDrive\Desktop\AI Traffic Accident Risk Prediction"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "pre_accident_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "pre_accident_preprocessor.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "UK_Accident.csv"
)


# ---------------------------------------------------------
# Load trained model and preprocessor
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


# ---------------------------------------------------------
# Create database
# ---------------------------------------------------------

create_database()


# ---------------------------------------------------------
# Feature names
# ---------------------------------------------------------

features = [
    "Police_Force",
    "Speed_limit",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Special_Conditions_at_Site",
    "Carriageway_Hazards",
    "Urban_or_Rural_Area",
    "Year"
]


# ---------------------------------------------------------
# Prepare SHAP background data
# ---------------------------------------------------------

background_df = pd.read_csv(DATA_PATH)

background_df = background_df[features].dropna()

# Use a small sample to keep the API lightweight
background_df = background_df.sample(
    n=min(100, len(background_df)),
    random_state=42
)

background_processed = preprocessor.transform(
    background_df
)

# Convert sparse matrix to dense array if necessary
if hasattr(background_processed, "toarray"):
    background_processed = background_processed.toarray()


# ---------------------------------------------------------
# Create SHAP explainer
# ---------------------------------------------------------

explainer = shap.Explainer(
    model,
    background_processed
)


# ---------------------------------------------------------
# Input data structure
# ---------------------------------------------------------

class AccidentInput(BaseModel):

    Police_Force: int
    Speed_limit: int
    Light_Conditions: str
    Weather_Conditions: str
    Road_Surface_Conditions: str
    Special_Conditions_at_Site: str
    Carriageway_Hazards: str
    Urban_or_Rural_Area: int
    Year: int


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "AI Traffic Accident Risk Prediction API",
        "status": "running"
    }


# ---------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "database": "connected",
        "shap": "enabled"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict_accident(data: AccidentInput):

    input_df = pd.DataFrame([{
        "Police_Force": data.Police_Force,
        "Speed_limit": data.Speed_limit,
        "Light_Conditions": data.Light_Conditions,
        "Weather_Conditions": data.Weather_Conditions,
        "Road_Surface_Conditions": data.Road_Surface_Conditions,
        "Special_Conditions_at_Site": data.Special_Conditions_at_Site,
        "Carriageway_Hazards": data.Carriageway_Hazards,
        "Urban_or_Rural_Area": data.Urban_or_Rural_Area,
        "Year": data.Year
    }])

    processed_data = preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        processed_data
    )[0]

    # Save prediction to SQLite
    save_prediction(
        police_force=data.Police_Force,
        speed_limit=data.Speed_limit,
        light_conditions=data.Light_Conditions,
        weather_conditions=data.Weather_Conditions,
        road_surface_conditions=data.Road_Surface_Conditions,
        special_conditions=data.Special_Conditions_at_Site,
        carriageway_hazards=data.Carriageway_Hazards,
        urban_or_rural=data.Urban_or_Rural_Area,
        year=data.Year,
        predicted_severity=int(prediction)
    )

    return {
        "predicted_accident_severity": int(prediction),
        "prediction_saved": True
    }


# ---------------------------------------------------------
# SHAP explanation endpoint
# ---------------------------------------------------------

@app.post("/explain")
def explain_prediction(data: AccidentInput):

    input_df = pd.DataFrame([{
        "Police_Force": data.Police_Force,
        "Speed_limit": data.Speed_limit,
        "Light_Conditions": data.Light_Conditions,
        "Weather_Conditions": data.Weather_Conditions,
        "Road_Surface_Conditions": data.Road_Surface_Conditions,
        "Special_Conditions_at_Site": data.Special_Conditions_at_Site,
        "Carriageway_Hazards": data.Carriageway_Hazards,
        "Urban_or_Rural_Area": data.Urban_or_Rural_Area,
        "Year": data.Year
    }])

    # Preprocess input
    processed_data = preprocessor.transform(
        input_df
    )

    # Convert sparse matrix to dense
    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    # Prediction
    prediction = model.predict(
        processed_data
    )[0]

    # SHAP explanation
    shap_result = explainer(
        processed_data
    )

    shap_values = np.asarray(
        shap_result.values
    )

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Handle different SHAP output shapes
    if shap_values.ndim == 3:

        # Multiclass:
        # Select the class predicted by the model
        class_index = list(model.classes_).index(prediction)

        feature_importance = shap_values[
            0,
            :,
            class_index
        ]

    elif shap_values.ndim == 2:

        feature_importance = shap_values[0]

    else:

        feature_importance = shap_values.flatten()

    # Create explanation dataframe
    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "impact": feature_importance
    })

    # Sort by absolute impact
    explanation_df["absolute_impact"] = (
        explanation_df["impact"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "absolute_impact",
        ascending=False
    )

    # Return top 5 important factors
    top_features = explanation_df.head(5)

    explanation = []

    for _, row in top_features.iterrows():

        if row["impact"] > 0:
            direction = "increased"
        elif row["impact"] < 0:
            direction = "decreased"
        else:
            direction = "neutral"

        explanation.append({
            "feature": row["feature"],
            "impact": round(float(row["impact"]), 4),
            "direction": direction
        })

    return {
        "predicted_accident_severity": int(prediction),
        "top_factors": explanation
    }