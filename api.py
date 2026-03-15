from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# create FastAPI app
app = FastAPI()

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load trained ML model
model = joblib.load("attrition_model.pkl")


@app.get("/")
def home():
    return {"message": "Attrition Prediction API is running"}


@app.post("/predict")
def predict(data: dict):

    # Load one sample row from training dataset
    df_template = pd.read_csv("MFG10YearTerminationData.csv").iloc[[0]].copy()

    # Replace values with incoming data
    df_template["age"] = data["age"]
    df_template["length_of_service"] = data["length_of_service"]
    df_template["city_name"] = data["city_name"]
    df_template["department_name"] = data["department_name"]
    df_template["job_title"] = data["job_title"]
    df_template["store_name"] = data["store_name"]
    df_template["gender_short"] = data["gender_short"]
    df_template["BUSINESS_UNIT"] = data["BUSINESS_UNIT"]
    df_template["STATUS_YEAR"] = data["STATUS_YEAR"]

    prediction = model.predict_proba(df_template)[0][1]

    return {"risk_score": float(prediction)}