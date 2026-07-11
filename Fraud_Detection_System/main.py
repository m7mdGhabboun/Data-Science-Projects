from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import time
import db_manager  # Importing the database script we just wrote

print("=== Starting ML API Service ===")

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Fraud Detection Inference API",
    description="Backend service for processing transactions and predicting fraud.",
    version="1.0"
)

# 2. Load the Pre-Trained Artifacts
# In a real environment, you would save these from your Jupyter Notebook using joblib
try:
    xgb_model = joblib.load("xgboost_model_v1.pkl")
    scaler = joblib.load("numerical_scaler.pkl")
    customer_avgs = joblib.load("customer_averages.pkl")
    merchant_pop = joblib.load("merchant_popularity.pkl")
    print("All ML artifacts loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load artifacts. Error: {e}")

# 3. Define the Data Validation Schema (Pydantic)
# This fulfills the requirement for proper validation and error checking
class TransactionInput(BaseModel):
    step: int = Field(..., description="1 step = 1 hour of time")
    customer_id: str = Field(..., max_length=50)
    zipcode_origin: str = Field(default="28007", max_length=20)
    merchant_id: str = Field(..., max_length=50)
    zip_merchant: str = Field(default="28007", max_length=20)
    age: str = Field(..., description="Age category (e.g., '2', '4')")
    gender: str = Field(..., description="Gender (e.g., 'M', 'F', 'U')")
    category: str = Field(..., description="Transaction category")
    amount: float = Field(..., gt=0, description="Transaction amount must be strictly greater than 0")

# 4. Helper Function: Live Feature Engineering
# def preprocess_live_transaction(data: TransactionInput) -> pd.DataFrame:
#     """
#     Transforms the raw incoming JSON into the exact 15 features expected by XGBoost.
#     """
#     # A. Scale the numerical amount
#     # Wrap in double brackets to create a 2D array, then extract the single value
#     scaled_amount = scaler.transform([[data.amount]])[0][0]
    
#     # B. Dictionary Lookups for Historical Data
#     # .get() uses a fallback value if a brand new customer/merchant appears
#     historical_cust_avg = customer_avgs.get(data.customer_id, 50.0) 
#     merchant_popularity_score = merchant_pop.get(data.merchant_id, 1) # Default to 1 transaction
    
#     # C. Calculate the engineered ratio
#     amount_to_cust_avg = data.amount / (historical_cust_avg + 0.0001)

#     # D. Construct the final 15-feature dictionary matching XGBoost's exact order
#     features = {
#         "amount": scaled_amount,
#         "HourOfDay": data.step % 24,
#         "DayOfWeek": (data.step // 24) % 7,
#         "Amount_to_Customer_Avg": amount_to_cust_avg,
#         "Merchant_Popularity": merchant_popularity_score,
#         "age_'2'": 1 if data.age == "2" else 0,
#         "age_'4'": 1 if data.age == "4" else 0,
#         "gender_'U'": 1 if data.gender == "U" else 0,
#         "category_'es_barsandrestaurants'": 1 if data.category == "es_barsandrestaurants" else 0,
#         "category_'es_fashion'": 1 if data.category == "es_fashion" else 0,
#         "category_'es_food'": 1 if data.category == "es_food" else 0,
#         "category_'es_hyper'": 1 if data.category == "es_hyper" else 0,
#         "category_'es_sportsandtoys'": 1 if data.category == "es_sportsandtoys" else 0,
#         "category_'es_transportation'": 1 if data.category == "es_transportation" else 0,
#         "category_'es_wellnessandbeauty'": 1 if data.category == "es_wellnessandbeauty" else 0
#     }
    
#     # Convert to a single-row DataFrame
#     return pd.DataFrame([features])

def preprocess_live_transaction(data: TransactionInput) -> pd.DataFrame:
    """
    Transforms raw JSON, handles BankSim quotes, and scales 5 numerical features simultaneously.
    """
    # 1. Clean incoming strings
    clean_age = data.age.replace("'", "")
    clean_gender = data.gender.replace("'", "")
    clean_category = data.category.replace("'", "")
    clean_customer = data.customer_id.replace("'", "")
    clean_merchant = data.merchant_id.replace("'", "")
    
    # 2. Dictionary Lookups for Historical Data
    dict_cust_key = f"'{clean_customer}'"
    dict_merch_key = f"'{clean_merchant}'"
    
    historical_cust_avg = customer_avgs.get(dict_cust_key, 50.0) 
    merchant_popularity_score = merchant_pop.get(dict_merch_key, 1) 
    
    # 3. Calculate raw numerical values
    raw_amount = data.amount
    raw_hour = data.step % 24
    raw_day = (data.step // 24) % 7
    raw_ratio = data.amount / (historical_cust_avg + 0.0001)
    raw_popularity = merchant_popularity_score

    # 4. Scale the 5 continuous features simultaneously
    # The list order MUST exactly match the 'numerical_cols' list from the notebook
    raw_features_array = [[raw_amount, raw_hour, raw_day, raw_ratio, raw_popularity]]
    scaled_array = scaler.transform(raw_features_array)[0] # Extract the transformed row

    # 5. Construct the final 15-feature dictionary matching XGBoost's exact order
    features = {
        # Unpack the newly scaled values
        "amount": scaled_array[0],
        "HourOfDay": scaled_array[1],
        "DayOfWeek": scaled_array[2],
        "Amount_to_Customer_Avg": scaled_array[3],
        "Merchant_Popularity": scaled_array[4],
        
        # Categorical Flags
        "age_'2'": 1 if clean_age == "2" else 0,
        "age_'4'": 1 if clean_age == "4" else 0,
        "gender_'U'": 1 if clean_gender == "U" else 0,
        "category_'es_barsandrestaurants'": 1 if clean_category == "es_barsandrestaurants" else 0,
        "category_'es_fashion'": 1 if clean_category == "es_fashion" else 0,
        "category_'es_food'": 1 if clean_category == "es_food" else 0,
        "category_'es_hyper'": 1 if clean_category == "es_hyper" else 0,
        "category_'es_sportsandtoys'": 1 if clean_category == "es_sportsandtoys" else 0,
        "category_'es_transportation'": 1 if clean_category == "es_transportation" else 0,
        "category_'es_wellnessandbeauty'": 1 if clean_category == "es_wellnessandbeauty" else 0
    }
    
    return pd.DataFrame([features])



# 5. Background Task: Database Insertion
def log_transaction_and_check_threshold(raw_data: TransactionInput, prediction_label: int):
    """
    Runs in the background to prevent slowing down the user's API response.
    """
    db_data = raw_data.dict()
    db_data['fraud'] = prediction_label
    
    # Insert into the secure vault
    inserted = db_manager.insert_raw_transaction(db_data)
    
    if inserted:
        # Check if we hit the limit to wake up the retraining script
        threshold_met = db_manager.track_and_check_threshold()
        if threshold_met:
            print(">>> THRESHOLD EVENT TRIGGERED: Initiating MLOps Retraining Cycle... <<<")
            # Here you would trigger the Continuous Training (CT) Python script using subprocess or a task queue

# 6. The Core API Endpoint
@app.post("/predict")
async def predict_fraud(transaction: TransactionInput, background_tasks: BackgroundTasks):
    """
    Receives frontend data, validates it, predicts fraud probability, and routes data to the DB.
    """
    try:
        # A. Preprocess the data
        df_features = preprocess_live_transaction(transaction)
        
        # B. Run Inference
        # (Assuming model is loaded. If testing without a model, use a mock probability)
        fraud_probability = xgb_model.predict_proba(df_features)[0][1]
        # fraud_probability = 0.05 # Mocked probability for API testing
        
        # Apply your strict mathematical threshold
        is_fraud = 1 if fraud_probability >= 0.80 else 0
        
        # C. Offload database insertion to a background thread
        background_tasks.add_task(log_transaction_and_check_threshold, transaction, is_fraud)
        
        # D. Return the JSON response to the Frontend
        return {
            "status": "success",
            "transaction_id_reference": f"{transaction.customer_id}-{transaction.step}",
            "fraud_probability": float(fraud_probability),
            "is_fraud": bool(is_fraud),
            "message": "Transaction analyzed securely."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server locally, you would execute: 
# uvicorn main:app --reload