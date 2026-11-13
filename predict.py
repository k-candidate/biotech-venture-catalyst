import pickle

from fastapi import FastAPI

from model import RawTrialInput, transform_raw_to_features

# Load the compiled production scikit-learn pipeline asset at system startup
model_file = 'model.bin'
with open(model_file, 'rb') as f_in:
    model_pipeline = pickle.load(f_in)

# Initialize FastAPI app layer
app = FastAPI(
    title="Biotech Venture Catalyst API",
    description="Production microservice accepting raw clinical parameters to predict Phase 1 Probability of Success (PoS).",
    version="1.1.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "pipeline_loaded": True}

@app.post("/predict")
def predict_trial_outcome(payload: RawTrialInput):
    """
    Accepts raw clinical-trial inputs, derives model features, 
    and returns the estimated probability of the positive outcome.
    """
    # Step 1: Execute encapsulated feature engineering logic
    engineered_dict = transform_raw_to_features(payload)
    
    # Step 2: Pass dictionary directly into scikit-learn pipeline
    # [0, 1] pulls the scalar probability value for Class 1 (Success)
    probability_of_success = float(model_pipeline.predict_proba([engineered_dict])[0, 1])
    
    return {
        "probability_of_success": round(probability_of_success, 5),
        "risk_status": "Low Risk" if probability_of_success >= 0.55 else "High Risk",
        "internals": engineered_dict  # Include the derived features in the response for transparency
    }
