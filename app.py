from fastapi import FastAPI, HTTPException
import pickle
import numpy as np

app = FastAPI(title="Cricket Win Probability Engine")

# Load the XGBoost model
try:
    model = pickle.load(open("models/xgb_model.pkl", "rb"))
except FileNotFoundError:
    raise RuntimeError("Model not found. Run ipl_model.py first to train and save it.")


@app.get("/")
def home():
    return {"message": "Cricket Win Probability Engine API is running"}


@app.get("/predict")
def predict(balls_remaining: int, wickets_fallen: int, crr: float, rrr: float):
    # Validate inputs
    if not (0 < balls_remaining <= 120):
        raise HTTPException(status_code=400, detail="balls_remaining must be between 1 and 120")
    if not (0 <= wickets_fallen <= 10):
        raise HTTPException(status_code=400, detail="wickets_fallen must be between 0 and 10")

    wickets_remaining = 10 - wickets_fallen

    # Feature order must match training: [balls_remaining, wickets_remaining, crr, rrr]
    data = np.array([[balls_remaining, wickets_remaining, crr, rrr]])
    prob = model.predict_proba(data)[0][1]

    # Match situation label
    if prob > 0.65:
        situation = "Strong chance of winning"
    elif prob > 0.45:
        situation = "Match is evenly poised"
    else:
        situation = "Under pressure"

    return {
        "win_probability": round(float(prob) * 100, 2),
        "lose_probability": round((1 - float(prob)) * 100, 2),
        "situation": situation,
        "inputs": {
            "balls_remaining": balls_remaining,
            "wickets_remaining": wickets_remaining,
            "current_run_rate": crr,
            "required_run_rate": rrr
        }
    }