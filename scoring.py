"""Score telco customers with the exported churn model — no retraining.

Loads models/ml_pipeline.joblib (preprocessing + Logistic Regression, trained
in the flagship repo) and models/target_labels.joblib (the label encoder).
score_customer is the single entry point both the Streamlit app and the
tests use.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=1)
def _artifacts():
    model = joblib.load(MODELS_DIR / "ml_pipeline.joblib")
    labels = joblib.load(MODELS_DIR / "target_labels.joblib")
    return model, labels


def score_customer(customer: dict) -> dict:
    """Return {"probability": P(churn) in [0, 1], "prediction": "Yes" | "No"}.

    `customer` carries the 19 training features (lowercase column names).
    """
    model, labels = _artifacts()
    features = pd.DataFrame([customer])
    probability = float(model.predict_proba(features)[0, 1])
    prediction = str(labels.inverse_transform([int(probability >= 0.5)])[0])
    return {"probability": probability, "prediction": prediction}
