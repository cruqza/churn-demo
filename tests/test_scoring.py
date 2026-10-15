"""Live Demo contract tests (Seam 2).

A sample telco customer submitted to the demo must return a churn probability
in [0, 1] within seconds. The two worked examples come from the flagship
repo's README and the verified FastAPI run: the month-to-month, tenure-1
customer churns ("Yes"); the two-year, tenure-72 customer stays ("No").
"""

import time

from scoring import score_customer

CHURNER = {
    "gender": "Female", "seniorcitizen": 0, "partner": "Yes", "dependents": "No",
    "tenure": 1, "phoneservice": "No", "multiplelines": "No phone service",
    "internetservice": "DSL", "onlinesecurity": "Yes", "onlinebackup": "Yes",
    "deviceprotection": "Yes", "techsupport": "Yes", "streamingtv": "Yes",
    "streamingmovies": "No", "contract": "Month-to-month",
    "paperlessbilling": "Yes", "paymentmethod": "Electronic check",
    "monthlycharges": 29.85, "totalcharges": 29.85,
}

LOYAL = {
    "gender": "Male", "seniorcitizen": 0, "partner": "Yes", "dependents": "Yes",
    "tenure": 72, "phoneservice": "Yes", "multiplelines": "Yes",
    "internetservice": "Fiber optic", "onlinesecurity": "Yes", "onlinebackup": "Yes",
    "deviceprotection": "Yes", "techsupport": "Yes", "streamingtv": "Yes",
    "streamingmovies": "Yes", "contract": "Two year",
    "paperlessbilling": "Yes", "paymentmethod": "Credit card (automatic)",
    "monthlycharges": 90.45, "totalcharges": 6869.35,
}


def test_sample_customer_returns_valid_probability_quickly():
    score_customer(CHURNER)  # warm the lru_cache: the app loads artifacts once at startup
    start = time.monotonic()
    result = score_customer(CHURNER)
    elapsed = time.monotonic() - start

    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in ("Yes", "No")
    assert elapsed < 5, f"scoring must take seconds, took {elapsed:.1f}s"


def test_churner_scores_higher_than_loyal_customer():
    churner = score_customer(CHURNER)
    loyal = score_customer(LOYAL)
    assert churner["prediction"] == "Yes"
    assert loyal["prediction"] == "No"
    assert churner["probability"] > loyal["probability"]
