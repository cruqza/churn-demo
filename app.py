"""Telco churn — live demo of the flagship model.

Streamlit front-end over scoring.score_customer. Everything a recruiter
needs is above the fold: pick a customer (or edit the fields), press
Predict, get a churn probability.
"""

import streamlit as st

from scoring import score_customer

st.set_page_config(page_title="Telco Churn — Live Demo", page_icon="📞", layout="centered")

EXAMPLES = {
    "At-risk customer (month-to-month, new)": {
        "gender": "Female", "seniorcitizen": 0, "partner": "Yes", "dependents": "No",
        "tenure": 1, "phoneservice": "No", "multiplelines": "No phone service",
        "internetservice": "DSL", "onlinesecurity": "Yes", "onlinebackup": "Yes",
        "deviceprotection": "Yes", "techsupport": "Yes", "streamingtv": "Yes",
        "streamingmovies": "No", "contract": "Month-to-month",
        "paperlessbilling": "Yes", "paymentmethod": "Electronic check",
        "monthlycharges": 29.85, "totalcharges": 29.85,
    },
    "Loyal customer (two-year contract, 6 years in)": {
        "gender": "Male", "seniorcitizen": 0, "partner": "Yes", "dependents": "Yes",
        "tenure": 72, "phoneservice": "Yes", "multiplelines": "Yes",
        "internetservice": "Fiber optic", "onlinesecurity": "Yes", "onlinebackup": "Yes",
        "deviceprotection": "Yes", "techsupport": "Yes", "streamingtv": "Yes",
        "streamingmovies": "Yes", "contract": "Two year",
        "paperlessbilling": "Yes", "paymentmethod": "Credit card (automatic)",
        "monthlycharges": 90.45, "totalcharges": 6869.35,
    },
}

st.title("📞 Will this customer churn?")
st.caption(
    "Live demo of a logistic-regression model trained on the IBM Telco dataset "
    "(7,032 customers) — the full project lives on "
    "[GitHub](https://github.com/cruqza/Teleco_churn-project)."
)

example = st.selectbox("Start from an example, then edit below", list(EXAMPLES))
customer = dict(EXAMPLES[example])

left, right = st.columns(2)
with left:
    customer["gender"] = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(customer["gender"]))
    customer["seniorcitizen"] = 1 if st.checkbox("Senior citizen") else 0
    customer["partner"] = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(customer["partner"]))
    customer["dependents"] = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(customer["dependents"]))
    customer["tenure"] = st.slider("Tenure (months)", 0, 72, customer["tenure"])
    customer["contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                        index=["Month-to-month", "One year", "Two year"].index(customer["contract"]))
    customer["paymentmethod"] = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)",
               "Credit card (automatic)"].index(customer["paymentmethod"]))
with right:
    customer["internetservice"] = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"],
                                               index=["DSL", "Fiber optic", "No"].index(customer["internetservice"]))
    customer["monthlycharges"] = st.number_input("Monthly charges ($)", 0.0, 200.0, float(customer["monthlycharges"]))
    customer["totalcharges"] = st.number_input("Total charges ($)", 0.0, 10000.0, float(customer["totalcharges"]))
    for field, label in [("phoneservice", "Phone service"), ("multiplelines", "Multiple lines"),
                         ("onlinesecurity", "Online security"), ("onlinebackup", "Online backup"),
                         ("deviceprotection", "Device protection"), ("techsupport", "Tech support"),
                         ("streamingtv", "Streaming TV"), ("streamingmovies", "Streaming movies"),
                         ("paperlessbilling", "Paperless billing")]:
        options = ["Yes", "No"] + (["No phone service"] if field == "multiplelines" else [])
        if field == "multiplelines" and customer[field] not in options:
            options.append(customer[field])
        current = customer[field] if customer[field] in options else "Yes"
        customer[field] = st.selectbox(label, options, index=options.index(current))

if st.button("Predict churn", type="primary", use_container_width=True):
    result = score_customer(customer)
    probability = result["probability"]
    if result["prediction"] == "Yes":
        st.error(f"### 🔴 Likely to churn — {probability:.0%}")
    else:
        st.success(f"### 🟢 Likely to stay — {probability:.0%}")
    st.progress(probability, text=f"Churn probability: {probability:.1%}")
    st.caption(
        "Model: Logistic Regression · test accuracy 0.80 · churn recall 0.57 · "
        "trained on the IBM Telco dataset. This demo scores instantly; no data is stored."
    )
