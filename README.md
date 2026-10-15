# Telco churn — interactive demo

Interactive demo of the churn model from
[Teleco_churn-project](https://github.com/cruqza/Teleco_churn-project): pick a
customer profile, edit the features, and get an instant churn probability from
the exported scikit-learn pipeline (Logistic Regression — test accuracy 0.80,
churn recall 0.57). No retraining happens here; this repo ships the same
artifacts the FastAPI service serves.

- **Model & analysis:** [Teleco_churn-project](https://github.com/cruqza/Teleco_churn-project)

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

`python -m pytest` — contract tests (Seam 2): a sample customer returns a
probability in [0, 1] within seconds, and an at-risk customer scores higher
than a loyal one.
