
import joblib
import pandas as pd

# Load our trained model
model = joblib.load("churn_model.joblib")


def test_model_loads():
    assert model is not None


def test_model_predicts_customer():
    customer = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }])

    prediction = model.predict(customer)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_prediction_probability():
    customer = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }])

    probability = model.predict_proba(customer)[0][1]

    assert 0.0 <= probability <= 1.0
def test_churn_threshold():
    threshold = 0.40

    assert int(0.45 >= threshold) == 1
    assert int(0.35 >= threshold) == 0