import joblib
import pandas as pd
import streamlit as st

# Load the saved machine learning model
model = joblib.load("churn_model.joblib")

st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate churn risk.")

# Customer details
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior citizen?", [0, 1])
partner = st.selectbox("Has partner?", ["Yes", "No"])
dependents = st.selectbox("Has dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)

phone = st.selectbox("Phone service", ["Yes", "No"])
multiple_lines = st.selectbox(
    "Multiple lines", ["No", "Yes", "No phone service"]
)
internet = st.selectbox(
    "Internet service", ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online security",
    ["Yes", "No", "No internet service"]
)
backup = st.selectbox(
    "Online backup",
    ["Yes", "No", "No internet service"]
)
protection = st.selectbox(
    "Device protection",
    ["Yes", "No", "No internet service"]
)
support = st.selectbox(
    "Tech support",
    ["Yes", "No", "No internet service"]
)
tv = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
movies = st.selectbox(
    "Streaming movies", ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)
paperless = st.selectbox("Paperless billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly charges", min_value=0.0, value=70.0
)
total = st.number_input(
    "Total charges", min_value=0.0, value=840.0
)

# Match the original model's input column names
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

if st.button("Predict churn"):
    prediction = model.predict(customer)[0]
    probabilities = model.predict_proba(customer)[0]
    churn_probability = probabilities[1]

    st.metric(
        "Estimated churn probability",
        f"{churn_probability:.1%}"
    )

    if prediction == 1:
        st.warning("The model predicts that this customer may churn.")
    else:
        st.success("The model predicts that this customer may stay.")