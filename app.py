import streamlit as st
import pandas as pd
import joblib
from datetime import date

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="SmartPremium Insurance Predictor",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    return model

model = load_model()

# -------------------------------
# App title
# -------------------------------
st.sidebar.info(
    """
    SmartPremium predicts insurance premium amounts
    using Machine Learning models and customer details.
    """
)

st.sidebar.subheader("Model Used")
st.sidebar.write("GradientBoostingRegressor")
st.title("💰 SmartPremium: Insurance Premium Prediction")
st.write("Predict insurance premium amount based on customer and policy details.")

st.divider()

st.sidebar.title("About Project")

# -------------------------------
# Input form
# -------------------------------
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    annual_income = st.number_input("Annual Income", min_value=0, value=500000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

with col2:
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    health_score = st.number_input("Health Score", min_value=0.0, max_value=100.0, value=50.0)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])

with col3:
    previous_claims = st.number_input("Previous Claims", min_value=0, max_value=20, value=0)
    vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=30, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    insurance_duration = st.number_input("Insurance Duration", min_value=1, max_value=30, value=5)
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])

col4, col5, col6 = st.columns(3)

with col4:
    exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])

with col5:
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

with col6:
    policy_start_date = st.date_input("Policy Start Date", value=date.today())

# -------------------------------
# Feature engineering
# -------------------------------
policy_year = policy_start_date.year
policy_month = policy_start_date.month
policy_day = policy_start_date.day

# -------------------------------
# Create input dataframe
# -------------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Annual Income": [annual_income],
    "Marital Status": [marital_status],
    "Number of Dependents": [dependents],
    "Education Level": [education],
    "Occupation": [occupation],
    "Health Score": [health_score],
    "Location": [location],
    "Policy Type": [policy_type],
    "Previous Claims": [previous_claims],
    "Vehicle Age": [vehicle_age],
    "Credit Score": [credit_score],
    "Insurance Duration": [insurance_duration],
    "Smoking Status": [smoking_status],
    "Exercise Frequency": [exercise_frequency],
    "Property Type": [property_type],
    "Policy_Year": [policy_year],
    "Policy_Month": [policy_month],
    "Policy_Day": [policy_day]
})

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Premium Amount"):
    prediction = model.predict(input_data)[0]

    st.markdown(
    f"""
    <div style="
        background-color:#d4edda;
        padding:20px;
        border-radius:10px;
        border:1px solid #28a745;
        text-align:center;
        font-size:28px;
        font-weight:bold;
        color:#155724;">
        Predicted Premium Amount <br><br>
        ₹ {prediction:,.2f}
    </div>
    """,
    unsafe_allow_html=True
)

    st.subheader("Input Data Used for Prediction")
    st.dataframe(input_data)