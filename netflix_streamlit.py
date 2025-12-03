import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load("netflix_churn_model.joblib")

st.set_page_config(page_title="Netflix Churned Customers Predictor", layout="centered")

st.title("🏡 Netflix Churn Customer Prediction App")
st.markdown("Enter the customer details below to predict their churn risk.")

# Define input fields
# Numeric columns
age = st.number_input("Age", min_value=18, max_value=70, value=25, step=1)
watch_hours = st.number_input("Total watch hours", min_value=0, max_value=120, value=20, step=1)
last_login_days  = st.number_input("No. of last login days", min_value=0, max_value=60, value=5, step=1)
monthly_fee = st.number_input("Monthly fee", min_value=8.99, max_value=17.99, value=8.99, step=1)
number_of_profiles = st.number_input("No. of profiles", min_value=1, max_value=5, value=1, step=1)
avg_watch_time_per_day = st.number_input("Average watch time per day", min_value=0, max_value=100, value=1, step=1)

# Categorical columns (same as those used in training)
gender = st.selectbox("Gender", ['Female', 'Male','Other'])
subscription_type = st.selectbox("Subscription Type", ['Premium', 'Basic', 'Standard'])
region = st.selectbox("Region", ['South America','Europe','North America','Asia','Africa','Oceania'])
device = st.selectbox("Device", ['Tablet','Laptop','Mobile','TV','Desktop'])
payment_method = st.selectbox("Payment Method", ['Debit Card','Paypal','Crypto','Gift Card','Credit Card'])
favorite_genre = st.selectbox("Favorite Genre", ['Drama','Documentary','Romance','Sci-Fi','Horror','Action','Comedy'])

feature_order = [
    "age", "gender","subscription_type","watch_hours", "last_login_days","region", "device", "monthly_fee","payment_method", "number_of_profiles", "avg_watch_time_per_day","favorite_genre"
]

# Create a DataFrame with one row for prediction
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "subscription_type": [subscription_type],
    "watch_hours": [watch_hours],
    "last_login_days": [last_login_days],
    "region": [region],
    "device": [device],
    "monthly_fee": [monthly_fee],
    "payment_method": [payment_method],
    "number_of_profiles": [number_of_profiles],
    "avg_watch_time_per_day": [avg_watch_time_per_day],
    "favorite_genre": [favorite_genre]
},columns=feature_order)


# Predict button
if st.button("PREDICT CHURN RISK"):
    try:
        prediction = model_pipeline.predict(input_data)[0]
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 CHURN PREDICTION: Customer is at **HIGH RISK** of Churning!")
        else:
            st.success(f"✅ CHURN PREDICTION: Customer is predicted to be **RETAINED**.")
    except Exception as e:
        st.error(f" Prediction failed: {e}")
        st.write("Please check that the model and inputs match the training columns.")