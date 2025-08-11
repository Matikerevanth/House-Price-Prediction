import streamlit as st
import pandas as pd
import pickle
import os

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet', 
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM', 
    'XGBoost', 'KNN'
]
models = {}
for name in model_names:
    try:
        with open(f'{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ {name}.pkl not found. Skipping...")

# Load evaluation results
results_df = pd.read_csv('model_evaluation_results.csv')

# Streamlit UI
st.title("🏠 House Price Prediction App")

# Model selection
model_name = st.selectbox("Choose a model:", model_names)

# Input fields
income = st.number_input("Avg. Area Income", min_value=0.0, format="%.2f")
house_age = st.number_input("Avg. Area House Age", min_value=0.0, format="%.2f")
rooms = st.number_input("Avg. Area Number of Rooms", min_value=0.0, format="%.2f")
bedrooms = st.number_input("Avg. Area Number of Bedrooms", min_value=0.0, format="%.2f")
population = st.number_input("Area Population", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([{
        'Avg. Area Income': income,
        'Avg. Area House Age': house_age,
        'Avg. Area Number of Rooms': rooms,
        'Avg. Area Number of Bedrooms': bedrooms,
        'Area Population': population
    }])
    
    if model_name in models:
        prediction = models[model_name].predict(input_df)[0]
        st.success(f"💰 Predicted Price using {model_name}: **${prediction:,.2f}**")
    else:
        st.error("Model not found.")

# Show model evaluation results
st.subheader("📊 Model Evaluation Results")
st.dataframe(results_df)


