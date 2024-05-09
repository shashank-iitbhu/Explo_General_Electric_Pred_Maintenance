# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predictive Maintenance App")

# Load the trained models
logistic_regression_model = joblib.load('./models/LR.joblib')
random_forest_model = joblib.load('./models/RF.joblib')

def predict_failure(model, type_of_material, air_temperature, process_temperature, rotational_speed, torque, tool_wear):
    prediction = model([type_of_material, air_temperature, process_temperature, rotational_speed, torque, tool_wear])
    return prediction

# Sidebar inputs
st.sidebar.title("Enter Machine Parameters")
type_of_material = st.sidebar.slider("Type of Material", 1, 3, 2)
air_temperature = st.sidebar.slider("Air Temperature (Kelvin)", 290, 310, 300)
process_temperature = st.sidebar.slider("Process Temperature (Kelvin)", 290, 310, 300)
rotational_speed = st.sidebar.slider("Rotational Speed", 0, 100, 50)
torque = st.sidebar.slider("Torque", 0, 100, 50)
tool_wear = st.sidebar.slider("Tool Wear", 0, 100, 50)

# Prediction
logistic_regression_prediction = predict_failure(logistic_regression_model, type_of_material, air_temperature, process_temperature, rotational_speed, torque, tool_wear)
random_forest_prediction = predict_failure(random_forest_model, type_of_material, air_temperature, process_temperature, rotational_speed, torque, tool_wear)

# Display predictions
st.write("Logistic Regression Prediction:", logistic_regression_prediction)
st.write("Random Forest Prediction:", random_forest_prediction)
