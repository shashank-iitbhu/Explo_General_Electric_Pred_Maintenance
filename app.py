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
    features = np.array([[type_of_material, air_temperature, process_temperature, rotational_speed, torque, tool_wear]])
    prediction = model.predict(features)
    return prediction[0]

# Sidebar inputs
st.sidebar.title("Enter Machine Parameters")
type_of_material = st.sidebar.slider("Type of Material", 1, 3, 2)
air_temperature = st.sidebar.slider("Air Temperature (Kelvin)", 290, 320, 305)
process_temperature = st.sidebar.slider("Process Temperature (Kelvin)", 290, 330, 310)
rotational_speed = st.sidebar.slider("Rotational Speed", 1200, 3000, 2100)
torque = st.sidebar.slider("Torque", 0, 80, 40)
tool_wear = st.sidebar.slider("Tool Wear", 0, 300, 150)

# Prediction
logistic_regression_prediction = predict_failure(logistic_regression_model, type_of_material, air_temperature, process_temperature, rotational_speed, torque, tool_wear)
random_forest_prediction = predict_failure(random_forest_model, type_of_material, air_temperature, process_temperature, rotational_speed, torque, tool_wear)

# Display predictions
st.write("Logistic Regression Prediction:", logistic_regression_prediction)
st.write("Random Forest Prediction:", random_forest_prediction)
