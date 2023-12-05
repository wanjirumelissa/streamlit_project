#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

# Print current working directory and list of files
print("Current Working Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())

# Load the model
model_path = "best_model1.pkl"
loaded_model = joblib.load(model_path)

# Streamlit App
st.title("Linear Regression Model Explorer")

# Upload CSV data through Streamlit
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Summary statistics and missing values
    st.subheader("Dataset Summary")
    st.write(df.describe())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # One-hot encoding
    st.subheader("One-Hot Encoded DataFrame")
    df_encoded = pd.get_dummies(df, columns=['marital', 'ed', 'jobsat', 'gender'], drop_first=True)
    st.write(df_encoded.head())

    # Standardization
    st.subheader("Standardized DataFrame")
    numeric_columns = ['age', 'inccat', 'car', 'carcat', 'employ']
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    st.write(df[numeric_columns].head())

    # Prediction using the linear regression model
    st.subheader("Linear Regression Model Prediction")

    # Select features for prediction
    features = st.multiselect("Select features for prediction", numeric_columns)

    if st.button("Predict"):
        X_pred = df[features]
        predictions = loaded_model.predict(X_pred)
        df["Predicted Income"] = predictions
        st.write(df[["income", "Predicted Income"]])

    # Model coefficients
    st.subheader("Model Coefficients")
    st.write("Coefficients:", loaded_model.coef_)
    st.write("Intercept:", loaded_model.intercept_)

    # Evaluation metrics
    st.subheader("Model Evaluation Metrics")
    y_test = df["income"]  
    y_pred = loaded_model.predict(df[numeric_columns])
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared:", r2_score(y_test, y_pred))


# In[ ]:




