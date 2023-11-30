#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
import joblib

# Load your dataset
file_path = 'C:/Users/admin/Documents/Project_ADS/dataset1.csv'
df = pd.read_csv(file_path)

# Load your dataset
file_path = 'C:/Users/admin/Documents/Project_ADS/dataset1.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Display general information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Use the pd.get_dummies() function for one-hot encoding
df_encoded = pd.get_dummies(df, columns=['marital', 'ed', 'jobsat', 'gender'], drop_first=True)

# Display the first few rows of the DataFrame after encoding
print(df_encoded.head())

# Select the columns to be standardized
numeric_columns = ['age', 'inccat', 'car', 'carcat', 'employ']

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Display the first few rows of the DataFrame after scaling
print(df[numeric_columns].head())

# Assume 'target' is the name of your target variable (the variable you want to predict)
target = 'income'

# Split the data into features (X) and target variable (y)
X = df[numeric_columns]
y = df[target]

# Split the data into training and testing sets
# Adjust the test_size parameter to control the proportion of data used for testing
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Create a linear regression model
linear_regression_model = LinearRegression()

# Fit the model to the training data
linear_regression_model.fit(X_train, y_train)

# Display the coefficients
print("Coefficients:", linear_regression_model.coef_)
print("Intercept:", linear_regression_model.intercept_)

# Make predictions on the test set
y_pred = linear_regression_model.predict(X_test)

# Evaluation metrics (you may replace this with your preferred evaluation metrics)
from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Use the model for predictions (replace with your actual model)
best_model = linear_regression_model

joblib.dump(best_model, "C:/Users/admin/Documents/Project_ADS/best_model1.pkl")


# In[ ]:




