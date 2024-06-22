import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

# Load the dataset
medical_df = pd.read_csv('insurance.csv')

# Data preprocessing
medical_df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)

# Split data into features and target variable
X = medical_df.drop('charges', axis=1)
y = medical_df['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Train the linear regression model
lg = LinearRegression()
lg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lg.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)

# Streamlit web app
st.title("Medical Insurance Prediction Model")

# Input features from the user part by part
age = st.number_input("Enter age:", min_value=0, max_value=100, step=1)
sex = st.selectbox("Select sex:", options=["male", "female"])
bmi = st.number_input("Enter BMI:", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Enter number of children:", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Is the person a smoker?", options=["yes", "no"])
region = st.selectbox("Select region:", options=["southeast", "southwest", "northwest", "northeast"])

# Convert categorical data to numerical
sex = 0 if sex == "male" else 1
smoker = 0 if smoker == "yes" else 1
region_map = {"southeast": 0, "southwest": 1, "northwest": 2, "northeast": 3}
region = region_map[region]

# Predict insurance charges based on user input
input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                        columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

prediction = lg.predict(input_df)

st.write("Estimated medical insurance charges for this person:", prediction[0])
