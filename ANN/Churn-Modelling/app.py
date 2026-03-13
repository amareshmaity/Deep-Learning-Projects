import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model = tf.keras.models.load_model('artifacts/model.h5')

## Load the encoder and scalar
with open('./artifacts/gender_label_encoder.pkl', 'rb') as file:
    gender_le = pickle.load(file)

with open('./artifacts/geo_onehot_encoder.pkl', 'rb') as file:
    geo_ohe = pickle.load(file)

with open('./artifacts/scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)



## Streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', geo_ohe.categories_[0])
gender = st.selectbox('Gender', gender_le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = geo_ohe.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_ohe.get_feature_names_out(['Geography']))

# Combine encoded cols to original data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Apply Scaling
input_data = scaler.transform(input_data)

# Prediction churn
prediction = model.predict(input_data)      # This usually returns something like [[0.857]]
prediction_prob = float(prediction[0][0])   # Extract the float value from the array
st.write(f"Churn Probability: {prediction_prob: .2f}")
possible_results = ['The customer is not likely to churn', 'The customer is likely to churn']
prediction_result = [possible_results[0] if prediction_prob<0.5 else possible_results[1]]
st.write(prediction_result[0])