#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import numpy as np
import pickle

# Load saved objects
model = pickle.load(open("rf_liver.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Liver Disease Prediction", layout="centered")

st.title("Liver Disease Prediction System")
st.write("Predict liver disease stage using clinical test values")

# Sidebar inputs
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", 1, 100)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
albumin = st.sidebar.number_input("Albumin")
alk_phos = st.sidebar.number_input("Alkaline Phosphatase")
alt = st.sidebar.number_input("ALT")
ast = st.sidebar.number_input("AST")
bilirubin = st.sidebar.number_input("Bilirubin")
cholinesterase = st.sidebar.number_input("Cholinesterase")
cholesterol = st.sidebar.number_input("Cholesterol")
creatinina = st.sidebar.number_input("Creatinina")
ggt = st.sidebar.number_input("GGT")
protein = st.sidebar.number_input("Protein")

# Encode sex
sex_encoded = 1 if sex == "Male" else 0

# Feature array (order MUST match training)
input_data = np.array([[ 
    age, sex_encoded, albumin, alk_phos, alt, ast,
    bilirubin, cholinesterase, cholesterol,
    creatinina, ggt, protein
]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f" Predicted Liver Disease Stage: **{prediction_label}**")

