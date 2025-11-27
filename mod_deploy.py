#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle


# In[2]:


log_model=pickle.load(open('logr.pkl','rb'))


# In[5]:


st.title('Model Deployment using LR')


# In[12]:


def user_input_parameter():    
    Pregnancies = st.sidebar.selectbox("Pregnancies", list(range(0, 16)))
    Glucose = st.sidebar.selectbox("Glucose Level", list(range(0, 301)))
    BloodPressure = st.sidebar.selectbox("Blood Pressure", list(range(0, 201)))
    SkinThickness = st.sidebar.selectbox("Skin Thickness", list(range(0, 101)))
    Insulin = st.sidebar.selectbox("Insulin", list(range(0, 901)))
    BMI = st.sidebar.selectbox("BMI", [x/1 for x in range(0, 701)])  
    DiabetesPedigreeFunction = st.sidebar.selectbox("DPF", [x/10 for x in range(0, 31)])
    Age = st.sidebar.selectbox("Age", list(range(1, 121)))
    
    dict1 = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    
    features = pd.DataFrame(dict1, index=[0])
    return features

df = user_input_parameter()
pred=log_model.predict(df)
pred_prob= log_model.predict_proba(df)
button=st.button('Predict')
if button is True:
    st.subheader('Predict')
    st.write('Diabities' if pred_prob[0][1]>=0.5 else 'Not diabities')
    st.subheader('Pred_Prob')
    st.write(pred_prob)


# In[ ]:




