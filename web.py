import pandas as pd
# import numpy as np
from IPython.display import (display, display_html, display_png, display_svg)
from matplotlib import pyplot
pyplot.rcParams['font.sans-serif'] = ['Microsoft YaHei']
pyplot.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier
import shap

import joblib
import streamlit as st

# Load the model
model = joblib.load('tabnet.pkl')

# Define feature names
feature_names = ['BMI','Sex','Age','GRACE risk score','TIMI risk score','History of smoking','No quitting smoking',
 'History of hypertension','History of diabetes','History of hyperlipemia','History of CVD','History of PU','History of myocardial infarction',
 'History of stent implantation','History of CABG','History of AF','Hemoglobin','Platelet',
 'Leukocytes','Neutrophils','Lymphocytes','Monocytes','Blood glucose','ALT','AST','Creatinine',
 'Creatinine clearance rate','TC','LDL-C','HDL-C','TG','UA','Hba1c','hs-CRP','IL-6','NT-pro BNP',
 'Peak troponin I','LVEF','Left atrial diameter','LVEDD','Hospitalization days']

# blood test indicators  21
blood_indicators = ['Hemoglobin','Platelet','Leukocytes', 'Neutrophils','Lymphocytes','Monocytes',
 'Blood glucose','ALT','AST','Creatinine',
 'Creatinine clearance rate','TC','LDL-C','HDL-C','TG','UA','Hba1c','hs-CRP','IL-6','NT-pro BNP',
 'Peak troponin I']
# 超声指标 4
echocardiographic_indicators = ['LVEF','Left atrial diameter','LVEDD','Hospitalization days']
# 人口学指标 3
demographic_indicators = ['BMI','Sex','Age']
# 临床分数 2
clinical_score = ['GRACE risk score','TIMI risk score']
# 既往史 11 
medical_history = ['History of smoking','No quitting smoking',
 'History of hypertension','History of diabetes','History of hyperlipemia','History of CVD','History of PU','History of myocardial infarction',
 'History of stent implantation','History of CABG','History of AF']

# Streamlit user interface
st.title("KILLIP Predictor")

input_values = {}  
feature_values = []

st.subheader("demographic indicators") 
col11, col12, col13, col14 = st.columns(4) 
for i, demographic_indicator in enumerate(demographic_indicators):
    if i == 0:
        with col11:
            input_value = st.number_input(f"{demographic_indicator}:")
            feature_values.append(input_value)
            input_values[demographic_indicator] = input_value 
    elif i == 1:
        with col12:
            input_value = st.number_input(f"{demographic_indicator}:")
            feature_values.append(input_value)
            input_values[demographic_indicator] = input_value 
    elif i == 2:
        with col13:
            input_value = st.number_input(f"{demographic_indicator}:")
            feature_values.append(input_value)
            input_values[demographic_indicator] = input_value


st.subheader("clinical scores")
col21, col22, col23, col24 = st.columns(4)
with col21:
    input_value = st.number_input(f"{clinical_score[0]}:")
    feature_values.append(input_value)   
    input_values[clinical_score[0]] = input_value
with col22:
    input_value = st.number_input(f"{clinical_score[1]}:")
    feature_values.append(input_value)   
    input_values[clinical_score[1]] = input_value


st.subheader("medical history")
col31, col32, col33, col34 = st.columns(4)
for i, medical_his in enumerate(medical_history):
    if i in [0, 4, 8, 12]:
        with col31:
            input_value = st.number_input(f"{medical_his}:")
            feature_values.append(input_value)
            input_values[medical_his] = input_value 
    elif i in [1, 5, 9]:
        with col32:
            input_value = st.number_input(f"{medical_his}:")
            feature_values.append(input_value)
            input_values[medical_his] = input_value 
    elif i in [2, 6, 10]:
        with col33:
            input_value = st.number_input(f"{medical_his}:")
            feature_values.append(input_value)
            input_values[medical_his] = input_value
    else:
        with col34:
            input_value = st.number_input(f"{medical_his}:")
            feature_values.append(input_value)
            input_values[medical_his] = input_value
            
st.subheader("blood indicators") 
col41, col42, col43, col44 = st.columns(4) 
for i, blood_indicator in enumerate(blood_indicators):
    if i in [0, 4, 8, 12, 16, 20]:
        with col41:
            input_value = st.number_input(f"{blood_indicator}:")
            feature_values.append(input_value)
            input_values[blood_indicator] = input_value 
    elif i in [1, 5, 9, 13, 17]:
        with col42:
            input_value = st.number_input(f"{blood_indicator}:")
            feature_values.append(input_value)
            input_values[blood_indicator] = input_value 
    elif i in [2, 6, 10, 14, 18]:
        with col43:
            input_value = st.number_input(f"{blood_indicator}:")
            feature_values.append(input_value)
            input_values[blood_indicator] = input_value
    else:
        with col44:
            input_value = st.number_input(f"{blood_indicator}:")
            feature_values.append(input_value)
            input_values[blood_indicator] = input_value  

col5, col6 = st.columns([0.75, 0.25])
with col5:
    st.subheader("echocardiographic indicators") 
    col51, col52, col53 = st.columns(3) 
    for i, echocardiographic_indicator in enumerate(echocardiographic_indicators):
        if i == 0:
            with col51:
                input_value = st.number_input(f"{echocardiographic_indicator}:")
                feature_values.append(input_value)
                input_values[echocardiographic_indicator] = input_value 
        elif i == 1:
            with col52:
                input_value = st.number_input(f"{echocardiographic_indicator}:")
                feature_values.append(input_value)
                input_values[echocardiographic_indicator] = input_value 
        elif i == 2:
            with col53:
                input_value = st.number_input(f"{echocardiographic_indicator}:")
                feature_values.append(input_value)
                input_values[echocardiographic_indicator] = input_value 
with col6:
    st.subheader("other")   
    input_value = st.number_input(f"{echocardiographic_indicators[3]}:")
    feature_values.append(input_value)   
    input_values[echocardiographic_indicators[3]] = input_value                                      


feature_final = []
for i, feature in enumerate(feature_names): 
    feature_final.append(input_values[feature])
features = pd.DataFrame([feature_final])

# features = np.array([feature_final])

# for i, feature in enumerate(feature_names):  # 从1到40  
#     # 构造变量的名字，这里简单地使用'var'加上序号  
#     var_name = f'var{i}'  
#     # 使用streamlit的number_input来获取输入值  
#     # 注意：由于streamlit的交互性，这里的变量名（var_name）不能直接用作字典的键来存储值  
#     # 因此，我们需要在循环外部使用一个固定的键（比如这里的变量名作为字符串）来存储对应的值  
#     input_value = st.number_input(f"{feature}:")  # 假设默认值为0  
      
#     # 将输入值存储在字典中，使用变量名（作为字符串）作为键  
#     input_values[var_name] = input_value 
#     feature_values.append(input_value) 

# # Process inputs and make predictions
# features = np.array([feature_values])
killip = {0: "KILLIP 1", 1: "KILLIP 2", 2: "KILLIP 3", 3: "KILLIP 4"}
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**Predicted Class:** {killip[predicted_class]}")
    probability = predicted_proba[predicted_class] * 100
    st.write(f"**Based on features values, prediction probabilities of {killip[predicted_class]} is:** {probability:.1f}%")


    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_final], columns=feature_names))
    if predicted_class == 0:
        shap.force_plot(explainer.expected_value[0], shap_values[0], pd.DataFrame([feature_final], columns=feature_names), matplotlib=True)
    elif predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], shap_values[1], pd.DataFrame([feature_final], columns=feature_names), matplotlib=True)
    elif predicted_class == 2:
        shap.force_plot(explainer.expected_value[2], shap_values[2], pd.DataFrame([feature_final], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer.expected_value[3], shap_values[3], pd.DataFrame([feature_final], columns=feature_names), matplotlib=True)
    pyplot.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")