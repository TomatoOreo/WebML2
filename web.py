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
# 人口学指标 16
demographic_indicators = ['BMI','Sex','Age','GRACE risk score','TIMI risk score','History of smoking','No quitting smoking',
 'History of hypertension','History of diabetes','History of hyperlipemia','History of CVD','History of PU','History of myocardial infarction',
 'History of stent implantation','History of CABG','History of AF']
# Streamlit user interface
st.title("KILLIP Predictor")

col1, col2 = st.columns(2)
input_values = {}  
feature_values = []
with col1:
    st.subheader("demographic indicators")
    col3, col4 = st.columns(2)
    for i, demographic_indicator in enumerate(demographic_indicators):
        if i % 2 == 0:
            with col4:
                input_value = st.number_input(f"{demographic_indicator}:")
                feature_values.append(input_value)
                input_values[demographic_indicator] = input_value 
        else:
            with col3:
                input_value = st.number_input(f"{demographic_indicator}:")
                feature_values.append(input_value)
                input_values[demographic_indicator] = input_value 
with col2:
    st.subheader("blood indicators")
    col5, col6 = st.columns(2)
    for i, blood_indicator in enumerate(blood_indicators):
        if i % 2 == 0:
            with col6:
                input_value = st.number_input(f"{blood_indicator}:")
                feature_values.append(input_value)
                input_values[blood_indicator] = input_value 
        else:
            with col5:
                input_value = st.number_input(f"{blood_indicator}:")
                feature_values.append(input_value)
                input_values[blood_indicator] = input_value 
col7, col8 = st.columns([0.75, 0.25])
with col7:
    st.subheader("echocardiographic indicators")
    col9, col10, col11 = st.columns(3)
    with col9:
        input_value = st.number_input(f"{echocardiographic_indicators[0]}:")
        feature_values.append(input_value)
        input_values[echocardiographic_indicators[0]] = input_value 
    with col10:
        input_value = st.number_input(f"{echocardiographic_indicators[1]}:")
        feature_values.append(input_value)
        input_values[echocardiographic_indicators[1]] = input_value
    with col11:
        input_value = st.number_input(f"{echocardiographic_indicators[2]}:")
        feature_values.append(input_value) 
        input_values[echocardiographic_indicators[2]] = input_value 
with col8:
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