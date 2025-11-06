import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🩺 Predicciones Médicas")
st.write("Modelos de costos médicos y diagnóstico de diabetes")

option = st.sidebar.selectbox("Selecciona modelo:", ["Seguro Médico", "Diabetes"])

if option == "Seguro Médico":
    st.header("Predicción de costos de seguro médico")
    
    # Inputs básicos
    age = st.number_input("Edad", 0, 100, 30)
    bmi = st.number_input("Índice de masa corporal (BMI)", 0.0, 60.0, 25.0)
    children = st.number_input("Número de hijos", 0, 5, 0)
    smoker = st.selectbox("¿Fuma?", ["No", "Sí"])
    sex = st.selectbox("Sexo", ["Femenino", "Masculino"])
    region = st.selectbox("Región", ["northwest", "northeast", "southwest", "southeast"])

    # Crear array con el orden correcto que espera el modelo
    data = [
        age,
        bmi,
        children,
        1 if sex == "Masculino" else 0,  # sex_male
        1 if smoker == "Sí" else 0,      # smoker_yes
        1 if region == "northwest" else 0,
        1 if region == "northeast" else 0,
        1 if region == "southeast" else 0
        # region_southwest se omite (es la categoría base)
    ]
    
    model = joblib.load('modulos/seguro.pkl')
    pred = model.predict([data])[0]
    st.success(f"Costo estimado del seguro: ${pred:,.2f}")

else:
    st.header("Predicción de diabetes")
    preg = st.number_input("Número de embarazos", 0, 20, 1)
    glucose = st.number_input("Nivel de glucosa", 0, 200, 100)
    bmi = st.number_input("Índice de masa corporal", 0.0, 60.0, 25.0)
    age = st.number_input("Edad", 0, 120, 30)

    model = joblib.load('modulos/diabetes.pkl')
    data = np.array([[preg, glucose, 0, 0, 0, bmi, 0, age]])
    pred_prob = model.predict_proba(data)[0, 1]
    st.info(f"Probabilidad de diabetes: {pred_prob:.2f}")

    if pred_prob > 0.4:
        st.error("Alta probabilidad de tener diabetes")
    else:
        st.success("Baja probabilidad de tener diabetes")