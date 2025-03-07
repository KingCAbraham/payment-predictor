"""
Interfaz Web para Predicción de Probabilidad de Pago
--------------------------------------------------
Esta aplicación carga un modelo de Random Forest preentrenado y permite predecir la probabilidad de pago
de un cliente ingresando sus datos manualmente.

Requisitos: streamlit, pandas, joblib
Ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y escalador
try:
    modelo = joblib.load('modelo_pago.pkl')
    escalador = joblib.load('escalador_pago.pkl')
except FileNotFoundError:
    st.error("Error: No se encontraron 'modelo_pago.pkl' o 'escalador_pago.pkl'. Ejecuta main.py primero.")
    st.stop()

# Definir columnas esperadas por el modelo
columnas_caracteristicas = ['EDAD_CLIENTE', 'SALDO', 'DIAS_ATRASO', 'MORATORIOS', 'SALDO_TOTAL',
                            'IMP_ULTIMO_PAGO', 'MONTO_PAGOS', 'ATRASO_MAXIMO', 'SEMANAS_ATRASO']

# Título y descripción
st.title("Predictor de Probabilidad de Pago")
st.write("Ingresa los datos del cliente para predecir la probabilidad de que pague su deuda.")

# Formulario para ingresar datos
with st.form(key='predict_form'):
    st.subheader("Datos del Cliente")
    edad = st.number_input("Edad del cliente", min_value=0, max_value=120, value=30)
    saldo = st.number_input("Saldo", min_value=0.0, value=10000.0, step=100.0)
    dias_atraso = st.number_input("Días de atraso", min_value=0, value=0)
    moratorios = st.number_input("Moratorios", min_value=0.0, value=0.0, step=10.0)
    saldo_total = st.number_input("Saldo total", min_value=0.0, value=10000.0, step=100.0)
    imp_ultimo_pago = st.number_input("Importe último pago", min_value=0.0, value=0.0, step=10.0)
    monto_pagos = st.number_input("Monto total de pagos", min_value=0.0, value=0.0, step=100.0)
    atraso_maximo = st.number_input("Atraso máximo (días)", min_value=0, value=0)
    semanas_atraso = st.number_input("Semanas de atraso", min_value=0, value=0)

    # Botón para predecir
    submit_button = st.form_submit_button(label="Calcular Probabilidad")

# Procesar la predicción
if submit_button:
    datos_cliente = {
        'EDAD_CLIENTE': edad,
        'SALDO': saldo,
        'DIAS_ATRASO': dias_atraso,
        'MORATORIOS': moratorios,
        'SALDO_TOTAL': saldo_total,
        'IMP_ULTIMO_PAGO': imp_ultimo_pago,
        'MONTO_PAGOS': monto_pagos,
        'ATRASO_MAXIMO': atraso_maximo,
        'SEMANAS_ATRASO': semanas_atraso
    }
    
    # Convertir a DataFrame y escalar
    cliente_df = pd.DataFrame([datos_cliente], columns=columnas_caracteristicas)
    cliente_escalado = escalador.transform(cliente_df)
    
    # Predecir probabilidad
    probabilidad = modelo.predict_proba(cliente_escalado)[0][1] * 100
    
    # Mostrar resultado
    st.success(f"Probabilidad de que el cliente pague: **{probabilidad:.2f}%**")
    if probabilidad > 70:
        st.write("Categoría: **Alto**")
    elif probabilidad > 30:
        st.write("Categoría: **Medio**")
    else:
        st.write("Categoría: **Bajo**")