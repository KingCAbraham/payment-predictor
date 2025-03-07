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

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Pago",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo CSS solo para el resultado
st.markdown("""
    <style>
    /* Estilo para el resultado */
    .resultado {
        padding: 1em;
        border-radius: 8px;
        text-align: center;
        margin-top: 1.5em;
        font-size: 1.1em;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo y escalador
try:
    modelo = joblib.load('modelo_pago.pkl')
    escalador = joblib.load('escalador_pago.pkl')
except FileNotFoundError:
    st.error("Error: No se encontraron 'modelo_pago.pkl' o 'escalador_pago.pkl'. Ejecuta main.py primero.")
    st.stop()

columnas_caracteristicas = ['EDAD_CLIENTE', 'DIAS_ATRASO', 'MORATORIOS', 'SALDO_TOTAL',
                            'IMP_ULTIMO_PAGO', 'MONTO_PAGOS']

st.title("Predictor de Probabilidad de Pago")
st.write("Ingresa los datos del cliente para predecir su probabilidad de pago.")

# Formulario
with st.form(key='predict_form'):
    st.subheader("Datos del Cliente")
    edad = st.number_input("Edad del cliente", min_value=0, max_value=120, value=30)
    dias_atraso = st.number_input("Días de atraso", min_value=0, value=0)
    moratorios = st.number_input("Moratorios", min_value=0.0, value=0.0, step=10.0)
    saldo_total = st.number_input("Saldo total", min_value=0.0, value=10000.0, step=100.0)
    imp_ultimo_pago = st.number_input("Importe último pago", min_value=0.0, value=0.0, step=10.0)
    monto_pagos = st.number_input("Monto total de pagos", min_value=0.0, value=0.0, step=100.0)

    submit_button = st.form_submit_button(label="Calcular Probabilidad")

# Procesar y mostrar resultado
if submit_button:
    datos_cliente = {
        'EDAD_CLIENTE': edad,
        'DIAS_ATRASO': dias_atraso,
        'MORATORIOS': moratorios,
        'SALDO_TOTAL': saldo_total,
        'IMP_ULTIMO_PAGO': imp_ultimo_pago,
        'MONTO_PAGOS': monto_pagos
    }
    
    cliente_df = pd.DataFrame([datos_cliente], columns=columnas_caracteristicas)
    cliente_escalado = escalador.transform(cliente_df)
    probabilidad = modelo.predict_proba(cliente_escalado)[0][1] * 100
    
    # Estilo dinámico para el resultado
    if probabilidad > 70:
        color_fondo = "#DCFCE7"
        color_texto = "#166534"
        categoria = "Alto"
    elif probabilidad > 30:
        color_fondo = "#FEF9C3"
        color_texto = "#854D0E"
        categoria = "Medio"
    else:
        color_fondo = "#FEE2E2"
        color_texto = "#991B1B"
        categoria = "Bajo"

    st.markdown(
        f'<div class="resultado" style="background-color: {color_fondo}; color: {color_texto};">'
        f'Probabilidad de pago: <b>{probabilidad:.2f}%</b><br>Categoría: <b>{categoria}</b></div>',
        unsafe_allow_html=True
    )