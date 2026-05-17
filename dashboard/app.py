import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_transaction

st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")
st.title("🔍 Sistema de Detecção de Fraudes")
st.caption("Machine Learning aplicado a transações financeiras")

@st.cache_data
def load_data():
    return pd.read_csv("data/transactions.csv")

df = load_data()

col1, col2, col3, col4 = st.columns(4)
total = len(df)
fraudes = df["is_fraud"].sum()
col1.metric("💳 Total de Transações", f"{total:,}")
col2.metric("🚨 Fraudes Detectadas", f"{fraudes:,}")
col3.metric("✅ Transações Normais", f"{total - fraudes:,}")
col4.metric("📊 Taxa de Fraude", f"{fraudes/total*100:.2f}%")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Valor das Transações")
    fig = px.histogram(df, x="amount", color="is_fraud",
                      color_discrete_map={0: "#00d4ff", 1: "#ff4444"},
                      labels={"is_fraud": "Fraude", "amount": "Valor (R$)"},
                      nbins=50, barmode="overlay", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🕐 Horário das Transações")
    fig = px.histogram(df, x="hour", color="is_fraud",
                      color_discrete_map={0: "#00d4ff", 1: "#ff4444"},
                      labels={"is_fraud": "Fraude", "hour": "Hora do Dia"},
                      nbins=24, barmode="overlay", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("🧪 Simular Transação em Tempo Real")
st.caption("Preencha os dados abaixo para verificar se uma transação é fraudulenta")

col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("💵 Valor da Transação (R$)", min_value=1.0, value=500.0, step=10.0)
    hour = st.slider("🕐 Hora da Transação", 0, 23, 14)
with col2:
    distance = st.number_input("📍 Distância de Casa (km)", min_value=0.0, value=10.0, step=1.0)
    transactions_24h = st.number_input("🔄 Transações nas últimas 24h", min_value=0, value=1, step=1)
with col3:
    different_location = st.selectbox("📌 Local diferente do habitual?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

if st.button("🔍 Analisar Transação", type="primary"):
    result = predict_transaction(amount, hour, distance, transactions_24h, different_location)

    if result["is_fraud"]:
        st.error(f"🚨 FRAUDE DETECTADA! Probabilidade: {result['fraud_probability']}% | Risco: {result['risk_level']}")
    else:
        st.success(f"✅ Transação Normal! Probabilidade de fraude: {result['fraud_probability']}% | Risco: {result['risk_level']}")