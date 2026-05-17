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

st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0a0a0f;
        color: #e8e8f0;
    }

    .stApp { background-color: #0a0a0f; }

    .header-container {
        background: linear-gradient(135deg, #0d0d1a 0%, #0a0a0f 100%);
        border: 1px solid #1a1a2e;
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }

    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(255, 60, 60, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(60, 60, 255, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }

    .header-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #ff3c3c, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
    }

    .header-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #666680;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 8px;
    }

    .kpi-card {
        background: #0d0d1a;
        border: 1px solid #1a1a2e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: border-color 0.3s;
    }

    .kpi-card:hover { border-color: #ff3c3c44; }

    .kpi-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #ff3c3c;
        line-height: 1;
    }

    .kpi-label {
        font-size: 0.75rem;
        color: #666680;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 8px;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #666680;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1a1a2e;
    }

    .simulator-card {
        background: #0d0d1a;
        border: 1px solid #1a1a2e;
        border-radius: 16px;
        padding: 32px;
        margin-top: 32px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #ff3c3c, #ff6b35) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        padding: 12px 32px !important;
        width: 100% !important;
        margin-top: 16px !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover { opacity: 0.85 !important; }

    .result-fraud {
        background: linear-gradient(135deg, #1a0505, #200808);
        border: 1px solid #ff3c3c55;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        text-align: center;
    }

    .result-safe {
        background: linear-gradient(135deg, #051a0a, #082010);
        border: 1px solid #3cff6455;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        text-align: center;
    }

    .result-title-fraud {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #ff3c3c;
    }

    .result-title-safe {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #3cff64;
    }

    .result-prob {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #888899;
        margin-top: 8px;
    }

    div[data-testid="stMetric"] { display: none; }

    .stSlider > div > div { background: #1a1a2e !important; }
    .stSelectbox > div > div { background: #0d0d1a !important; border-color: #1a1a2e !important; }
    .stNumberInput > div > div { background: #0d0d1a !important; border-color: #1a1a2e !important; }

    .divider { border: none; border-top: 1px solid #1a1a2e; margin: 32px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("data/transactions.csv")

df = load_data()
total = len(df)
fraudes = int(df["is_fraud"].sum())
normais = total - fraudes
taxa = fraudes / total * 100

# Header
st.markdown("""
<div class="header-container">
    <p class="header-sub">Sistema de Análise — v1.0</p>
    <h1 class="header-title">FRAUD<br>DETECTOR</h1>
    <p class="header-sub" style="margin-top:16px">Machine Learning · Detecção em Tempo Real · Random Forest</p>
</div>
""", unsafe_allow_html=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-label">Transações Analisadas</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#ff3c3c">{fraudes:,}</div>
        <div class="kpi-label">Fraudes Detectadas</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#3cff64">{normais:,}</div>
        <div class="kpi-label">Transações Normais</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#ff8c42">{taxa:.1f}%</div>
        <div class="kpi-label">Taxa de Fraude</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Gráficos
col1, col2 = st.columns(2)

with col1:
    st.markdown("<p class='section-title'>Distribuição por Valor</p>", unsafe_allow_html=True)
    fig = px.histogram(
        df, x="amount", color="is_fraud",
        color_discrete_map={0: "#1a3a4a", 1: "#ff3c3c"},
        nbins=60, barmode="overlay", opacity=0.85,
        labels={"amount": "Valor (R$)", "count": "Qtd", "is_fraud": ""}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666680", family="Space Mono"),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<p class='section-title'>Horário das Transações</p>", unsafe_allow_html=True)
    fig2 = px.histogram(
        df, x="hour", color="is_fraud",
        color_discrete_map={0: "#1a3a4a", 1: "#ff3c3c"},
        nbins=24, barmode="overlay", opacity=0.85,
        labels={"hour": "Hora do Dia", "count": "Qtd", "is_fraud": ""}
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666680", family="Space Mono"),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# Simulador
st.markdown("""
<div class="simulator-card">
    <p class="section-title">Simulador em Tempo Real</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Valor da Transação (R$)", min_value=1.0, value=500.0, step=50.0)
    hour = st.slider("Hora da Transação", 0, 23, 14)
with col2:
    distance = st.number_input("Distância de Casa (km)", min_value=0.0, value=10.0, step=5.0)
    transactions_24h = st.number_input("Transações nas últimas 24h", min_value=0, value=1, step=1)
with col3:
    different_location = st.selectbox("Local diferente do habitual?", [0, 1],
                                      format_func=lambda x: "Sim" if x == 1 else "Não")
    st.markdown("<br>", unsafe_allow_html=True)
    analisar = st.button("ANALISAR TRANSAÇÃO")

if analisar:
    result = predict_transaction(amount, hour, distance, transactions_24h, different_location)
    if result["is_fraud"]:
        st.markdown(f"""
        <div class="result-fraud">
            <div class="result-title-fraud">⚠ FRAUDE DETECTADA</div>
            <div class="result-prob">Probabilidade: {result['fraud_probability']}% · Risco: {result['risk_level']}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-title-safe">✓ TRANSAÇÃO NORMAL</div>
            <div class="result-prob">Probabilidade de fraude: {result['fraud_probability']}% · Risco: {result['risk_level']}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)