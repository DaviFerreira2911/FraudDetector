# 🔍 Fraud Detector — Sistema de Detecção de Fraudes

Machine Learning aplicado à detecção de transações financeiras fraudulentas.

## 🎯 Objetivo

Identificar automaticamente transações suspeitas em tempo real usando Random Forest,
com precisão de 100% nos testes realizados sobre 100.000 transações simuladas.

## 🤖 Como funciona

O modelo analisa 5 variáveis por transação: valor, horário, distância de casa,
frequência de transações e localização. Com base nesses padrões, classifica cada
transação como normal ou fraudulenta e calcula a probabilidade de fraude.

## 🛠️ Tecnologias

Python, Scikit-learn, Pandas, NumPy, Streamlit, Plotly, SMOTE (imbalanced-learn)

## 📊 Resultados do Modelo

Precision: 1.00 | Recall: 1.00 | F1-Score: 1.00

Treinado com 100.000 transações, sendo 2.000 fraudes (2% do dataset).
Técnica SMOTE utilizada para balanceamento das classes.

## 📦 Como executar

    git clone https://github.com/DaviFerreira2911/FraudDetector.git
    cd FraudDetector
    pip install -r requirements.txt
    py src/generate_data.py
    py src/train_model.py
    py -m streamlit run dashboard/app.py

## 🗂️ Estrutura

    FraudDetector/
    ├── src/            # Geração de dados, treino e predição
    ├── dashboard/      # Dashboard Streamlit
    ├── data/           # Dataset de transações
    ├── models/         # Modelo treinado
    └── requirements.txt