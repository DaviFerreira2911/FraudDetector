import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/fraud_model.pkl")
features = joblib.load("models/features.pkl")

def predict_transaction(amount, hour, distance, transactions_24h, different_location):
    data = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "distance_from_home": distance,
        "transactions_last_24h": transactions_24h,
        "different_location": different_location
    }], columns=features)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability) * 100, 2),
        "risk_level": "🔴 ALTO" if probability > 0.7 else ("🟡 MÉDIO" if probability > 0.3 else "🟢 BAIXO")
    }

if __name__ == "__main__":
    print("=== TESTANDO TRANSAÇÕES ===\n")

    transacoes = [
        {"amount": 5000, "hour": 3, "distance": 300, "t24h": 15, "diff_loc": 1},
        {"amount": 50,   "hour": 14, "distance": 5,   "t24h": 1,  "diff_loc": 0},
        {"amount": 1200, "hour": 1,  "distance": 150, "t24h": 8,  "diff_loc": 1},
    ]

    for i, t in enumerate(transacoes, 1):
        result = predict_transaction(t["amount"], t["hour"], t["distance"], t["t24h"], t["diff_loc"])
        print(f"Transação {i}: R$ {t['amount']}")
        print(f"  Fraude: {result['is_fraud']}")
        print(f"  Probabilidade: {result['fraud_probability']}%")
        print(f"  Risco: {result['risk_level']}\n")