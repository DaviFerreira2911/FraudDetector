import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

print("📂 Carregando dados...")
df = pd.read_csv("data/transactions.csv")

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚖️ Balanceando classes com SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("🤖 Treinando modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_bal, y_train_bal)

print("\n📊 RESULTADOS:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")
print("✅ Modelo salvo em models/fraud_model.pkl")