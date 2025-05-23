import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les données
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Prétraitement
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraînement du modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Sauvegarde
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Modèle et scaler sauvegardés avec succès.")
