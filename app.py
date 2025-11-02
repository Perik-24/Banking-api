from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Cargar modelo entrenado
model = joblib.load("modelo_banking.pkl")

# === CONEXIÓN A MONGODB ATLAS ===
uri = os.environ.get('MONGODB_URI')
if not uri:
    raise ValueError("Falta MONGODB_URI en variables de entorno")

client = MongoClient(uri)
db = client.banking_predictions
collection = db.predictions
# ===================================


@app.route('/')
def home():
    return "✅ API del modelo Banking corriendo localmente"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Crear DataFrame con una sola fila (el modelo espera columnas con nombres)
    input_df = pd.DataFrame([data])

    # Hacer predicción
    probabilities = model.predict_proba(input_df)[0]

    # Probabilidad de 'Sí aceptará' (la clase 1)
    probability_of_yes = probabilities[1] 

    # Asumimos que la predicción 'final' es 1 si la probabilidad de sí es >= 0.5, sino 0.
    prediction = 1 if probability_of_yes >= 0.5 else 0 

    resultado = "✅ Cliente aceptará el producto" if prediction == 1 else "❌ Cliente no aceptará"

    return jsonify({
        "prediccion": int(prediction),
        "score_probabilidad": float(probability_of_yes), # Nuevo campo con el score
        "resultado": resultado
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
