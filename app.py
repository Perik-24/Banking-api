from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["http://18.222.131.92", "*"])  # ← Permite tu AWS

# Cargar modelo
model = joblib.load("modelo_banking.pkl")

# === CONEXIÓN A MONGODB ATLAS ===
uri = os.environ.get('MONGODB_URI')
if not uri:
    raise ValueError("Falta MONGODB_URI en Azure")
client = MongoClient(uri)
db = client.banking_predictions
collection = db.predictions
# ===================================

@app.route('/')
def home():
    return "API Banking + MongoDB Atlas"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # === DATAFRAME PARA EL MODELO ===
        input_df = pd.DataFrame([{
            'age': data.get('age'),
            'job': data.get('job'),
            'marital': data.get('marital'),
            'education': data.get('education', 'unknown'),
            'default': data.get('default', 'no'),
            'balance': data.get('balance'),
            'housing': data.get('housing', 'yes'),
            'loan': data.get('loan', 'no'),
            'contact': data.get('contact', 'cellular'),
            'day': data.get('day'),
            'month': data.get('month'),
            'duration': data.get('duration'),
            'campaign': data.get('campaign'),
            'poutcome': data.get('poutcome', 'unknown')
        }])

        # === PREDICCIÓN ===
        prob = model.predict_proba(input_df)[0][1]
        pred = 1 if prob >= 0.5 else 0
        resultado = "Cliente aceptará" if pred == 1 else "Cliente no aceptará"

        # === GUARDAR EN MONGODB ===
        document = {
            **data,
            "prediction": pred,
            "probability": round(prob, 4),
            "resultado": resultado,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "AWS Frontend"
        }
        collection.insert_one(document)
        # ===============================

        return jsonify({
            "prediccion": pred,
            "score_probabilidad": round(prob, 4),
            "resultado": resultado
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '_main_':
    app.run()