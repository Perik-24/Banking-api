from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # ← ESTO ES CLAVE
import joblib
import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["http://18.222.131.92", "*"])  # ← PERMITE TU AWS + cualquier origen

# Cargar modelo
model = joblib.load("modelo_banking.pkl")

# === CONEXIÓN A MONGODB ATLAS ===
uri = os.environ.get('MONGODB_URI')
if not uri:
    raise ValueError("Falta MONGODB_URI")
client = MongoClient(uri)
db = client.banking_predictions
collection = db.predictions
# ===================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # === CREAR DATAFRAME PARA EL MODELO ===
        input_df = pd.DataFrame([{
            'age': data.get('age'),
            'balance': data.get('balance'),
            'duration': data.get('duration'),
            'campaign': data.get('campaign'),
            'job': data.get('job'),
            'marital': data.get('marital'),
            'education': data.get('education', 'unknown'),
            'default': data.get('default', 'no'),
            'housing': data.get('housing', 'yes'),
            'loan': data.get('loan', 'no'),
            'contact': data.get('contact', 'cellular'),
            'day': data.get('day'),
            'month': data.get('month'),
            'poutcome': data.get('poutcome', 'unknown')
        }])

        # === PREDICCIÓN ===
        probabilities = model.predict_proba(input_df)[0]
        probability_of_yes = probabilities[1]
        prediction = 1 if probability_of_yes >= 0.5 else 0
        resultado = "Cliente aceptará el producto" if prediction == 1 else "Cliente no aceptará"

        # === GUARDAR EN MONGODB ===
        document = {
            **data,
            "prediction": prediction,
            "score_probabilidad": round(probability_of_yes, 4),
            "resultado": resultado,
            "timestamp": datetime.utcnow(),
            "client_ip": request.remote_addr
        }
        collection.insert_one(document)
        # ===============================

        return jsonify({
            "prediccion": int(prediction),
            "score_probabilidad": float(probability_of_yes),
            "resultado": resultado
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()