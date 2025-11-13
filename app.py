from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime
from pymongo.errors import PyMongoError

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banking-api")

app = Flask(__name__)
CORS(app, origins=["http://18.190.157.12"])

# Cargar modelo (con error handling)
try:
    model = joblib.load("modelo_banking.pkl")
    logger.info("Modelo cargado OK")
except Exception as e:
    logger.error(f"Error modelo: {e}")
    model = None

# Conexión MongoDB (con error handling)
try:
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        raise ValueError("Falta MONGODB_URI")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.server_info()
    db = client.banking_predictions
    collection = db.predictions
    logger.info("MongoDB conectado OK")
except Exception as e:
    logger.error(f"Error MongoDB: {e}")
    client = None

@app.route('/')
def home():
    return f"API viva! Modelo: {'OK' if model else 'FAIL'}. MongoDB: {'OK' if client else 'FAIL'}.<br>Error Mongo: {getattr(client, 'error_mongo', 'No hay error')}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not model:
            return jsonify({"error": "Modelo no cargado"}), 500

        # DataFrame con TODOS tus campos del HTML
        input_df = pd.DataFrame([{
            'age': data.get('age'),
            'balance': data.get('balance'),
            'duration': data.get('duration'),
            'campaign': data.get('campaign'),
            'job': data.get('job'),
            'marital': data.get('marital'),
            'education': data.get('education'),
            'pdays': data.get('pdays'),
            'loan': data.get('loan'),
            'month': data.get('month'),
            'poutcome': data.get('poutcome'),
            'housing': data.get('housing'),
            'default': data.get('default'),
            'previous': data.get('previous'),
            'contact': data.get('contact'),
            'day': data.get('day')
        }])

        # Predicción
        prob = model.predict_proba(input_df)[0][1]
        pred = 1 if prob >= 0.5 else 0
        resultado = "Cliente aceptará el producto" if pred == 1 else "Cliente no aceptará"

        # Guardar en MongoDB (si funciona)
        if client:
            try:
                document = {**data, "prediction": pred, "score_probabilidad": round(prob, 4), "resultado": resultado, "timestamp": datetime.utcnow().isoformat()}
                collection.insert_one(document)
                print("Guardado en MongoDB OK")
            except PyMongoError as mongo_e:
                print(f"Error guardado MongoDB: {mongo_e}")
            except Exception as mongo_e:
                print(f"Error inesperado guardado Mongo: {mongo_e}")

        return jsonify({
            "prediccion": pred,
            "score_probabilidad": round(prob, 4),
            "resultado": resultado
        })

    except Exception as e:
        logger.error(f"Error MongoDB: {e}")
        client = None
        # AGREGAR:
        setattr(client, 'error_mongo', str(e))

if __name__ == '__main__':
    app.run()