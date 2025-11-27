from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from datetime import datetime
from pymongo.errors import PyMongoError

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    load_model = None

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Banking-api")

app = Flask(__name__)
CORS(app, origins=["http://18.190.157.12/Graficas.html"])

# Cargar modelo SVM (con error handling)
try:
    model_svm = joblib.load("modelo_banking.pkl")
    logger.info("✓ Modelo SVM cargado OK")
except Exception as e:
    logger.error(f"Error modelo SVM: {e}")
    model_svm = None

# Cargar modelo DL (con error handling)
model_dl = None
preprocessor_dl = None
try:
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow no está disponible")
    model_dl = load_model("modelo_dl_banking.h5")
    preprocessor_dl = joblib.load("preprocessor_dl.pkl")
    logger.info("✓ Modelo DL cargado OK")
except Exception as e:
    logger.error(f"Error modelo DL: {e}")
    model_dl = None
    preprocessor_dl = None

# Conexión MongoDB (con error handling)
client = None
collection = None
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
    status = {
        "status": "API Running",
        "modelo_svm": "✓ OK" if model_svm else "✗ FAIL",
        "modelo_dl": "✓ OK" if model_dl else "✗ FAIL",
        "mongodb": "✓ OK" if client else "✗ FAIL",
        "endpoints": [
            "/predict (POST) - Predicción con SVM",
            "/predict_dl (POST) - Predicción con Deep Learning",
            "/predict_both (POST) - Predicción con ambos modelos",
            "/graficas (GET) - Lista de todas las gráficas",
            "/static/plots/<filename> (GET) - Ver gráfica específica"
        ]
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    """Predicción usando modelo SVM"""
    try:
        data = request.get_json()
        if not model_svm:
            return jsonify({"error": "Modelo SVM no cargado"}), 500

        # DataFrame con TODOS los campos
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

        # Predicción SVM
        prob = model_svm.predict_proba(input_df)[0][1]
        pred = 1 if prob >= 0.5 else 0
        resultado = "✅ Cliente aceptará el producto" if pred == 1 else "❌ Cliente no aceptará"

        # Guardar en MongoDB (si funciona)
        if client:
            try:
                document = {
                    **data, 
                    "prediction_svm": pred, 
                    "score_svm": round(prob, 4), 
                    "resultado_svm": resultado, 
                    "timestamp": datetime.utcnow().isoformat(),
                    "modelo": "SVM"
                }
                collection.insert_one(document)
                logger.info("Guardado en MongoDB OK")
            except Exception as mongo_e:
                logger.error(f"Error guardado MongoDB: {mongo_e}")

        return jsonify({
            "modelo": "SVM",
            "prediccion": pred,
            "score_probabilidad": round(prob, 4),
            "resultado": resultado,
            "graficas": {
                "confusion_matrix": "/static/plots/svm_confusion.png",
                "roc_curve": "/static/plots/svm_roc.png"
            }
        })

    except Exception as e:
        logger.error(f"Error predict_svm: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_dl', methods=['POST'])
def predict_dl():
    """Predicción usando modelo Deep Learning"""
    try:
        data = request.get_json()
        if not model_dl or not preprocessor_dl:
            return jsonify({"error": "Modelo DL no cargado"}), 500

        # DataFrame con TODOS los campos
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

        # Preprocesar datos
        X_proc = preprocessor_dl.transform(input_df)
        
        # Predicción DL
        prob = float(model_dl.predict(X_proc, verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0
        resultado = "✅ Cliente aceptará el producto" if pred == 1 else "❌ Cliente no aceptará"

        # Guardar en MongoDB (si funciona)
        if client:
            try:
                document = {
                    **data, 
                    "prediction_dl": pred, 
                    "score_dl": round(prob, 4), 
                    "resultado_dl": resultado, 
                    "timestamp": datetime.utcnow().isoformat(),
                    "modelo": "Deep Learning"
                }
                collection.insert_one(document)
                logger.info("Guardado en MongoDB OK")
            except Exception as mongo_e:
                logger.error(f"Error guardado MongoDB: {mongo_e}")

        return jsonify({
            "modelo": "Deep Learning",
            "prediccion": pred,
            "score_probabilidad": round(prob, 4),
            "resultado": resultado,
            "graficas": {
                "confusion_matrix": "/static/plots/dl_confusion.png",
                "roc_curve": "/static/plots/dl_roc.png",
                "training_loss": "/static/plots/dl_loss.png",
                "training_accuracy": "/static/plots/dl_accuracy.png"
            }
        })

    except Exception as e:
        logger.error(f"Error predict_dl: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_both', methods=['POST'])
def predict_both():
    """Predicción usando ambos modelos (SVM + DL)"""
    try:
        data = request.get_json()
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_data": data
        }

        # Predicción SVM
        if model_svm:
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
            
            prob_svm = model_svm.predict_proba(input_df)[0][1]
            pred_svm = 1 if prob_svm >= 0.5 else 0
            
            results["svm"] = {
                "prediccion": pred_svm,
                "score_probabilidad": round(prob_svm, 4),
                "resultado": "✅ Cliente aceptará" if pred_svm == 1 else "❌ Cliente no aceptará",
                "graficas": {
                    "confusion_matrix": "/static/plots/svm_confusion.png",
                    "roc_curve": "/static/plots/svm_roc.png"
                }
            }
        else:
            results["svm"] = {"error": "Modelo no disponible"}

        # Predicción DL
        if model_dl and preprocessor_dl:
            X_proc = preprocessor_dl.transform(input_df)
            prob_dl = float(model_dl.predict(X_proc, verbose=0)[0][0])
            pred_dl = 1 if prob_dl >= 0.5 else 0
            
            results["deep_learning"] = {
                "prediccion": pred_dl,
                "score_probabilidad": round(prob_dl, 4),
                "resultado": "✅ Cliente aceptará" if pred_dl == 1 else "❌ Cliente no aceptará",
                "graficas": {
                    "confusion_matrix": "/static/plots/dl_confusion.png",
                    "roc_curve": "/static/plots/dl_roc.png",
                    "training_loss": "/static/plots/dl_loss.png",
                    "training_accuracy": "/static/plots/dl_accuracy.png"
                }
            }
        else:
            results["deep_learning"] = {"error": "Modelo no disponible"}

        # Predicción de consenso (promedio)
        if model_svm and model_dl and preprocessor_dl:
            prob_avg = (prob_svm + prob_dl) / 2
            pred_avg = 1 if prob_avg >= 0.5 else 0
            results["ensemble"] = {
                "prediccion": pred_avg,
                "score_probabilidad": round(prob_avg, 4),
                "resultado": "✅ Cliente aceptará" if pred_avg == 1 else "❌ Cliente no aceptará",
                "metodo": "Promedio de probabilidades"
            }

        # Guardar en MongoDB
        if client:
            try:
                collection.insert_one({**results, "_saved_at": datetime.utcnow()})
                logger.info("Predicción completa guardada en MongoDB")
            except Exception as mongo_e:
                logger.error(f"Error guardado MongoDB: {mongo_e}")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error predict_both: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/graficas', methods=['GET'])
def get_graficas():
    """Retorna lista de todas las gráficas disponibles"""
    graficas = {
        "svm": {
            "confusion_matrix": "/static/plots/svm_confusion.png",
            "roc_curve": "/static/plots/svm_roc.png"
        },
        "deep_learning": {
            "confusion_matrix": "/static/plots/dl_confusion.png",
            "roc_curve": "/static/plots/dl_roc.png",
            "training_loss": "/static/plots/dl_loss.png",
            "training_accuracy": "/static/plots/dl_accuracy.png"
        },
        "nota": "Accede a las imágenes directamente con la URL completa"
    }
    return jsonify(graficas)


@app.route('/static/plots/<filename>', methods=['GET'])
def serve_plot(filename):
    """Servir archivos de gráficas"""
    try:
        return send_from_directory('static/plots', filename)
    except Exception as e:
        logger.error(f"Error sirviendo gráfica {filename}: {e}")
        return jsonify({"error": "Gráfica no encontrada"}), 404
    

if __name__ == '__main__':
    app.run()