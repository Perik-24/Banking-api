from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from datetime import datetime
from pymongo.errors import PyMongoError
import threading

# Reducir logs ruidosos de TensorFlow antes de que se importe
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 0 = all, 1 = filter INFO, 2 = filter INFO & WARNING, 3 = filter ERROR
# Si quieres desactivar oneDNN (mensajes o resultados numéricos diferentes), descomenta:
# os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

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

# Variables para modelo DL (lazy load)
model_dl = None
preprocessor_dl = None
_dl_lock = threading.Lock()
_tensorflow_imported = False

def load_dl_model():
    """Carga TensorFlow y el modelo DL de forma perezosa y thread-safe."""
    global model_dl, preprocessor_dl, _tensorflow_imported
    if model_dl is not None and preprocessor_dl is not None:
        return True
    with _dl_lock:
        # doble-check después del lock
        if model_dl is not None and preprocessor_dl is not None:
            return True
        try:
            # Importar tensorflow y funciones dentro de la función para evitar costo al inicio
            from tensorflow.keras.models import load_model as tf_load_model
            _tensorflow_imported = True
        except Exception as e:
            logger.error(f"TensorFlow no disponible o error import: {e}")
            _tensorflow_imported = False
            return False

        try:
            # Cargar artefactos del disco
            model = tf_load_model("modelo_dl_banking.h5")
            prep = joblib.load("preprocessor_dl.pkl")
            # asignar a variables globales
            globals()['model_dl'] = model
            globals()['preprocessor_dl'] = prep
            logger.info("✓ Modelo DL cargado OK (lazy-loaded)")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo DL o preprocessor: {e}")
            globals()['model_dl'] = None
            globals()['preprocessor_dl'] = None
            return False

# No intentamos cargar TensorFlow aquí — lo haremos on-demand en load_dl_model()

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
        "modelo_dl": "✓ OK" if model_dl else "✗ NOT LOADED (use /predict_dl to attempt lazy load)",
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

# (El resto de endpoints se mantiene igual salvo que en /predict_dl y en /predict_both
#  llamamos a load_dl_model() antes de usar model_dl)

@app.route('/predict', methods=['POST'])
@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    """Predicción usando modelo SVM"""
    try:
        data = request.get_json()
        if not model_svm:
            return jsonify({"error": "Modelo SVM no cargado"}), 500

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

        prob = model_svm.predict_proba(input_df)[0][1]
        pred = 1 if prob >= 0.5 else 0
        resultado = "✅ Cliente aceptará el producto" if pred == 1 else "❌ Cliente no aceptará"

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
    """Predicción usando modelo Deep Learning (lazy load)"""
    try:
        data = request.get_json()
        # Intentar cargar el modelo DL si aún no está cargado
        ok = load_dl_model()
        if not ok or not globals().get('model_dl') or not globals().get('preprocessor_dl'):
            return jsonify({"error": "Modelo DL no cargado: revisa logs o instala tensorflow-cpu en requirements"}), 500

        model = globals()['model_dl']
        preprocessor = globals()['preprocessor_dl']

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

        X_proc = preprocessor.transform(input_df)

        prob = float(model.predict(X_proc, verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0
        resultado = "✅ Cliente aceptará el producto" if pred == 1 else "❌ Cliente no aceptará"

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

        # Intentar cargar DL sólo si lo necesitamos
        dl_ok = load_dl_model()
        if dl_ok and globals().get('model_dl') and globals().get('preprocessor_dl'):
            model = globals()['model_dl']
            preprocessor = globals()['preprocessor_dl']
            X_proc = preprocessor.transform(input_df)
            prob_dl = float(model.predict(X_proc, verbose=0)[0][0])
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

        if model_svm and globals().get('model_dl') and globals().get('preprocessor_dl'):
            prob_avg = (prob_svm + prob_dl) / 2
            pred_avg = 1 if prob_avg >= 0.5 else 0
            results["ensemble"] = {
                "prediccion": pred_avg,
                "score_probabilidad": round(prob_avg, 4),
                "resultado": "✅ Cliente aceptará" if pred_avg == 1 else "❌ Cliente no aceptará",
                "metodo": "Promedio de probabilidades"
            }

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
    try:
        return send_from_directory('static/plots', filename)
    except Exception as e:
        logger.error(f"Error sirviendo gráfica {filename}: {e}")
        return jsonify({"error": "Gráfica no encontrada"}), 404

if __name__ == '__main__':
    app.run()