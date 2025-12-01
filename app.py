from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import os
from datetime import datetime
from pymongo.errors import PyMongoError
import threading

# Reduce TF logs (no import of TF here)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Banking-api")

app = Flask(__name__)
# CORS: permitir solo el host (sin ruta)
CORS(app, resources={r"/*": {"origins": ["http://18.190.157.12"]}})

# Inicializar variables de modelos como None -> cargar perezosamente
model_svm = None
model_dl = None
preprocessor_dl = None

# Locks para carga thread-safe
_svm_lock = threading.Lock()
_dl_lock = threading.Lock()

def load_svm_model():
    """Carga el modelo SVM de forma perezosa y thread-safe."""
    global model_svm
    if model_svm is not None:
        return True
    with _svm_lock:
        if model_svm is not None:
            return True
        try:
            # importar joblib (y numpy/pandas cuando sea necesario) aquí, no en el toplevel
            import joblib
            model = joblib.load("modelo_banking.pkl")
            model_svm = model
            logger.info("✓ Modelo SVM cargado OK (lazy)")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo SVM (lazy): {e}")
            model_svm = None
            return False

def load_dl_model():
    """Carga TensorFlow y el modelo DL de forma perezosa y thread-safe."""
    global model_dl, preprocessor_dl
    if model_dl is not None and preprocessor_dl is not None:
        return True
    with _dl_lock:
        if model_dl is not None and preprocessor_dl is not None:
            return True
        try:
            # Importar tensorflow dentro de la función (evita costoso import en arranque)
            from tensorflow.keras.models import load_model as tf_load_model
        except Exception as e:
            logger.error(f"TensorFlow no disponible o error import: {e}")
            return False

        try:
            import joblib
            m = tf_load_model("modelo_dl_banking.h5")
            p = joblib.load("preprocessor_dl.pkl")
            globals()['model_dl'] = m
            globals()['preprocessor_dl'] = p
            logger.info("✓ Modelo DL cargado OK (lazy-loaded)")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo DL o preprocessor: {e}")
            globals()['model_dl'] = None
            globals()['preprocessor_dl'] = None
            return False

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
    collection = None

@app.route('/')
def home():
    status = {
        "status": "API Running",
        "modelo_svm": "✓ OK" if model_svm else "✗ NOT LOADED (use /predict_svm to lazy load)",
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

@app.route('/predict', methods=['POST'])
@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    """Predicción usando modelo SVM"""
    try:
        data = request.get_json()
        # Intentar cargar SVM si no está
        if not load_svm_model():
            return jsonify({"error": "Modelo SVM no cargado: revisa logs o reinstala numpy/scikit-learn en requirements"}), 500

        # Importar pandas y numpy aquí (evitar import temprano que pueda romper el arranque)
        import pandas as pd
        import numpy as np

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

        if client is not None and collection is not None:
            try:
                document = {
                    **data,
                    "prediction_svm": int(pred),
                    "score_svm": round(float(prob), 4),
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
            "prediccion": int(pred),
            "score_probabilidad": round(float(prob), 4),
            "resultado": resultado,
            "graficas": {
                "confusion_matrix": "/static/plots/svm_confusion.png",
                "roc_curve": "/static/plots/svm_roc.png"
            }
        })

    except Exception as e:
        logger.error(f"Error predict_svm: {e}", exc_info=True)
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

        # Importar pandas/numpy localmente
        import pandas as pd
        import numpy as np

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

        if client is not None and collection is not None:
            try:
                document = {
                    **data,
                    "prediction_dl": int(pred),
                    "score_dl": round(float(prob), 4),
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
            "prediccion": int(pred),
            "score_probabilidad": round(float(prob), 4),
            "resultado": resultado,
            "graficas": {
                "confusion_matrix": "/static/plots/dl_confusion.png",
                "roc_curve": "/static/plots/dl_roc.png",
                "training_loss": "/static/plots/dl_loss.png",
                "training_accuracy": "/static/plots/dl_accuracy.png"
            }
        })

    except Exception as e:
        logger.error(f"Error predict_dl: {e}", exc_info=True)
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

        # Intentar SVM (lazy)
        if load_svm_model():
            try:
                import pandas as pd
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
                    "prediccion": int(pred_svm),
                    "score_probabilidad": round(float(prob_svm), 4),
                    "resultado": "✅ Cliente aceptará" if pred_svm == 1 else "❌ Cliente no aceptará",
                    "graficas": {
                        "confusion_matrix": "/static/plots/svm_confusion.png",
                        "roc_curve": "/static/plots/svm_roc.png"
                    }
                }
            except Exception as e:
                logger.error(f"Error calculando SVM en predict_both: {e}", exc_info=True)
                results["svm"] = {"error": str(e)}
        else:
            results["svm"] = {"error": "Modelo no disponible"}

        # Intentar DL
        dl_ok = load_dl_model()
        if dl_ok and globals().get('model_dl') and globals().get('preprocessor_dl'):
            try:
                import pandas as pd
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
                model = globals()['model_dl']
                preprocessor = globals()['preprocessor_dl']
                X_proc = preprocessor.transform(input_df)
                prob_dl = float(model.predict(X_proc, verbose=0)[0][0])
                pred_dl = 1 if prob_dl >= 0.5 else 0
                results["deep_learning"] = {
                    "prediccion": int(pred_dl),
                    "score_probabilidad": round(float(prob_dl), 4),
                    "resultado": "✅ Cliente aceptará" if pred_dl == 1 else "❌ Cliente no aceptará",
                    "graficas": {
                        "confusion_matrix": "/static/plots/dl_confusion.png",
                        "roc_curve": "/static/plots/dl_roc.png",
                        "training_loss": "/static/plots/dl_loss.png",
                        "training_accuracy": "/static/plots/dl_accuracy.png"
                    }
                }
            except Exception as e:
                logger.error(f"Error calculando DL en predict_both: {e}", exc_info=True)
                results["deep_learning"] = {"error": str(e)}
        else:
            results["deep_learning"] = {"error": "Modelo no disponible"}

        # Ensemble si ambos disponibles
        try:
            if "svm" in results and "deep_learning" in results and isinstance(results["svm"], dict) and isinstance(results["deep_learning"], dict):
                if "score_probabilidad" in results["svm"] and "score_probabilidad" in results["deep_learning"]:
                    prob_avg = (results["svm"]["score_probabilidad"] + results["deep_learning"]["score_probabilidad"]) / 2
                    pred_avg = 1 if prob_avg >= 0.5 else 0
                    results["ensemble"] = {
                        "prediccion": int(pred_avg),
                        "score_probabilidad": round(float(prob_avg), 4),
                        "resultado": "✅ Cliente aceptará" if pred_avg == 1 else "❌ Cliente no aceptará",
                        "metodo": "Promedio de probabilidades"
                    }
        except Exception as e:
            logger.error(f"Error calculando ensemble: {e}", exc_info=True)

        if client is not None and collection is not None:
            try:
                collection.insert_one({**results, "_saved_at": datetime.utcnow()})
                logger.info("Predicción completa guardada en MongoDB")
            except Exception as mongo_e:
                logger.error(f"Error guardado MongoDB: {mongo_e}")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error predict_both: {e}", exc_info=True)
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