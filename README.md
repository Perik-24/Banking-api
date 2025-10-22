Este proyecto implementa un sistema de Machine Learning de extremo a extremo (End-to-End) que predice si un cliente bancario aceptará o no un nuevo producto, basándose en su perfil demográfico y su historial de contacto.

La solución utiliza un modelo de Support Vector Machine (SVM) y se despliega como una API en la nube (Azure App Service) para su consumo en tiempo real.

✨ Características Principales

Modelo de Clasificación: Utiliza un clasificador Support Vector Classification (SVC) de scikit-learn para predecir la propensión del cliente.

Pipeline Completo: El modelo .pkl serializado incluye el pipeline de preprocesamiento (imputación, One-Hot Encoding y estandarización de datos) para garantizar la consistencia en la predicción.

API en la Nube: API RESTful construida con Flask y desplegada en Azure App Service para escalabilidad y disponibilidad.

Interfaz de Usuario (Front-End): Incluye un index.html simple que actúa como cliente, enviando datos de prueba a la API y mostrando el resultado de la predicción en tiempo real.

Estructura Robusta: Utiliza joblib para la serialización y requirements.txt para la gestión de dependencias en el entorno de Azure.

🛠️ Tecnologías Utilizadas

Lenguaje: Python 3.x

Framework API: Flask

Machine Learning: Scikit-learn (SVC), NumPy, Pandas

Serialización: joblib

Despliegue: Azure App Service (Gunicorn server en Linux)

📁 Estructura del Proyecto

Banking_api/
├── app.py                  # Código de la API Flask para la predicción en tiempo real.
├── train_model.py          # Script para entrenar el modelo SVC/SVM y guardarlo.
├── modelo_banking.pkl      # Modelo binario SVC/SVM entrenado y serializado.
├── index.html              # Interfaz web (cliente) para enviar datos a la API.
├── requirements.txt        # Dependencias de Python necesarias para Azure.
├── bank-full.csv           # Archivo de datos de entrenamiento.
└── startup.txt             # Comando de inicio para Gunicorn en Azure App Service.


🚀 Despliegue en Azure

El proyecto está diseñado para ser desplegado fácilmente en un Azure App Service con pila de tiempo de ejecución Python (Linux).

Configuración de App Service: Se configura el comando de inicio en Azure para Gunicorn (gunicorn app:app).

Acceso a la API: Una vez desplegado, la predicción se realiza mediante una solicitud POST al endpoint /predict.

Ejemplo de Petición (desde el index.html o un cliente externo):

POST https://[tu-nombre-app].azurewebsites.net/predict
Content-Type: application/json

{
    "age": 45,
    "balance": 1500,
    "duration": 300,
    "campaign": 1,
    "...": "..."
}


Respuesta de la API:

{
    "prediccion": 1,
    "resultado": "✅ Cliente aceptará el producto",
    "score_probabilidad": 0.854
}
