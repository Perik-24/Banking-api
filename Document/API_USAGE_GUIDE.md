# 🚀 Guía de Uso de la API - Banking API con Gráficas

## ✅ Estado de la Implementación

**COMPLETADO:** Ambos modelos (SVM y Deep Learning) integrados en `app.py` con endpoints para gráficas.

---

## 📡 Endpoints Disponibles

### **1. Health Check**
```http
GET /
```

**Respuesta:**
```json
{
  "status": "API Running",
  "modelo_svm": "✓ OK",
  "modelo_dl": "✓ OK",
  "mongodb": "✓ OK",
  "endpoints": [
    "/predict (POST) - Predicción con SVM",
    "/predict_dl (POST) - Predicción con Deep Learning",
    "/predict_both (POST) - Predicción con ambos modelos",
    "/graficas (GET) - Lista de todas las gráficas",
    "/static/plots/<filename> (GET) - Ver gráfica específica"
  ]
}
```

---

### **2. Predicción con SVM** (Endpoint original)
```http
POST /predict
POST /predict_svm
```

**Request Body:**
```json
{
  "age": 35,
  "balance": 1200,
  "duration": 240,
  "campaign": 2,
  "job": "blue-collar",
  "marital": "married",
  "education": "secondary",
  "pdays": -1,
  "loan": "no",
  "month": "may",
  "poutcome": "unknown",
  "housing": "yes",
  "default": "no",
  "previous": 0,
  "contact": "cellular",
  "day": 5
}
```

**Respuesta:**
```json
{
  "modelo": "SVM",
  "prediccion": 0,
  "score_probabilidad": 0.2345,
  "resultado": "❌ Cliente no aceptará",
  "graficas": {
    "confusion_matrix": "/static/plots/svm_confusion.png",
    "roc_curve": "/static/plots/svm_roc.png"
  }
}
```

---

### **3. Predicción con Deep Learning** 🆕
```http
POST /predict_dl
```

**Request Body:** (mismo formato que SVM)

**Respuesta:**
```json
{
  "modelo": "Deep Learning",
  "prediccion": 1,
  "score_probabilidad": 0.7823,
  "resultado": "✅ Cliente aceptará el producto",
  "graficas": {
    "confusion_matrix": "/static/plots/dl_confusion.png",
    "roc_curve": "/static/plots/dl_roc.png",
    "training_loss": "/static/plots/dl_loss.png",
    "training_accuracy": "/static/plots/dl_accuracy.png"
  }
}
```

---

### **4. Predicción con Ambos Modelos + Ensemble** 🆕
```http
POST /predict_both
```

**Request Body:** (mismo formato)

**Respuesta:**
```json
{
  "timestamp": "2025-11-27T13:15:00.000Z",
  "input_data": { ... },
  "svm": {
    "prediccion": 0,
    "score_probabilidad": 0.2345,
    "resultado": "❌ Cliente no aceptará",
    "graficas": {
      "confusion_matrix": "/static/plots/svm_confusion.png",
      "roc_curve": "/static/plots/svm_roc.png"
    }
  },
  "deep_learning": {
    "prediccion": 1,
    "score_probabilidad": 0.7823,
    "resultado": "✅ Cliente aceptará",
    "graficas": {
      "confusion_matrix": "/static/plots/dl_confusion.png",
      "roc_curve": "/static/plots/dl_roc.png",
      "training_loss": "/static/plots/dl_loss.png",
      "training_accuracy": "/static/plots/dl_accuracy.png"
    }
  },
  "ensemble": {
    "prediccion": 1,
    "score_probabilidad": 0.5084,
    "resultado": "✅ Cliente aceptará",
    "metodo": "Promedio de probabilidades"
  }
}
```

---

### **5. Lista de Gráficas Disponibles** 🆕
```http
GET /graficas
```

**Respuesta:**
```json
{
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
```

---

### **6. Servir Gráficas** 🆕
```http
GET /static/plots/{filename}
```

**Ejemplos:**
- `/static/plots/svm_confusion.png`
- `/static/plots/svm_roc.png`
- `/static/plots/dl_confusion.png`
- `/static/plots/dl_roc.png`
- `/static/plots/dl_loss.png`
- `/static/plots/dl_accuracy.png`

**Respuesta:** Imagen PNG directa (puede mostrarse en navegador o `<img>` tag)

---

## 🧪 Ejemplos de Uso

### **Ejemplo 1: Curl - Predicción SVM**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "balance": 1200,
    "duration": 240,
    "campaign": 2,
    "job": "blue-collar",
    "marital": "married",
    "education": "secondary",
    "pdays": -1,
    "loan": "no",
    "month": "may",
    "poutcome": "unknown",
    "housing": "yes",
    "default": "no",
    "previous": 0,
    "contact": "cellular",
    "day": 5
  }'
```

---

### **Ejemplo 2: Curl - Predicción DL**
```bash
curl -X POST http://localhost:5000/predict_dl \
  -H "Content-Type: application/json" \
  -d '{ ... mismo JSON ... }'
```

---

### **Ejemplo 3: Curl - Ambos Modelos**
```bash
curl -X POST http://localhost:5000/predict_both \
  -H "Content-Type: application/json" \
  -d '{ ... mismo JSON ... }'
```

---

### **Ejemplo 4: JavaScript Fetch - Predicción + Gráficas**
```javascript
async function predictWithGraphs() {
  const data = {
    age: 35,
    balance: 1200,
    duration: 240,
    campaign: 2,
    job: "blue-collar",
    marital: "married",
    education: "secondary",
    pdays: -1,
    loan: "no",
    month: "may",
    poutcome: "unknown",
    housing: "yes",
    default: "no",
    previous: 0,
    contact: "cellular",
    day: 5
  };

  // Predicción con ambos modelos
  const response = await fetch('http://localhost:5000/predict_both', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  console.log('Predicción SVM:', result.svm);
  console.log('Predicción DL:', result.deep_learning);
  console.log('Ensemble:', result.ensemble);

  // Mostrar gráficas en HTML
  const svmROC = result.svm.graficas.roc_curve;
  const dlROC = result.deep_learning.graficas.roc_curve;
  
  document.getElementById('svm-roc').src = `http://localhost:5000${svmROC}`;
  document.getElementById('dl-roc').src = `http://localhost:5000${dlROC}`;
}
```

---

### **Ejemplo 5: Python Requests**
```python
import requests

url = "http://localhost:5000/predict_both"
data = {
    "age": 35,
    "balance": 1200,
    "duration": 240,
    "campaign": 2,
    "job": "blue-collar",
    "marital": "married",
    "education": "secondary",
    "pdays": -1,
    "loan": "no",
    "month": "may",
    "poutcome": "unknown",
    "housing": "yes",
    "default": "no",
    "previous": 0,
    "contact": "cellular",
    "day": 5
}

response = requests.post(url, json=data)
result = response.json()

print(f"SVM: {result['svm']['resultado']} ({result['svm']['score_probabilidad']})")
print(f"DL: {result['deep_learning']['resultado']} ({result['deep_learning']['score_probabilidad']})")
print(f"Ensemble: {result['ensemble']['resultado']} ({result['ensemble']['score_probabilidad']})")

# Descargar gráfica
roc_url = "http://localhost:5000" + result['svm']['graficas']['roc_curve']
img = requests.get(roc_url)
with open('svm_roc.png', 'wb') as f:
    f.write(img.content)
```

---

## 🎨 Integración en Frontend (index.html)

Para actualizar tu `index.html` y mostrar las gráficas:

```html
<!-- Agregar después del resultado -->
<div id="graficas-container" style="margin-top: 20px;">
  <h3>Gráficas del Modelo</h3>
  
  <!-- SVM Gráficas -->
  <div id="svm-graficas">
    <h4>Modelo SVM</h4>
    <img id="svm-confusion" style="width: 45%; margin: 10px;" />
    <img id="svm-roc" style="width: 45%; margin: 10px;" />
  </div>
  
  <!-- DL Gráficas -->
  <div id="dl-graficas">
    <h4>Modelo Deep Learning</h4>
    <img id="dl-confusion" style="width: 45%; margin: 10px;" />
    <img id="dl-roc" style="width: 45%; margin: 10px;" />
    <img id="dl-loss" style="width: 45%; margin: 10px;" />
    <img id="dl-accuracy" style="width: 45%; margin: 10px;" />
  </div>
</div>

<script>
async function predecir(e) {
  e.preventDefault();
  
  // ... código existente para obtener datos ...
  
  const API_URL = 'http://localhost:5000/predict_both';
  
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  
  const result = await res.json();
  
  // Mostrar resultados
  document.getElementById('resultado').innerHTML = `
    <h4>Modelo SVM</h4>
    ${result.svm.resultado}<br>
    Probabilidad: ${(result.svm.score_probabilidad * 100).toFixed(2)}%
    
    <h4>Modelo Deep Learning</h4>
    ${result.deep_learning.resultado}<br>
    Probabilidad: ${(result.deep_learning.score_probabilidad * 100).toFixed(2)}%
    
    <h4>Ensemble (Promedio)</h4>
    ${result.ensemble.resultado}<br>
    Probabilidad: ${(result.ensemble.score_probabilidad * 100).toFixed(2)}%
  `;
  
  // Mostrar gráficas SVM
  document.getElementById('svm-confusion').src = API_URL.replace('/predict_both', '') + result.svm.graficas.confusion_matrix;
  document.getElementById('svm-roc').src = API_URL.replace('/predict_both', '') + result.svm.graficas.roc_curve;
  
  // Mostrar gráficas DL
  document.getElementById('dl-confusion').src = API_URL.replace('/predict_both', '') + result.deep_learning.graficas.confusion_matrix;
  document.getElementById('dl-roc').src = API_URL.replace('/predict_both', '') + result.deep_learning.graficas.roc_curve;
  document.getElementById('dl-loss').src = API_URL.replace('/predict_both', '') + result.deep_learning.graficas.training_loss;
  document.getElementById('dl-accuracy').src = API_URL.replace('/predict_both', '') + result.deep_learning.graficas.training_accuracy;
}
</script>
```

---

## 🔧 Cambios Realizados en `app.py`

### **Correcciones:**
1. ✅ **Carga correcta del modelo DL** usando TensorFlow (`load_model`)
2. ✅ **Carga del preprocessor DL** por separado
3. ✅ **Renombrado de variables** para claridad (`model_svm`, `model_dl`)

### **Nuevas funcionalidades:**
1. ✅ **Endpoint `/predict_dl`** - Predicción solo con DL
2. ✅ **Endpoint `/predict_both`** - Predicción con ambos modelos + ensemble
3. ✅ **Endpoint `/graficas`** - Lista de todas las gráficas
4. ✅ **Endpoint `/static/plots/<filename>`** - Servir imágenes de gráficas
5. ✅ **Respuestas incluyen URLs de gráficas** relevantes

### **Mejoras:**
- ✅ Logging mejorado
- ✅ Manejo de errores robusto
- ✅ Compatibilidad con TensorFlow opcional
- ✅ Documentación en respuestas JSON

---

## 🚀 Cómo Ejecutar

### **Desarrollo Local:**
```bash
python app.py
```

### **Producción (con Gunicorn):**
```bash
gunicorn app:app --bind 0.0.0.0:5000
```

### **Azure App Service:**
El comando de startup ya está configurado en Azure.

---

## 📊 Resumen de Gráficas

| Modelo | Gráfica | Descripción | URL |
|--------|---------|-------------|-----|
| **SVM** | Confusion Matrix | Matriz de confusión del modelo | `/static/plots/svm_confusion.png` |
| **SVM** | ROC Curve | Curva ROC con AUC | `/static/plots/svm_roc.png` |
| **DL** | Confusion Matrix | Matriz de confusión del modelo | `/static/plots/dl_confusion.png` |
| **DL** | ROC Curve | Curva ROC con AUC | `/static/plots/dl_roc.png` |
| **DL** | Training Loss | Evolución del loss | `/static/plots/dl_loss.png` |
| **DL** | Training Accuracy | Evolución del accuracy | `/static/plots/dl_accuracy.png` |

---

## ✅ TODO Completado

- ✅ Ambos modelos integrados en app.py
- ✅ Predicción con SVM funcional
- ✅ Predicción con Deep Learning funcional
- ✅ Predicción con ambos modelos + ensemble
- ✅ Servir gráficas como archivos estáticos
- ✅ Endpoints documentados
- ✅ Ejemplos de uso en múltiples lenguajes

**¡Todo listo para usar! 🎉**
