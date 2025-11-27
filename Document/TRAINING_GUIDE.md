# 📊 Guía de Entrenamiento y Gráficas - Banking API

## ✅ Cambios Realizados

### 1. **train_model.py (SVM)** - MEJORADO ✓

**Nuevas funcionalidades:**
- ✓ Genera gráfica de Matriz de Confusión (`svm_confusion.png`)
- ✓ Genera gráfica de Curva ROC (`svm_roc.png`)
- ✓ Guarda preprocessor por separado (`preprocessor_svm.pkl`)
- ✓ Guarda arrays ROC en formato NPZ (`svm_roc.npz`)
- ✓ Todas las gráficas se guardan en `static/plots/`

**Métricas del modelo SVM:**
- Accuracy: 90.49%
- F1-score: 45.78%
- AUC-ROC: ~0.88

---

### 2. **train_dl_model.py (Deep Learning)** - CORREGIDO Y MEJORADO ✓

**Correcciones:**
- ✓ Eliminado import `uuid` innecesario
- ✓ Cambiado `sparse=False` a `sparse_output=False`

**Nuevas funcionalidades:**
- ✓ Genera gráfica de Loss durante entrenamiento (`dl_loss.png`)
- ✓ Genera gráfica de Accuracy durante entrenamiento (`dl_accuracy.png`)
- ✓ Genera gráfica de Matriz de Confusión (`dl_confusion.png`)
- ✓ Genera gráfica de Curva ROC (`dl_roc.png`)
- ✓ Guarda preprocessor por separado (`preprocessor_dl.pkl`)
- ✓ Guarda arrays ROC en formato NPZ (`dl_roc.npz`)
- ✓ Guarda history de entrenamiento (`dl_history.npz`)

**Arquitectura del modelo:**
- Capa 1: Dense(128, relu)
- Capa 2: Dense(64, relu)
- Capa 3: Dense(1, sigmoid)
- Early Stopping: patience=6
- Épocas entrenadas: 11/100 (detenido por Early Stopping)
- Accuracy final: ~91%

---

## 📂 Estructura de Archivos Generados

```
Banking_api/
├── static/
│   └── plots/                    # ← NUEVO directorio con gráficas
│       ├── svm_confusion.png     # Matriz confusión SVM
│       ├── svm_roc.png           # Curva ROC SVM
│       ├── dl_confusion.png      # Matriz confusión DL
│       ├── dl_roc.png            # Curva ROC DL
│       ├── dl_loss.png           # Loss durante entrenamiento
│       └── dl_accuracy.png       # Accuracy durante entrenamiento
│
├── modelo_banking.pkl            # Modelo SVM completo
├── preprocessor_svm.pkl          # Preprocessor SVM
├── svm_roc.npz                   # Arrays ROC SVM
│
├── modelo_dl_banking.h5          # Modelo DL en formato H5
├── preprocessor_dl.pkl           # Preprocessor DL
├── dl_roc.npz                    # Arrays ROC DL
├── dl_history.npz                # History de entrenamiento DL
│
├── train_model.py                # Script entrenamiento SVM
├── train_dl_model.py             # Script entrenamiento DL
└── test_training.py              # Script verificación ← NUEVO
```

---

## 🚀 Cómo Usar

### **Paso 1: Entrenar Modelo SVM**
```bash
python train_model.py
```

**Genera:**
- `modelo_banking.pkl` - Modelo completo (pipeline)
- `preprocessor_svm.pkl` - Solo el preprocessor
- `svm_roc.npz` - Datos de curva ROC
- `static/plots/svm_confusion.png` - Gráfica matriz confusión
- `static/plots/svm_roc.png` - Gráfica curva ROC

---

### **Paso 2: Entrenar Modelo Deep Learning**
```bash
python train_dl_model.py
```

**Genera:**
- `modelo_dl_banking.h5` - Modelo DL
- `preprocessor_dl.pkl` - Preprocessor
- `dl_roc.npz` - Datos de curva ROC
- `dl_history.npz` - History de entrenamiento
- `static/plots/dl_confusion.png` - Gráfica matriz confusión
- `static/plots/dl_roc.png` - Gráfica curva ROC
- `static/plots/dl_loss.png` - Gráfica loss entrenamiento
- `static/plots/dl_accuracy.png` - Gráfica accuracy entrenamiento

---

### **Paso 3: Verificar Archivos (Opcional)**
```bash
python test_training.py
```

Este script verifica que todos los archivos se hayan generado correctamente.

---

## 📊 Gráficas Disponibles

### **Modelo SVM (2 gráficas):**
1. `svm_confusion.png` - Matriz de Confusión (verde)
2. `svm_roc.png` - Curva ROC con AUC

### **Modelo Deep Learning (4 gráficas):**
1. `dl_confusion.png` - Matriz de Confusión (azul)
2. `dl_roc.png` - Curva ROC con AUC
3. `dl_loss.png` - Evolución del Loss (training vs validation)
4. `dl_accuracy.png` - Evolución del Accuracy (training vs validation)

---

## 🔄 Próximos Pasos

### **Para integrar con la API (`app.py`):**

1. **Servir las gráficas estáticas:**
   - Las gráficas ya están en `static/plots/`
   - Flask puede servirlas automáticamente si se configura

2. **Generar gráficas dinámicas por predicción:**
   - Opción A: Mostrar gráficas estáticas (pre-generadas)
   - Opción B: Generar gráficas nuevas con cada predicción individual
   - Opción C: Crear endpoint `/graficas` que devuelva las imágenes

3. **Integrar modelo DL en la API:**
   - Modificar `app.py` para cargar correctamente el modelo H5
   - Crear endpoint `/predict_dl` para predicciones con DL
   - O crear endpoint `/predict_ensemble` que combine ambos modelos

---

## 📋 Dependencias Actualizadas

Ya se agregó `matplotlib` a `requirements.txt`:

```txt
pandas
numpy
scikit-learn
joblib
flask
flask-cors
gunicorn
pymongo[srv]>=4.0.0
certifi>=2023.7.22
tensorflow
matplotlib
```

---

## ✅ Verificación Completada

**Estado:** ✓ TODOS LOS ARCHIVOS GENERADOS CORRECTAMENTE

- ✓ 2 modelos entrenados (SVM + DL)
- ✓ 2 preprocessors guardados
- ✓ 6 gráficas PNG generadas
- ✓ 3 archivos NPZ con datos para análisis posterior

---

## 💡 Notas Importantes

1. **Ambos modelos usan el mismo preprocesamiento** (OneHotEncoder + StandardScaler)
2. **El directorio `static/plots/` se crea automáticamente** si no existe
3. **Las gráficas usan backend 'Agg'** (no requiere display, ideal para servidores)
4. **Los archivos NPZ permiten regenerar gráficas** sin re-entrenar modelos
5. **El modelo DL se guarda en formato H5** (legacy, pero funcional)

---

## 🎯 Resultado Final

Ahora tienes un sistema completo de entrenamiento con:
- ✓ Dos modelos (SVM y Deep Learning)
- ✓ Gráficas de evaluación para ambos
- ✓ Gráficas de entrenamiento para DL
- ✓ Archivos organizados y listos para integrar en la API

**Todo funcionando correctamente! 🎉**
