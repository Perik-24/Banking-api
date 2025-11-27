# train_dl_model.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# CONFIG
CSV_PATH = "bank-full.csv"
TARGET = "y"
MODEL_OUTPUT = "modelo_dl_banking.h5"
PREPROCESSOR_OUTPUT = "preprocessor_dl.pkl"
ROC_OUTPUT = "dl_roc.npz"
HISTORY_OUTPUT = "dl_history.npz"
PLOTS_DIR = "static/plots"
RANDOM_STATE = 42
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1) Carga datos
df = pd.read_csv(CSV_PATH, sep=';')
if TARGET not in df.columns:
    raise ValueError(f"La columna target '{TARGET}' no está en el CSV.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 2) Adaptar target a 0/1
if y.dtype == object:
    y = y.map(lambda v: 1 if str(v).lower() in ("yes", "si", "y", "1", "true") else 0)

# 3) Preprocesamiento (igual que SVM)
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
], remainder="drop")

X_proc = preprocessor.fit_transform(X)

# Guardamos el preprocessor para usarlo en la API
joblib.dump(preprocessor, PREPROCESSOR_OUTPUT)
print(f"Preprocessor guardado en {PREPROCESSOR_OUTPUT}")

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 5) Definición del modelo DL
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # salida binaria
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es], verbose=2)

# 6) Guardar curvas de entrenamiento
np.savez(HISTORY_OUTPUT,
         loss=np.array(history.history.get("loss", [])),
         val_loss=np.array(history.history.get("val_loss", [])),
         acc=np.array(history.history.get("accuracy", [])),
         val_acc=np.array(history.history.get("val_accuracy", [])))
print(f"History guardado en {HISTORY_OUTPUT}")

# Generar gráficas de entrenamiento - Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Loss (Binary Crossentropy)", fontsize=12)
plt.title("Curvas de Loss - Deep Learning", fontsize=14, fontweight="bold")
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dl_loss.png"), dpi=100)
plt.close()

# Generar gráficas de entrenamiento - Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Curvas de Accuracy - Deep Learning", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dl_accuracy.png"), dpi=100)
plt.close()
print(f"Gráficas de training (loss y accuracy) guardadas en {PLOTS_DIR}")

# 7) Evaluación: ROC y matriz de confusión
y_pred_prob = model.predict(X_test).ravel()
y_pred_bin = (y_pred_prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_bin)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Matriz de confusión - DL")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dl_confusion.png"))
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - DL")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dl_roc.png"))
plt.close()

# Guardar arrays ROC para usarlos en la app
np.savez(ROC_OUTPUT, fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)
print(f"ROC arrays guardados en {ROC_OUTPUT}")

# 8) Guardar modelo
model.save(MODEL_OUTPUT)
print(f"Modelo DL guardado en {MODEL_OUTPUT}")
print("Gráficas de entrenamiento y evaluación guardadas en", PLOTS_DIR)