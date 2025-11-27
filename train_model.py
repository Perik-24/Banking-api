# train_model.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ------------- CONFIG ----------
CSV_PATH = "bank-full.csv"
TARGET = "y"
MODEL_OUTPUT = "modelo_banking.pkl"
PREPROCESSOR_OUTPUT = "preprocessor_svm.pkl"
ROC_OUTPUT = "svm_roc.npz"
PLOTS_DIR = "static/plots"
RANDOM_STATE = 42
os.makedirs(PLOTS_DIR, exist_ok=True)
# -------------------------------

# 1) Carga datos
df = pd.read_csv(CSV_PATH, sep=';')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2) Separar X e y
if TARGET not in df.columns:
    raise ValueError(f"La columna target '{TARGET}' no está en el CSV. Cambia TARGET en el script.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Si el target es 'yes'/'no', convertir a 0/1
if y.dtype == object:
    y = y.map(lambda v: 1 if str(v).lower() in ("yes", "si", "y", "1", "true") else 0)

# 3) Identificar columnas numéricas y categóricas
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# 4) Preprocesadores
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
], remainder="drop")

# 5) Pipeline completo: preprocessor + modelo
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", SVC(probability=True, random_state=RANDOM_STATE))
])

# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 7) Guardar preprocessor por separado (para uso en la API)
joblib.dump(preprocessor, PREPROCESSOR_OUTPUT)
print(f"Preprocessor guardado en {PREPROCESSOR_OUTPUT}")

# 8) Entrenar
print("Entrenando modelo SVM...")
model.fit(X_train, y_train)

# 9) Evaluar
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred))

# 10) Generar gráficas - Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Greens")
plt.title("Matriz de Confusión - SVM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "svm_confusion.png"), dpi=100)
plt.close()
print(f"Matriz de confusión guardada en {PLOTS_DIR}/svm_confusion.png")

# 11) Generar gráfica ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="green", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Curva ROC - SVM", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "svm_roc.png"), dpi=100)
plt.close()
print(f"Curva ROC guardada en {PLOTS_DIR}/svm_roc.png")

# 12) Guardar arrays ROC para uso posterior
np.savez(ROC_OUTPUT, fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)
print(f"ROC arrays guardados en {ROC_OUTPUT}")

# 13) Guardar pipeline completo (preprocesamiento + modelo)
joblib.dump(model, MODEL_OUTPUT)
print(f"Modelo SVM guardado en {MODEL_OUTPUT}")
print("=" * 50)
print("✓ Entrenamiento completado exitosamente")
print(f"✓ Gráficas guardadas en {PLOTS_DIR}/")
print("=" * 50)
