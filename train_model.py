# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# ------------- CONFIG ----------
CSV_PATH = "bank-full.csv"   #CSV con tus datos aquí
TARGET = "y"               # nombre de la columna objetivo (cambiar si es otro)
MODEL_OUTPUT = "modelo_banking.pkl"
RANDOM_STATE = 42
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

# 7) Entrenar
print("Entrenando modelo...")
model.fit(X_train, y_train)

# 8) Evaluar
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred))

# 9) Guardar pipeline completo (preprocesamiento + modelo)
joblib.dump(model, MODEL_OUTPUT)
print(f"Modelo guardado en {MODEL_OUTPUT}")
