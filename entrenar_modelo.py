# entrenar_modelo.py  (con calibración + umbral óptimo, import corregido)
import os, json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import joblib

DATA = "dataset_modelo.csv"
MODEL = "modelo_xgb.pkl"
META  = "metadata.json"
CALIB = "calibrador_isotonic.pkl"

if not os.path.isfile(DATA) or os.path.getsize(DATA) == 0:
    raise RuntimeError(f"{DATA} está vacío o no existe. Revisa pasos previos del pipeline.")

df = pd.read_csv(DATA)
if df.empty or df.shape[1] == 0:
    raise RuntimeError(f"{DATA} no contiene filas/columnas. Revisa preparar_datos.py.")

df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
feature_cols = [c for c in df.columns if c not in ["y","tourney_date"]]
X = df[feature_cols]
y = df["y"].astype(int)
fechas = df["tourney_date"]

# Splits temporales
train_mask = fechas < "2024-01-01"
valid_mask = (fechas >= "2024-01-01") & (fechas < "2025-01-01")
test_mask  = fechas >= "2025-01-01"

X_train, y_train = X[train_mask].values, y[train_mask].values
X_valid, y_valid = X[valid_mask].values, y[valid_mask].values
X_test,  y_test  = X[test_mask].values,  y[test_mask].values

# Modelo (sin early stopping para compatibilidad)
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=700,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# Probabilidades sin calibrar
p_val_raw  = model.predict_proba(X_valid)[:, 1]
p_test_raw = model.predict_proba(X_test)[:, 1]

# Calibración isotónica en validación 2024
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_val_raw, y_valid)
p_val  = iso.transform(p_val_raw)
p_test = iso.transform(p_test_raw)

# Umbral óptimo (max Accuracy) en validación calibrada
ths = np.linspace(0.3, 0.7, 401)
accs = [(t, accuracy_score(y_valid, (p_val >= t).astype(int))) for t in ths]
best_thr, best_acc = max(accs, key=lambda x: x[1])

# Métricas en test (calibradas)
print("== Evaluación TEST (calibrado) ==")
print("LogLoss:", log_loss(y_test, p_test))
print("Brier:", brier_score_loss(y_test, p_test))
print("AUC:", roc_auc_score(y_test, p_test))
print(f"Acc@thr={best_thr:.3f}:", accuracy_score(y_test, (p_test >= best_thr).astype(int)))

# Guarda artefactos
joblib.dump(model, MODEL)
joblib.dump(iso,   CALIB)
with open(META, "w") as f:
    json.dump(
        {
            "feature_order": feature_cols,
            "train":"<2024-01-01",
            "valid":"2024-01-01..2024-12-31",
            "test":">=2025-01-01",
            "data_file": DATA,
            "calibration": "isotonic",
            "threshold": float(best_thr)
        },
        f, indent=2
    )

print(f"✅ Modelo guardado en {MODEL}, calibrador en {CALIB} y metadatos en {META}")