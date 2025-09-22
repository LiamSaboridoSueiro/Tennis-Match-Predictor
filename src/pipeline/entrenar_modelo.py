from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb

# --------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts" / "datasets" / "dataset_modelo.csv"
MODEL = ROOT / "artifacts" / "models" / "modelo_xgb.pkl"
META  = ROOT / "artifacts" / "models" / "metadata.json"
CALIB = ROOT / "artifacts" / "models" / "calibrador_isotonic.pkl"
BEST  = ROOT / "artifacts" / "models" / "best_xgb.json"  # generado por tune_xgb.py

# Carga de mejores hiperparámetros (si existen)
best_params: dict | None = None
best_thr_ext: float | None = None
if BEST.is_file():
    with open(BEST) as f:
        best_json = json.load(f)
        best_params = best_json.get("best_params")
        best_thr_ext = best_json.get("threshold")


def train():
    if not DATA.is_file() or DATA.stat().st_size == 0:
        raise RuntimeError(f"{DATA} está vacío o no existe. Revisa pasos previos del pipeline.")

    df = pd.read_csv(DATA, low_memory=False)
    if df.empty or df.shape[1] == 0:
        raise RuntimeError(f"{DATA} no contiene filas/columnas. Revisa preparar_datos.py.")

    # Tipos y orden
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # --------- FILTROS DE UNIVERSO (coherentes con el tuner) ----------
    if "best_of" in df.columns:
        df = df[df["best_of"].fillna(3).astype(int) == 3].copy()
    # if "level_code" in df.columns:
    #     df = df[df["level_code"].isin([0, 1, 2])].copy()
    # ------------------------------------------------------------------

    if df.empty:
        raise RuntimeError("Tras filtrar el universo, no quedan datos.")

    feature_cols = [c for c in df.columns if c not in ["y", "tourney_date"]]
    if not feature_cols:
        raise RuntimeError("No hay columnas de características tras excluir ['y','tourney_date'].")

    X = df[feature_cols]
    y = df["y"].astype(int)
    fechas = df["tourney_date"]

    # Splits temporales
    train_mask = fechas < "2024-01-01"
    valid_mask = (fechas >= "2024-01-01") & (fechas < "2025-01-01")
    test_mask  = fechas >= "2025-01-01"

    if X[train_mask].empty or X[valid_mask].empty or X[test_mask].empty:
        n_tr, n_va, n_te = X[train_mask].shape[0], X[valid_mask].shape[0], X[test_mask].shape[0]
        raise RuntimeError(f"Split temporal inválido. train/valid/test = {n_tr}/{n_va}/{n_te}")

    X_train, y_train = X[train_mask].values, y[train_mask].values
    X_valid, y_valid = X[valid_mask].values, y[valid_mask].values
    X_test,  y_test  = X[test_mask].values,  y[test_mask].values

    # Instanciación del modelo (fusión segura)
    base_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    if best_params:
        merged = dict(best_params)
        for k, v in base_params.items():
            merged.setdefault(k, v)
        model = XGBClassifier(**merged)
    else:
        model = XGBClassifier(
            **base_params,
            n_estimators=700,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
        )

    print(f"[xgboost] versión cargada: {xgb.__version__}")

    # Entrenamiento (compatibilidad: sin early stopping)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    # Probabilidades sin calibrar
    p_val_raw  = model.predict_proba(X_valid)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    # Calibración isotónica (ajustada en validación)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_valid)
    p_val  = iso.transform(p_val_raw)
    p_test = iso.transform(p_test_raw)

    # --------- UMBRALES ----------
    # 1) Umbral global
    ths = np.linspace(0.3, 0.7, 401)
    accs = [(t, accuracy_score(y_valid, (p_val >= t).astype(int))) for t in ths]
    best_thr_calc, best_acc_val = max(accs, key=lambda x: x[1])
    best_thr = float(best_thr_ext) if best_thr_ext is not None else float(best_thr_calc)

    print("== Validación 2024 ==")
    print(f"Acc_max@thr: {best_acc_val:.4f} @thr={best_thr_calc:.3f}")
    if best_thr_ext is not None:
        acc_forced = accuracy_score(y_valid, (p_val >= best_thr).astype(int))
        print(f"Acc@thr_forzado({best_thr:.3f} desde tuner): {acc_forced:.4f}")

    print("== Test 2025+ (calibrado) ==")
    print("LogLoss:", log_loss(y_test, p_test))
    print("Brier:", brier_score_loss(y_test, p_test))
    print("AUC:", roc_auc_score(y_test, p_test))
    print(f"Acc@thr={best_thr:.3f}:", accuracy_score(y_test, (p_test >= best_thr).astype(int)))


    # 2) Meta-umbrales robustos por contexto (level_code + surface_code), si existen
    def _ctx_cols(Xblock: pd.DataFrame):
        ctx = {}
        if "level_code" in Xblock.columns:
            ctx["level_code"] = Xblock["level_code"].to_numpy()
        if "surface_code" in Xblock.columns:
            ctx["surface_code"] = Xblock["surface_code"].to_numpy()
        return ctx

    X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
    X_test_df  = pd.DataFrame(X_test,  columns=feature_cols)
    ctx_val = _ctx_cols(X_valid_df)
    ctx_tst = _ctx_cols(X_test_df)

    # define claves disponibles en orden (level, surface)
    keys = [k for k in ["level_code", "surface_code"] if k in ctx_val]
    thresholds_by_ctx = {}
    acc_ctx = None

    if keys:
        from collections import defaultdict

        def make_key(i, ctx):
            return tuple(int(ctx[k][i]) for k in keys)

        groups_val = defaultdict(list)
        for i in range(len(y_valid)):
            groups_val[make_key(i, ctx_val)].append(i)

        # grupo mínimo grande para evitar ruido
        MIN_GROUP = 1000

        for g, idxs in groups_val.items():
            if len(idxs) < MIN_GROUP:
                continue
            p = p_val[idxs]
            yv = y_valid[idxs]
            accs_g = [(t, accuracy_score(yv, (p >= t).astype(int))) for t in ths]
            thr_g, _ = max(accs_g, key=lambda x: x[1])
            thresholds_by_ctx[g] = float(thr_g)

        if thresholds_by_ctx:
            y_pred_ctx = np.zeros_like(y_test)
            for i in range(len(y_test)):
                g = make_key(i, ctx_tst)
                thr = thresholds_by_ctx.get(g, best_thr)
                y_pred_ctx[i] = 1 if p_test[i] >= thr else 0
            acc_ctx = accuracy_score(y_test, y_pred_ctx)

    # --------- MÉTRICAS ----------
    print("== Validación 2024 ==")
    print(f"Acc_max@thr: {best_acc_val:.4f} @thr={best_thr_calc:.3f}")
    if best_thr_ext is not None:
        acc_forced = accuracy_score(y_valid, (p_val >= best_thr).astype(int))
        print(f"Acc@thr_forzado({best_thr:.3f} desde tuner): {acc_forced:.4f}")

    print("== Test 2025+ (calibrado) ==")
    print("LogLoss:", log_loss(y_test, p_test))
    print("Brier:", brier_score_loss(y_test, p_test))
    print("AUC:", roc_auc_score(y_test, p_test))
    print(f"Acc@thr={best_thr:.3f}:", accuracy_score(y_test, (p_test >= best_thr).astype(int)))
    if acc_ctx is not None:
        print(f"Acc con meta-umbrales por contexto ({'+'.join(keys)}): {acc_ctx:.4f}")

    # --------- Guardado ---------
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    joblib.dump(iso,   CALIB)
    meta_payload = {
        "feature_order": feature_cols,
        "train": "<2024-01-01",
        "valid": "2024-01-01..2024-12-31",
        "test": ">=2025-01-01",
        "data_file": str(DATA),
        "calibration": "isotonic",
        "threshold": float(best_thr),
        "used_best_params": bool(best_params is not None),
    }
    if thresholds_by_ctx:
        meta_payload["thresholds_by_context"] = {",".join(map(str, k)): v for k, v in thresholds_by_ctx.items()}
        meta_payload["context_keys"] = keys

    with open(META, "w") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"Modelo: {MODEL}")
    print(f"Calibrador: {CALIB}")
    print(f"Metadatos: {META}")


if __name__ == "__main__":
    train()
