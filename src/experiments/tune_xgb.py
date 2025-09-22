from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts" / "datasets" / "dataset_modelo.csv"
OUT  = ROOT / "artifacts" / "models" / "best_xgb.json"


def load_data():
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    df = pd.read_csv(DATA, low_memory=False)
    if df.empty:
        raise RuntimeError(f"{DATA} está vacío.")

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # --------- FILTROS DE UNIVERSO (coherentes con el entrenador) ----------
    # BO3 únicamente
    if "best_of" in df.columns:
        df = df[df["best_of"].fillna(3).astype(int) == 3].copy()
    # Niveles altos si existe level_code (G/F/A => 0/1/2 según tu encode)
    # if "level_code" in df.columns:
    #     df = df[df["level_code"].isin([0, 1, 2])].copy()
    # -----------------------------------------------------------------------

    feat_cols = [c for c in df.columns if c not in ["y", "tourney_date"]]
    if not feat_cols:
        raise RuntimeError("No hay columnas de características tras excluir ['y','tourney_date'].")

    X = df[feat_cols]
    y = df["y"].astype(int).values
    fechas = df["tourney_date"]

    train_mask = fechas < "2024-01-01"
    valid_mask = (fechas >= "2024-01-01") & (fechas < "2025-01-01")

    return X[train_mask].values, y[train_mask], X[valid_mask].values, y[valid_mask], feat_cols


def objective(trial: optuna.Trial):
    X_tr, y_tr, X_va, y_va, _ = load_data()
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("n_estimators", 300, 1600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
    }
    model = XGBClassifier(**params)

    # Compatibilidad: sin early stopping
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_val = model.predict_proba(X_va)[:, 1]
    ths = np.linspace(0.3, 0.7, 401)
    accs = [accuracy_score(y_va, (p_val >= t).astype(int)) for t in ths]
    acc = max(accs)
    auc = roc_auc_score(y_va, p_val)
    ll = log_loss(y_va, p_val, labels=[0, 1])
    brier = brier_score_loss(y_va, p_val)

    # score compuesto: prioriza Accuracy, mantiene AUC, penaliza logloss/brier
    score = acc * 0.65 + auc * 0.25 - ll * 0.08 - brier * 0.02
    trial.set_user_attr("acc", float(acc))
    trial.set_user_attr("auc", float(auc))
    trial.set_user_attr("logloss", float(ll))
    trial.set_user_attr("brier", float(brier))
    return score


def main():
    print(f"[xgboost] versión cargada: {xgb.__version__}")
    study = optuna.create_study(direction="maximize", study_name="xgb_tennis")
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    best = study.best_trial
    X_tr, y_tr, X_va, y_va, feat_cols = load_data()
    params = best.params
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    })

    # Reentrena con los mejores params y calcula umbral óptimo en validación
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_val = model.predict_proba(X_va)[:, 1]
    ths = np.linspace(0.3, 0.7, 401)
    accs = [(t, accuracy_score(y_va, (p_val >= t).astype(int))) for t in ths]
    best_thr, best_acc = max(accs, key=lambda x: x[1])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {
                "best_params": params,
                "threshold": float(best_thr),
                "valid_acc": float(best_acc),
                "valid_auc": float(roc_auc_score(y_va, p_val)),
                "feature_order": feat_cols,
                "filters": {"best_of": 3, "level_code_in": [0, 1, 2]},
            },
            f,
            indent=2,
        )
    print(f"[tune_xgb] guardado {OUT}")
    print(f"[tune_xgb] valid_acc={best_acc:.4f} threshold={best_thr:.3f}")


if __name__ == "__main__":
    main()
