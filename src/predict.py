# predict.py
import pandas as pd
import numpy as np
import joblib, json
from datetime import datetime
from utils import normalizar_nombre

MODEL = "artifacts/models/modelo_xgb.pkl"
META  = "artifacts/models/metadata.json"
HIST  = "artifacts/datasets/partidos_enriquecidos_final.csv"

def encode_surface(s):
    s = (s or "").strip().lower()
    return {"hard":0,"clay":1,"grass":2,"carpet":3}.get(s, -1)

def encode_level(l):
    l = (l or "").strip().upper()
    return {"G":0,"F":1,"A":2,"D":3,"M":1}.get(l, -1)

round_order = ["R128","R64","R32","R16","QF","SF","F"]
def encode_round(r):
    try: return round_order.index(r)
    except: return -1

def ultimo_valor(df, jugador, fecha, col_when_win, col_when_lose):
    sub = df[(df["tourney_date"] < fecha) & ((df["winner_name"]==jugador)|(df["loser_name"]==jugador))]
    if sub.empty: return np.nan
    row = sub.iloc[-1]
    return row[col_when_win] if row["winner_name"]==jugador else row[col_when_lose]

def build_features(j1, j2, fecha, surface, level="A", best_of=3, rnd="R32"):
    # carga histórico
    hist = pd.read_csv(HIST)
    hist["tourney_date"] = pd.to_datetime(hist["tourney_date"])
    hist["winner_name"] = hist["winner_name"].astype(str).map(normalizar_nombre)
    hist["loser_name"]  = hist["loser_name"].astype(str).map(normalizar_nombre)

    j1 = normalizar_nombre(j1); j2 = normalizar_nombre(j2)
    fecha = pd.to_datetime(fecha)

    # valores últimos antes de 'fecha'
    def U(j, cw, cl): return ultimo_valor(hist, j, fecha, cw, cl)
    rank_j1  = U(j1, "winner_rank","loser_rank")
    rank_j2  = U(j2, "winner_rank","loser_rank")
    rp_j1    = U(j1, "winner_rank_points","loser_rank_points")
    rp_j2    = U(j2, "winner_rank_points","loser_rank_points")
    age_j1   = U(j1, "winner_age","loser_age")
    age_j2   = U(j2, "winner_age","loser_age")
    ht_j1    = U(j1, "winner_ht","loser_ht")
    ht_j2    = U(j2, "winner_ht","loser_ht")
    elo_j1   = U(j1, "winner_elo","loser_elo")
    elo_j2   = U(j2, "winner_elo","loser_elo")
    elos_j1  = U(j1, "winner_elo_surface","loser_elo_surface")
    elos_j2  = U(j2, "winner_elo_surface","loser_elo_surface")
    wrs_j1   = U(j1, "winner_winrate_surface","loser_winrate_surface")
    wrs_j2   = U(j2, "winner_winrate_surface","loser_winrate_surface")
    f5_j1    = U(j1, "winner_form_last5","loser_form_last5")
    f5_j2    = U(j2, "winner_form_last5","loser_form_last5")
    f10_j1   = U(j1, "winner_form_last10","loser_form_last10")
    f10_j2   = U(j2, "winner_form_last10","loser_form_last10")
    f25_j1   = U(j1, "winner_form_last25","loser_form_last25")
    f25_j2   = U(j2, "winner_form_last25","loser_form_last25")
    fs5_j1   = U(j1, "winner_form_surface_last5","loser_form_surface_last5")
    fs5_j2   = U(j2, "winner_form_surface_last5","loser_form_surface_last5")
    fs10_j1  = U(j1, "winner_form_surface_last10","loser_form_surface_last10")
    fs10_j2  = U(j2, "winner_form_surface_last10","loser_form_surface_last10")
    rest_j1  = U(j1, "winner_rest_days","loser_rest_days")
    rest_j2  = U(j2, "winner_rest_days","loser_rest_days")
    fat_j1   = U(j1, "winner_fatigue","loser_fatigue")
    fat_j2   = U(j2, "winner_fatigue","loser_fatigue")

    def diff(a,b):
        if pd.isna(a) and pd.isna(b): return np.nan
        return (a if pd.notna(a) else 0) - (b if pd.notna(b) else 0)

    row = {
        "tourney_date": fecha,
        "surface_code": encode_surface(surface),
        "level_code": encode_level(level),
        "best_of": best_of,
        "round_code": encode_round(rnd),

        "rank_diff": diff(rank_j1, rank_j2),
        "rank_points_diff": diff(rp_j1, rp_j2),
        "age_diff": diff(age_j1, age_j2),
        "ht_diff": diff(ht_j1, ht_j2),

        "elo_diff": diff(elo_j1, elo_j2),
        "elo_surface_diff": diff(elos_j1, elos_j2),  # ✅ nuevo

        "winrate_surface_diff": diff(wrs_j1, wrs_j2),
        "form_last5_diff": diff(f5_j1, f5_j2),
        "form_last10_diff": diff(f10_j1, f10_j2),
        "form_last25_diff": diff(f25_j1, f25_j2),
        "form_surface_last5_diff": diff(fs5_j1, fs5_j2),
        "form_surface_last10_diff": diff(fs10_j1, fs10_j2),
        "rest_days_diff": diff(rest_j1, rest_j2),
        "fatigue_diff": diff(fat_j1, fat_j2),
    }
    return pd.DataFrame([row])

def predict_match(j1, j2, fecha, surface, level="A", best_of=3, rnd="R32"):
    model = joblib.load(MODEL)
    with open(META) as f: meta = json.load(f)
    feature_order = [c for c in meta["feature_order"] if c != "y"]  # seguridad

    X = build_features(j1, j2, fecha, surface, level, best_of, rnd)
    # respeta orden (y aseguramos que faltantes existan)
    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_order]
    proba = model.predict_proba(X.values)[:,1][0]
    return proba

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--j1", required=True)
    ap.add_argument("--j2", required=True)
    ap.add_argument("--fecha", required=True, help="YYYY-MM-DD")
    ap.add_argument("--surface", required=True, choices=["hard","clay","grass","carpet"])
    ap.add_argument("--level", default="A")
    ap.add_argument("--best_of", type=int, default=3)
    ap.add_argument("--round", default="R32")
    args = ap.parse_args()

    p = predict_match(args.j1, args.j2, args.fecha, args.surface, args.level, args.best_of, args.round)
    print(f"Probabilidad de que {args.j1} venza a {args.j2}: {p:.3f}")
