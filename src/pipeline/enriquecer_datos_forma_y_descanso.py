# src/pipeline/enriquecer_datos_forma_y_descanso.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------
# Este script vive en: repo_root/src/pipeline/enriquecer_datos_forma_y_descanso.py
# repo_root = parents[2]
ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "artifacts" / "datasets" / "partidos_enriquecidos.csv"
OUT_PATH = ROOT / "artifacts" / "datasets" / "partidos_enriquecidos_v2.csv"


def winrate_lastN(lst, N: int):
    arr = lst[-N:]
    return np.mean(arr) if len(arr) >= 3 else np.nan


def avg_nonan_lastN(lst, N: int, default=np.nan):
    arr = [x for x in lst[-N:] if pd.notna(x)]
    return np.mean(arr) if arr else default

def ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.divide(a, b, where=(b > 0))
    return np.nan_to_num(r, nan=np.nan)

def avg_lastN(arr, N):
    arr = [x for x in arr[-N:] if pd.notna(x)]
    return float(np.mean(arr)) if arr else np.nan

def main():
    if not IN_PATH.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {IN_PATH}")

    df = pd.read_csv(IN_PATH, low_memory=False)
    if df.empty:
        raise RuntimeError(f"{IN_PATH} está vacío.")

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"]).copy()
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    # Historiales por jugador
    hist = {}  # player_id -> dict con listas
    def _get(pid):
        if pid not in hist:
            hist[pid] = {
                "wins": [], "dates": [], "dur": [],
                "ace": [], "df": [], "svpt": [],
                "fst_in": [], "fst_won": [], "snd_won": [],
                "bp_saved": [], "bp_faced": [],
            }
        return hist[pid]

    # salidas
    last5_w1,last5_w2,last10_w1,last10_w2,last25_w1,last25_w2 = [],[],[],[],[],[]
    rest_days_1,rest_days_2,fatigue_1,fatigue_2 = [],[],[],[]

    # nuevas métricas de servicio
    svc_ace_r_1, svc_ace_r_2 = [], []
    svc_df_r_1,  svc_df_r_2  = [], []
    fst_in_pct_1, fst_in_pct_2 = [], []
    fst_win_pct_1, fst_win_pct_2 = [], []
    snd_win_pct_1, snd_win_pct_2 = [], []
    bp_save_r_1, bp_save_r_2 = [], []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Forma y descanso + servicio"):
        d = row.tourney_date
        p1, p2 = int(row.winner_id), int(row.loser_id)
        h1, h2 = _get(p1), _get(p2)

        # forma previa
        last5_w1.append(winrate_lastN(h1["wins"], 5))
        last5_w2.append(winrate_lastN(h2["wins"], 5))
        last10_w1.append(winrate_lastN(h1["wins"], 10))
        last10_w2.append(winrate_lastN(h2["wins"], 10))
        last25_w1.append(winrate_lastN(h1["wins"], 25))
        last25_w2.append(winrate_lastN(h2["wins"], 25))

        # descanso
        def days_since(lst):
            return np.nan if not lst else (d - lst[-1]).days
        rest_days_1.append(days_since(h1["dates"]))
        rest_days_2.append(days_since(h2["dates"]))

        # fatiga
        fatigue_1.append(avg_nonan_lastN(h1["dur"], 5, default=90.0))
        fatigue_2.append(avg_nonan_lastN(h2["dur"], 5, default=90.0))

        # servicio previo (rolling 5)
        def service_feats(H):
            ace_r = avg_lastN(ratio(H["ace"], H["svpt"]), 5)
            df_r  = avg_lastN(ratio(H["df"], H["svpt"]), 5)
            f_in  = avg_lastN(ratio(H["fst_in"], H["svpt"]), 5)
            f_win = avg_lastN(ratio(H["fst_won"], H["fst_in"]), 5)
            s_win = avg_lastN(ratio(H["snd_won"], (np.array(H["svpt"]) - np.array(H["fst_in"]))), 5)
            bp_r  = avg_lastN(ratio(H["bp_saved"], np.maximum(H["bp_faced"], 1)), 5)
            return ace_r, df_r, f_in, f_win, s_win, bp_r

        a1, d1, fi1, fw1, sw1, br1 = service_feats(h1)
        a2, d2, fi2, fw2, sw2, br2 = service_feats(h2)
        svc_ace_r_1.append(a1); svc_ace_r_2.append(a2)
        svc_df_r_1.append(d1);  svc_df_r_2.append(d2)
        fst_in_pct_1.append(fi1); fst_in_pct_2.append(fi2)
        fst_win_pct_1.append(fw1); fst_win_pct_2.append(fw2)
        snd_win_pct_1.append(sw1); snd_win_pct_2.append(sw2)
        bp_save_r_1.append(br1);  bp_save_r_2.append(br2)

        # actualizar posterior con los datos del partido actual
        h1["wins"].append(1); h2["wins"].append(0)
        h1["dates"].append(d); h2["dates"].append(d)
        dur = row.minutes if pd.notna(row.minutes) else np.nan
        h1["dur"].append(dur); h2["dur"].append(dur)

        # mapear columnas w_* para winner y l_* para loser
        h1["ace"].append(getattr(row, "w_ace", np.nan));  h2["ace"].append(getattr(row, "l_ace", np.nan))
        h1["df"].append(getattr(row, "w_df", np.nan));    h2["df"].append(getattr(row, "l_df", np.nan))
        h1["svpt"].append(getattr(row, "w_svpt", np.nan)); h2["svpt"].append(getattr(row, "l_svpt", np.nan))
        h1["fst_in"].append(getattr(row, "w_1stIn", np.nan)); h2["fst_in"].append(getattr(row, "l_1stIn", np.nan))
        h1["fst_won"].append(getattr(row, "w_1stWon", np.nan)); h2["fst_won"].append(getattr(row, "l_1stWon", np.nan))
        h1["snd_won"].append(getattr(row, "w_2ndWon", np.nan)); h2["snd_won"].append(getattr(row, "l_2ndWon", np.nan))
        h1["bp_saved"].append(getattr(row, "w_bpSaved", np.nan)); h2["bp_saved"].append(getattr(row, "l_bpSaved", np.nan))
        h1["bp_faced"].append(getattr(row, "w_bpFaced", np.nan)); h2["bp_faced"].append(getattr(row, "l_bpFaced", np.nan))

    # columnas existentes
    df["winner_form_last5"] = last5_w1; df["loser_form_last5"] = last5_w2
    df["winner_form_last10"] = last10_w1; df["loser_form_last10"] = last10_w2
    df["winner_form_last25"] = last25_w1; df["loser_form_last25"] = last25_w2
    df["winner_rest_days"] = rest_days_1; df["loser_rest_days"] = rest_days_2
    df["winner_fatigue"] = fatigue_1; df["loser_fatigue"] = fatigue_2

    # nuevas columnas de servicio (rolling 5)
    df["winner_svc_ace_rate_5"] = svc_ace_r_1; df["loser_svc_ace_rate_5"] = svc_ace_r_2
    df["winner_svc_df_rate_5"]  = svc_df_r_1;  df["loser_svc_df_rate_5"]  = svc_df_r_2
    df["winner_first_in_pct_5"] = fst_in_pct_1; df["loser_first_in_pct_5"] = fst_in_pct_2
    df["winner_first_win_pct_5"] = fst_win_pct_1; df["loser_first_win_pct_5"] = fst_win_pct_2
    df["winner_second_win_pct_5"] = snd_win_pct_1; df["loser_second_win_pct_5"] = snd_win_pct_2
    df["winner_bp_save_rate_5"] = bp_save_r_1; df["loser_bp_save_rate_5"] = bp_save_r_2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Guardado: {OUT_PATH} (filas: {len(df)})")


if __name__ == "__main__":
    main()
