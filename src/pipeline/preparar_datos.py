# src/pipeline/preparar_datos.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "datasets" / "partidos_enriquecidos_final.csv"
OUT_DATA = ROOT / "artifacts" / "datasets" / "dataset_modelo.csv"


# --------------------------------------------------------------------
# Utilidades de codificación y saneo
# --------------------------------------------------------------------
def safe_str_lower(x: object) -> str:
    if pd.isna(x):
        return ""
    try:
        return str(x).strip().lower()
    except Exception:
        return ""


def encode_surface(s):
    s = safe_str_lower(s)
    return {"hard": 0, "clay": 1, "grass": 2, "carpet": 3}.get(s, -1)


def encode_level(l):
    l = safe_str_lower(l).upper()
    return {"G": 0, "F": 1, "A": 2, "D": 3, "M": 1}.get(l, -1)


round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]


def encode_round(r):
    r = ("" if pd.isna(r) else str(r).strip().upper())
    try:
        return round_order.index(r)
    except Exception:
        return -1


def safe_num(x, default=np.nan):
    try:
        v = pd.to_numeric(x)
        return v
    except Exception:
        return default


def safe_diff(a, b):
    if pd.isna(a) and pd.isna(b):
        return np.nan
    return (a if pd.notna(a) else 0) - (b if pd.notna(b) else 0)


# --------------------------------------------------------------------
# Proceso principal
# --------------------------------------------------------------------
def main():
    if not SRC.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {SRC}")

    df = pd.read_csv(SRC, low_memory=False)
    if df.empty:
        raise RuntimeError(f"{SRC} está vacío. Revisa pasos previos del pipeline.")

    # Tipos base y orden temporal
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)

    # Comprobación suave: si no existen las columnas nuevas de servicio, avisa
    svc_cols_needed = [
        "winner_svc_ace_rate_5","loser_svc_ace_rate_5",
        "winner_svc_df_rate_5","loser_svc_df_rate_5",
        "winner_first_in_pct_5","loser_first_in_pct_5",
        "winner_first_win_pct_5","loser_first_win_pct_5",
        "winner_second_win_pct_5","loser_second_win_pct_5",
        "winner_bp_save_rate_5","loser_bp_save_rate_5",
    ]
    missing = [c for c in svc_cols_needed if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Faltan columnas de servicio recientes en el dataset de entrada: "
            + ", ".join(missing)
            + ". Ejecuta antes src/pipeline/enriquecer_datos_forma_y_descanso.py."
        )

    rows = []
    for row in df.itertuples(index=False):
        # lecturas defensivas
        surface_val = getattr(row, "surface", "")
        level_val = getattr(row, "tourney_level", "")
        best_of_val = getattr(row, "best_of", 3)
        round_val = getattr(row, "round", "")

        # dos vistas por partido (A=uno, B=otro)
        for AisWinner in [True, False]:
            if AisWinner:
                # A = ganador, B = perdedor
                A = {
                    "rank": getattr(row, "winner_rank", np.nan),
                    "rank_points": getattr(row, "winner_rank_points", np.nan),
                    "age": getattr(row, "winner_age", np.nan),
                    "ht": getattr(row, "winner_ht", np.nan),
                    "elo": getattr(row, "winner_elo", np.nan),
                    "elo_surface": getattr(row, "winner_elo_surface", np.nan),
                    "winrate_surface": getattr(row, "winner_winrate_surface", np.nan),
                    "form5": getattr(row, "winner_form_last5", np.nan),
                    "form10": getattr(row, "winner_form_last10", np.nan),
                    "form25": getattr(row, "winner_form_last25", np.nan),
                    "form_surf5": getattr(row, "winner_form_surface_last5", np.nan),
                    "form_surf10": getattr(row, "winner_form_surface_last10", np.nan),
                    "rest": getattr(row, "winner_rest_days", np.nan),
                    "fatigue": getattr(row, "winner_fatigue", np.nan),
                    # servicio recientes (rolling 5)
                    "svc_ace_r5": getattr(row, "winner_svc_ace_rate_5", np.nan),
                    "svc_df_r5": getattr(row, "winner_svc_df_rate_5", np.nan),
                    "first_in5": getattr(row, "winner_first_in_pct_5", np.nan),
                    "first_win5": getattr(row, "winner_first_win_pct_5", np.nan),
                    "second_win5": getattr(row, "winner_second_win_pct_5", np.nan),
                    "bp_save5": getattr(row, "winner_bp_save_rate_5", np.nan),
                }
                B = {
                    "rank": getattr(row, "loser_rank", np.nan),
                    "rank_points": getattr(row, "loser_rank_points", np.nan),
                    "age": getattr(row, "loser_age", np.nan),
                    "ht": getattr(row, "loser_ht", np.nan),
                    "elo": getattr(row, "loser_elo", np.nan),
                    "elo_surface": getattr(row, "loser_elo_surface", np.nan),
                    "winrate_surface": getattr(row, "loser_winrate_surface", np.nan),
                    "form5": getattr(row, "loser_form_last5", np.nan),
                    "form10": getattr(row, "loser_form_last10", np.nan),
                    "form25": getattr(row, "loser_form_last25", np.nan),
                    "form_surf5": getattr(row, "loser_form_surface_last5", np.nan),
                    "form_surf10": getattr(row, "loser_form_surface_last10", np.nan),
                    "rest": getattr(row, "loser_rest_days", np.nan),
                    "fatigue": getattr(row, "loser_fatigue", np.nan),
                    "svc_ace_r5": getattr(row, "loser_svc_ace_rate_5", np.nan),
                    "svc_df_r5": getattr(row, "loser_svc_df_rate_5", np.nan),
                    "first_in5": getattr(row, "loser_first_in_pct_5", np.nan),
                    "first_win5": getattr(row, "loser_first_win_pct_5", np.nan),
                    "second_win5": getattr(row, "loser_second_win_pct_5", np.nan),
                    "bp_save5": getattr(row, "loser_bp_save_rate_5", np.nan),
                }
                y = 1
            else:
                # A = perdedor, B = ganador
                A = {
                    "rank": getattr(row, "loser_rank", np.nan),
                    "rank_points": getattr(row, "loser_rank_points", np.nan),
                    "age": getattr(row, "loser_age", np.nan),
                    "ht": getattr(row, "loser_ht", np.nan),
                    "elo": getattr(row, "loser_elo", np.nan),
                    "elo_surface": getattr(row, "loser_elo_surface", np.nan),
                    "winrate_surface": getattr(row, "loser_winrate_surface", np.nan),
                    "form5": getattr(row, "loser_form_last5", np.nan),
                    "form10": getattr(row, "loser_form_last10", np.nan),
                    "form25": getattr(row, "loser_form_last25", np.nan),
                    "form_surf5": getattr(row, "loser_form_surface_last5", np.nan),
                    "form_surf10": getattr(row, "loser_form_surface_last10", np.nan),
                    "rest": getattr(row, "loser_rest_days", np.nan),
                    "fatigue": getattr(row, "loser_fatigue", np.nan),
                    "svc_ace_r5": getattr(row, "loser_svc_ace_rate_5", np.nan),
                    "svc_df_r5": getattr(row, "loser_svc_df_rate_5", np.nan),
                    "first_in5": getattr(row, "loser_first_in_pct_5", np.nan),
                    "first_win5": getattr(row, "loser_first_win_pct_5", np.nan),
                    "second_win5": getattr(row, "loser_second_win_pct_5", np.nan),
                    "bp_save5": getattr(row, "loser_bp_save_rate_5", np.nan),
                }
                B = {
                    "rank": getattr(row, "winner_rank", np.nan),
                    "rank_points": getattr(row, "winner_rank_points", np.nan),
                    "age": getattr(row, "winner_age", np.nan),
                    "ht": getattr(row, "winner_ht", np.nan),
                    "elo": getattr(row, "winner_elo", np.nan),
                    "elo_surface": getattr(row, "winner_elo_surface", np.nan),
                    "winrate_surface": getattr(row, "winner_winrate_surface", np.nan),
                    "form5": getattr(row, "winner_form_last5", np.nan),
                    "form10": getattr(row, "winner_form_last10", np.nan),
                    "form25": getattr(row, "winner_form_last25", np.nan),
                    "form_surf5": getattr(row, "winner_form_surface_last5", np.nan),
                    "form_surf10": getattr(row, "winner_form_surface_last10", np.nan),
                    "rest": getattr(row, "winner_rest_days", np.nan),
                    "fatigue": getattr(row, "winner_fatigue", np.nan),
                    "svc_ace_r5": getattr(row, "winner_svc_ace_rate_5", np.nan),
                    "svc_df_r5": getattr(row, "winner_svc_df_rate_5", np.nan),
                    "first_in5": getattr(row, "winner_first_in_pct_5", np.nan),
                    "first_win5": getattr(row, "winner_first_win_pct_5", np.nan),
                    "second_win5": getattr(row, "winner_second_win_pct_5", np.nan),
                    "bp_save5": getattr(row, "winner_bp_save_rate_5", np.nan),
                }
                y = 0

            rows.append(
                {
                    "tourney_date": getattr(row, "tourney_date", pd.NaT),
                    "surface_code": encode_surface(surface_val),
                    "level_code": encode_level(level_val),
                    "best_of": int(safe_num(best_of_val, 3)) if not pd.isna(best_of_val) else 3,
                    "round_code": encode_round(round_val),

                    "rank_diff": safe_diff(A["rank"], B["rank"]),
                    "rank_points_diff": safe_diff(A["rank_points"], B["rank_points"]),
                    "age_diff": safe_diff(A["age"], B["age"]),
                    "ht_diff": safe_diff(A["ht"], B["ht"]),

                    "elo_diff": safe_diff(A["elo"], B["elo"]),
                    "elo_surface_diff": safe_diff(A["elo_surface"], B["elo_surface"]),
                    "winrate_surface_diff": safe_diff(A["winrate_surface"], B["winrate_surface"]),
                    "form_last5_diff": safe_diff(A["form5"], B["form5"]),
                    "form_last10_diff": safe_diff(A["form10"], B["form10"]),
                    "form_last25_diff": safe_diff(A["form25"], B["form25"]),
                    "form_surface_last5_diff": safe_diff(A["form_surf5"], B["form_surf5"]),
                    "form_surface_last10_diff": safe_diff(A["form_surf10"], B["form_surf10"]),
                    "rest_days_diff": safe_diff(A["rest"], B["rest"]),
                    "fatigue_diff": safe_diff(A["fatigue"], B["fatigue"]),

                    # nuevas diferencias de servicio (rolling 5)
                    "svc_ace_rate_5_diff": safe_diff(A["svc_ace_r5"], B["svc_ace_r5"]),
                    "svc_df_rate_5_diff": safe_diff(A["svc_df_r5"], B["svc_df_r5"]),
                    "first_in_pct_5_diff": safe_diff(A["first_in5"], B["first_in5"]),
                    "first_win_pct_5_diff": safe_diff(A["first_win5"], B["first_win5"]),
                    "second_win_pct_5_diff": safe_diff(A["second_win5"], B["second_win5"]),
                    "bp_save_rate_5_diff": safe_diff(A["bp_save5"], B["bp_save5"]),

                    "y": y,
                }
            )

    X = pd.DataFrame(rows)

    if X.empty:
        raise RuntimeError("No se generaron filas para el dataset. Revisa columnas de entrada.")

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(OUT_DATA, index=False)
    print(f"Guardado: {OUT_DATA} (filas: {len(X)})")


if __name__ == "__main__":
    main()
