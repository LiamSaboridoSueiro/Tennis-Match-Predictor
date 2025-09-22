# src/pipeline/enriquecer_datos.py
from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------
# Este script vive en: repo_root/src/pipeline/enriquecer_datos.py
# repo_root = parents[2]
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "artifacts" / "datasets" / "partidos_enriquecidos.csv"

# Habilitar import de utilidades locales
import sys  # noqa: E402

sys.path.append(str(ROOT / "src"))
try:
    from utils import normalizar_nombre  # función de normalización de nombres
except Exception:
    def normalizar_nombre(s):  # fallback mínimo
        return str(s).strip()


# --------------------------------------------------------------------
# Columnas objetivo y alias
# --------------------------------------------------------------------
TARGET_COLS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "winner_hand",
    "loser_hand",
    "winner_ht",
    "loser_ht",
    "winner_age",
    "loser_age",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]

ALIASES = {
    "winner_player_id": "winner_id",
    "loser_player_id": "loser_id",
    "tourney_date_str": "tourney_date",
    "minutes_total": "minutes",
}

# --------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------
def _discover_csv_files(data_dir: Path) -> List[Path]:
    """Devuelve solo los CSV en el directorio raíz de data/ (no recursivo)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_dir}")
    files = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {data_dir}")
    return files



def md5_id_from_name(name: str) -> int:
    s = normalizar_nombre(name)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:10], 16)


def parse_tourney_date(series: pd.Series) -> pd.Series:
    """Parsea fechas tipo 20250131 o variantes mixtas."""
    s_digits = series.astype(str).str.replace(r"[^\d]", "", regex=True)
    mask8 = s_digits.str.len() == 8
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if mask8.any():
        out.loc[mask8] = pd.to_datetime(s_digits.loc[mask8], format="%Y%m%d", errors="coerce")
    if (~mask8).any():
        out.loc[~mask8] = pd.to_datetime(series.loc[~mask8], errors="coerce")
    return out


def get_round_weight(r):
    r = str(r).strip().upper() if pd.notna(r) else ""
    base = {
        "R128": 0.95,
        "R64": 1.00,
        "R32": 1.00,
        "R16": 1.05,
        "QF": 1.08,
        "SF": 1.12,
        "F": 1.18,
    }
    return base.get(r, 1.0)


def get_k(round_name, best_of):
    k0 = 32.0
    w_round = get_round_weight(round_name)
    w_bo = 1.0 if pd.isna(best_of) or int(best_of) == 3 else 1.1  # ligeramente mayor en BO5
    return k0 * w_round * w_bo


def _coerce_numeric(df: pd.DataFrame, cols, to_float=False):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if to_float:
            df[c] = df[c].astype(float)
    return df


def load_all(csv_files: List[Path]) -> pd.DataFrame:
    """Carga y normaliza todos los CSV para asegurar TARGET_COLS."""
    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            # Algunos CSV enormes o corruptos pueden fallar; se omiten
            continue

        existing = set(df.columns)
        # Renombrar alias si procede
        for src, dst in ALIASES.items():
            if src in existing and dst not in existing:
                df = df.rename(columns={src: dst})

        # Añadir faltantes para encajar TARGET_COLS
        for c in TARGET_COLS:
            if c not in df.columns:
                df[c] = np.nan

        # Normalizaciones básicas
        df["winner_name"] = df["winner_name"].astype(str).map(normalizar_nombre)
        df["loser_name"] = df["loser_name"].astype(str).map(normalizar_nombre)
        df["surface"] = df["surface"].fillna("").astype(str)

        frames.append(df[TARGET_COLS].copy())

    if not frames:
        raise FileNotFoundError(f"No se pudieron cargar CSV válidos en {DATA_DIR}")

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------
# Proceso principal
# --------------------------------------------------------------------
def main():
    csv_files = _discover_csv_files(DATA_DIR)
    print(f"[enriquecer_datos] Archivos CSV detectados: {len(csv_files)}")

    # Carga y tipado
    df = load_all(csv_files)
    df["tourney_date"] = parse_tourney_date(df["tourney_date"])

    num_ints = [
        "winner_id",
        "loser_id",
        "match_num",
        "draw_size",
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points",
        "best_of",
    ]
    _coerce_numeric(df, num_ints, to_float=False)

    num_floats = [
        "winner_ht",
        "loser_ht",
        "winner_age",
        "loser_age",
        "minutes",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]
    _coerce_numeric(df, num_floats, to_float=True)

    # IDs de fallback por nombre
    mw, ml = df["winner_id"].isna(), df["loser_id"].isna()
    if mw.any():
        df.loc[mw, "winner_id"] = df.loc[mw, "winner_name"].apply(md5_id_from_name)
    if ml.any():
        df.loc[ml, "loser_id"] = df.loc[ml, "loser_name"].apply(md5_id_from_name)

    # Filtrado por fecha válida
    before = len(df)
    df = df.dropna(subset=["tourney_date"]).copy()
    print(f"[enriquecer_datos] Filas descartadas por fecha NaN: {before - len(df)}")

    # Orden temporal estable
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"], kind="mergesort").reset_index(
        drop=True
    )

    # ----------------------------------------------------------------
    # Winrate previo por superficie
    # ----------------------------------------------------------------
    win_surface_1, win_surface_2 = [], []
    hist_surface = {}  # (player_id, surface) -> [wins]

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Winrate superficie"):
        p1, p2 = int(row.winner_id), int(row.loser_id)
        surf = row.surface or ""
        k1, k2 = (p1, surf), (p2, surf)
        w1 = hist_surface.get(k1, [])
        w2 = hist_surface.get(k2, [])
        win_surface_1.append(np.mean(w1) if w1 else np.nan)
        win_surface_2.append(np.mean(w2) if w2 else np.nan)
        hist_surface[k1] = w1 + [1]
        hist_surface[k2] = w2 + [0]

    # ----------------------------------------------------------------
    # Elo con decaimiento (global y por superficie)
    # ----------------------------------------------------------------
    TAU_G = 240.0  # días
    TAU_S = 180.0  # días

    elo_global = {}  # player_id -> rating
    elo_surface = {}  # (player_id, surface) -> rating
    last_date_g = {}  # player_id -> last date
    last_date_s = {}  # (player_id, surface) -> last date

    elo_winner_g, elo_loser_g = [], []
    elo_winner_s, elo_loser_s = [], []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Elo previos"):
        p1, p2 = int(row.winner_id), int(row.loser_id)
        surf = row.surface or ""
        d = row.tourney_date
        best_of = row.best_of
        rnd = row.round

        def decay_rating(r, last_d, tau):
            if last_d is None:
                return r
            dt = (d - last_d).days
            if dt <= 0 or pd.isna(dt):
                return r
            lam = math.exp(-dt / tau)
            return 1500.0 + (r - 1500.0) * lam

        r1g = elo_global.get(p1, 1500.0)
        r2g = elo_global.get(p2, 1500.0)
        r1s = elo_surface.get((p1, surf), 1500.0)
        r2s = elo_surface.get((p2, surf), 1500.0)

        r1g_eff = decay_rating(r1g, last_date_g.get(p1), TAU_G)
        r2g_eff = decay_rating(r2g, last_date_g.get(p2), TAU_G)
        r1s_eff = decay_rating(r1s, last_date_s.get((p1, surf)), TAU_S)
        r2s_eff = decay_rating(r2s, last_date_s.get((p2, surf)), TAU_S)

        elo_winner_g.append(r1g_eff)
        elo_loser_g.append(r2g_eff)
        elo_winner_s.append(r1s_eff)
        elo_loser_s.append(r2s_eff)

        K = get_k(rnd, best_of)

        # actualización global
        exp1g = 1.0 / (1.0 + 10 ** ((r2g_eff - r1g_eff) / 400))
        r1g_new = r1g_eff + K * (1 - exp1g)
        r2g_new = r2g_eff - K * (1 - exp1g)

        # actualización superficie
        exp1s = 1.0 / (1.0 + 10 ** ((r2s_eff - r1s_eff) / 400))
        r1s_new = r1s_eff + K * (1 - exp1s)
        r2s_new = r2s_eff - K * (1 - exp1s)

        elo_global[p1], elo_global[p2] = r1g_new, r2g_new
        elo_surface[(p1, surf)], elo_surface[(p2, surf)] = r1s_new, r2s_new
        last_date_g[p1], last_date_g[p2] = d, d
        last_date_s[(p1, surf)], last_date_s[(p2, surf)] = d, d

    # ----------------------------------------------------------------
    # H2H previo (diferencia de partidos ganados entre ambos)
    # ----------------------------------------------------------------
    h2h_diff, h2h = [], {}
    for row in df.itertuples(index=False):
        p1, p2 = int(row.winner_id), int(row.loser_id)
        a, b = (p1, p2) if p1 < p2 else (p2, p1)
        dct = h2h.get((a, b), {a: 0, b: 0})
        h2h_diff.append(dct.get(p1, 0) - dct.get(p2, 0))
        dct[p1] = dct.get(p1, 0) + 1
        h2h[(a, b)] = dct

    # ----------------------------------------------------------------
    # Ensamble final y guardado
    # ----------------------------------------------------------------
    df["winner_winrate_surface"] = win_surface_1
    df["loser_winrate_surface"] = win_surface_2
    df["winner_elo"] = elo_winner_g
    df["loser_elo"] = elo_loser_g
    df["winner_elo_surface"] = elo_winner_s
    df["loser_elo_surface"] = elo_loser_s
    df["h2h_diff"] = h2h_diff

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[enriquecer_datos] Guardado: {OUT_PATH} (filas: {len(df)})")


if __name__ == "__main__":
    main()
