# src/pipeline/enriquecer_forma_superficie.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------
# Este script vive en: repo_root/src/pipeline/enriquecer_forma_superficie.py
# repo_root = parents[2]
ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "artifacts" / "datasets" / "partidos_enriquecidos_v2.csv"
OUT_PATH = ROOT / "artifacts" / "datasets" / "partidos_enriquecidos_final.csv"


def winrate_last_n(history: List[int], n: int) -> float:
    """Media de victorias de las últimas n observaciones. Requiere >=3 muestras."""
    arr = history[-n:]
    return float(np.mean(arr)) if len(arr) >= 3 else np.nan


def main():
    if not IN_PATH.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {IN_PATH}")

    df = pd.read_csv(IN_PATH, low_memory=False)
    if df.empty:
        raise RuntimeError(f"{IN_PATH} está vacío.")

    # Tipos y orden temporal
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"]).copy()
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)

    # Normaliza superficie a string para claves del historial
    df["surface"] = df["surface"].fillna("").astype(str)

    # Historial por (player_id, surface): lista de 1/0 con resultados previos
    hist: Dict[Tuple[int, str], List[int]] = {}

    def _get(key: Tuple[int, str]) -> List[int]:
        if key not in hist:
            hist[key] = []
        return hist[key]

    wf5_1: List[float] = []
    wf5_2: List[float] = []
    wf10_1: List[float] = []
    wf10_2: List[float] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Forma por superficie"):
        s = row.surface or ""
        k1 = (int(row.winner_id), s)
        k2 = (int(row.loser_id), s)
        h1 = _get(k1)
        h2 = _get(k2)

        # Forma previa
        wf5_1.append(winrate_last_n(h1, 5))
        wf5_2.append(winrate_last_n(h2, 5))
        wf10_1.append(winrate_last_n(h1, 10))
        wf10_2.append(winrate_last_n(h2, 10))

        # Actualización posterior al partido
        h1.append(1)
        h2.append(0)
        hist[k1], hist[k2] = h1, h2

    # Nuevas columnas
    df["winner_form_surface_last5"] = wf5_1
    df["loser_form_surface_last5"] = wf5_2
    df["winner_form_surface_last10"] = wf10_1
    df["loser_form_surface_last10"] = wf10_2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Guardado: {OUT_PATH} (filas: {len(df)})")


if __name__ == "__main__":
    main()
