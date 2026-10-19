"""Reproducible synthetic potato-harvest data generator."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "season", "cooperative_support"]
ALL_COLUMNS = FEATURES + ["yield_tons", "high_risk"]

def verification_code(seed: int) -> str:
    return f"AI-A2-{seed:04d}"

def generate_dataset(seed: int, output_path: str | Path | None = None) -> pd.DataFrame:
    if not 0 <= seed <= 9999:
        raise ValueError("Seed must be the last four registration digits (0000–9999).")
    rng = np.random.default_rng(seed)
    n = 240 + (seed % 41)
    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    moisture = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], n)
    support = rng.integers(0, 2, n)
    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, n)
    yield_tons = (0.004*rainfall + 0.06*moisture + 0.02*fertilizer + 1.30*size - 0.25*np.abs(temp-18) + 0.70*support + season_effect + noise)
    yield_tons = np.clip(yield_tons, 0.50, None)
    df = pd.DataFrame({"rainfall_mm":rainfall,"avg_temp_c":temp,"soil_moisture_pct":moisture,"fertilizer_kg":fertilizer,"field_size_ha":size,"season":season,"cooperative_support":support,"yield_tons":yield_tons})
    # Risk is defined using yield per hectare, before deliberate corruption.
    per_ha = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (per_ha < per_ha.median()).astype(int)
    m = round(0.05*n)
    moisture_idx = rng.choice(n, size=m, replace=False)
    fertilizer_idx = rng.choice(n, size=m, replace=False)
    df.loc[moisture_idx, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_idx, "fertilizer_kg"] = np.nan
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    if output_path is not None:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.4f")
    return df

def fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--output',default='data/AI_A2_2527096.csv'); a=p.parse_args()
    data=generate_dataset(a.seed,a.output)
    print(f"Verification code: {verification_code(a.seed)}")
    print(f"Rows including deliberate duplicates: {len(data)}")
    print(f"Wrote: {a.output}")
