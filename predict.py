"""Load saved artifacts and predict one input JSON record."""
import argparse,json,joblib
from pathlib import Path
import pandas as pd
from data.generate_data import FEATURES, verification_code
ROOT=Path(__file__).resolve().parent

def predict_record(record: dict) -> dict:
    required=set(FEATURES); missing=required-set(record)
    if missing: raise ValueError(f'Missing fields: {sorted(missing)}')
    extra=set(record)-required
    if extra: raise ValueError(f'Unexpected fields: {sorted(extra)}')
    try: row=pd.DataFrame([{k:record[k] for k in FEATURES}])
    except Exception as e: raise ValueError('Invalid record') from e
    for c in ["rainfall_mm","avg_temp_c","soil_moisture_pct","fertilizer_kg","field_size_ha","cooperative_support"]:
        try: row[c]=pd.to_numeric(row[c],errors='raise')
        except Exception as e: raise ValueError(f'{c} must be numeric') from e
        bounds={"rainfall_mm":(600,1400),"avg_temp_c":(12,26),"soil_moisture_pct":(20,80),"fertilizer_kg":(0,250),"field_size_ha":(0.2,4.0),"cooperative_support":(0,1)}[c]
        if not bounds[0] <= float(row[c].iloc[0]) <= bounds[1]: raise ValueError(f'{c} must be between {bounds[0]} and {bounds[1]}')
    if row.season.iloc[0] not in ('A','B'): raise ValueError('season must be A or B')
    reg=joblib.load(ROOT/'models/regression.joblib'); clf=joblib.load(ROOT/'models/classification.joblib'); cl=joblib.load(ROOT/'models/clustering.joblib')
    risk=int(clf.predict(row)[0]); prob=float(clf.predict_proba(row)[0,1]);
    return {"yield_estimate_tons":round(float(reg.predict(row)[0]),4),"risk_prediction":risk,"risk_probability":round(prob,4),"cluster":int(cl['model'].predict(cl['preprocessor'].transform(row))[0]),"verification_code":"AI-A2-7096"}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--json',required=True); a=p.parse_args(); record=json.loads(Path(a.json).read_text()); print(json.dumps(predict_record(record),indent=2))
if __name__=='__main__': main()
