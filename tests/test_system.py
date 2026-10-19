import json, subprocess, sys
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from data.generate_data import generate_dataset
from src.pipeline import validate_and_clean
from predict import predict_record
from app import app
ROOT=Path(__file__).resolve().parents[1]
client=TestClient(app)
VALID=json.loads((ROOT/'samples/valid.json').read_text())

def test_dataset_reproducibility():
    a=generate_dataset(7096); b=generate_dataset(7096); pd.testing.assert_frame_equal(a,b)
def test_cleaning_removes_duplicates_and_records_missing():
    raw=generate_dataset(7096); clean,report=validate_and_clean(raw)
    assert report['duplicates_removed']==3 and len(clean)==len(raw)-3
    assert report['missing_before_cleaning']['soil_moisture_pct']>0
def test_valid_inference():
    # Models are created by run_all.py before tests are run.
    result=predict_record(VALID)
    assert 'yield_estimate_tons' in result and result['verification_code']=='AI-A2-7096'
def test_missing_field_rejected():
    bad=dict(VALID); bad.pop('season')
    with pytest.raises(ValueError,match='Missing fields'): predict_record(bad)
def test_wrong_type_rejected():
    bad=dict(VALID); bad['rainfall_mm']='rain'
    with pytest.raises(ValueError,match='must be numeric'): predict_record(bad)
def test_range_error_rejected():
    bad=dict(VALID); bad['rainfall_mm']=1500
    with pytest.raises(ValueError,match='between 600 and 1400'): predict_record(bad)
def test_health_endpoint():
    assert client.get('/health').status_code==200 and client.get('/health').json()['status']=='ok'
def test_api_predict_valid():
    r=client.post('/predict',json=VALID); assert r.status_code==200 and 'risk_probability' in r.json()
def test_api_predict_invalid_range():
    bad=dict(VALID); bad['rainfall_mm']=1500
    assert client.post('/predict',json=bad).status_code==422
