"""FastAPI inference service for the Musanze Harvest Intelligence prototype."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from predict import predict_record
app=FastAPI(title='Musanze Harvest Intelligence Service',version='1.0.0',description='Classroom decision-support prototype trained on synthetic data.')
class HarvestInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    rainfall_mm: float=Field(ge=600,le=1400)
    avg_temp_c: float=Field(ge=12,le=26)
    soil_moisture_pct: float=Field(ge=20,le=80)
    fertilizer_kg: float=Field(ge=0,le=250)
    field_size_ha: float=Field(ge=0.2,le=4.0)
    season: str=Field(pattern='^[AB]$')
    cooperative_support: int=Field(ge=0,le=1)
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/model-info')
def model_info(): return {'service':'Musanze Harvest Intelligence Service','version':'1.0.0','verification_code':'AI-A2-7096','data_status':'synthetic','tasks':['regression','classification','clustering'],'human_review_required':True}
@app.post('/predict')
def predict(payload: HarvestInput):
    try: return predict_record(payload.model_dump())
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
