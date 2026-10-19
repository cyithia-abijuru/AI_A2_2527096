"""One-command reproducible assignment runner."""
import argparse, json, hashlib
from pathlib import Path
import pandas as pd
from data.generate_data import generate_dataset, verification_code, fingerprint
from src.pipeline import validate_and_clean, train_all
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('--student-id',required=True); p.add_argument('--seed',type=int,required=True); a=p.parse_args()
    digits=''.join(ch for ch in a.student_id.split('@')[0] if ch.isdigit())
    if len(digits)<4 or int(digits[-4:])!=a.seed: raise SystemExit('ERROR: seed must equal the last four digits of the student registration username/ID.')
    code=verification_code(a.seed); print(f'Student: ABIJURU CYNTHIA | ID: 25/27096 | Username: {a.student_id}'); print(f'Verification code: {code}')
    data_path=ROOT/'data/AI_A2_2527096.csv'; raw=generate_dataset(a.seed,data_path); fp=fingerprint(data_path)
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True); (artifacts/'data_fingerprint.txt').write_text(f'SHA-256: {fp}\nVerification code: {code}\n')
    clean,report=validate_and_clean(raw); (artifacts/'data_validation.json').write_text(json.dumps(report,indent=2))
    result=train_all(clean,a.seed,artifacts,a.student_id)
    print(f'Dataset rows: {len(raw)} raw; {len(clean)} after cleaning'); print(f'SHA-256: {fp}'); print('Artifacts regenerated in artifacts/ and models/.'); print('Regression selected:',result['regression']['selected_model']); print('Classification selected:',result['classification']['selected_model']); print('Selected K:',result['clustering']['selected_K'])
if __name__=='__main__': main()
