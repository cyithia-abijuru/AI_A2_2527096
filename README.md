# Musanze Harvest Intelligence Service

**Student:** ABIJURU CYNTHIA  
**Registration number:** 25/27096  
**Institutional username:** ug2527096@ines.ac.rw  
**Verification code:** AI-A2-7096  
**Operating system:** Linux Ubuntu

## Problem
A potato cooperative in Musanze needs a reproducible classroom prototype to estimate harvest quantity, flag high production risk, and group farms with similar input conditions. The data is entirely synthetic and is not evidence about real farms.

## Setup (Ubuntu)
```bash
cd AI_A2_2527096
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce data, models, metrics and plots
```bash
python run_all.py --student-id ug2527096@ines.ac.rw --seed 7096
```
The seed must equal the last four digits of the student registration ID. The script regenerates the CSV, validates and cleans data, trains regression/classification/clustering models, and regenerates artifacts.

## CLI prediction and tests
```bash
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json
pytest -q
```
The invalid example is expected to fail validation; this is intentional.

## Start API
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs`. Endpoints: `GET /health`, `GET /model-info`, `POST /predict`.

## Artifact guide
- `data/AI_A2_2527096.csv`: generated raw data, including deliberate missing values and three exact duplicate rows.
- `artifacts/data_fingerprint.txt`: SHA-256 of the generated CSV bytes.
- `artifacts/data_validation.json`: cleaning evidence.
- `artifacts/regression_metrics.json`, `classification_metrics.json`, `clustering_report.json`: metrics and selection reasons.
- `artifacts/test_predictions.csv`: held-out predictions, errors, risk probabilities and clusters.
- `artifacts/plots/`: residual evidence, confusion matrix and cluster plot.
- `models/`: saved pipelines and clustering artifacts.
- `docs/AI_A2_REGDIGITS_REPORT.pdf`: technical report.
- `evidence/`: templates/instructions for student-recorded evidence.

## Limitations and responsible use
The training data is synthetic, generated from the assignment formula. Metrics measure behavior on a held-out split of synthetic data and do not establish real-world accuracy. Do not use this prototype to determine access to finance, insurance, public services, or other consequential decisions. A qualified agronomist must review predictions. Real deployment requires representative, consented farm data, external validation, monitoring, and stakeholder review.

## Repository and commit
Repository URL: add your own repository URL after creating and pushing the repository.  
Final commit hash: add the actual commit hash after committing; do not invent one.
