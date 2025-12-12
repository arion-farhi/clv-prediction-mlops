
import functions_framework
import pandas as pd
import requests
from google.cloud import storage, bigquery
from evidently import Report
from evidently.presets import DataDriftPreset

@functions_framework.http
def check_drift(request):
    """Check for drift and trigger retrain if detected."""
    
    # Load reference data
    client = storage.Client()
    bucket = client.bucket("clv-prediction-data")
    bucket.blob("features/clv_features.parquet").download_to_filename("/tmp/reference.parquet")
    reference = pd.read_parquet("/tmp/reference.parquet")
    
    # Load recent predictions from BigQuery
    bq_client = bigquery.Client()
    query = """
        SELECT recency, frequency, monetary, prediction
        FROM `clv-predictions-mlops.retail_data.prediction_logs`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    """
    current = bq_client.query(query).to_dataframe()
    
    if len(current) < 10:
        return {"status": "skipped", "reason": "Not enough data", "rows": len(current)}
    
    # Run Evidently drift check
    ref_subset = reference[["recency_days", "frequency", "monetary"]].rename(columns={"recency_days": "recency"})
    cur_subset = current[["recency", "frequency", "monetary"]]
    
    drift_report = Report([DataDriftPreset()])
    result = drift_report.run(reference_data=ref_subset, current_data=cur_subset)
    
    drift_detected = result.dict()["metrics"][0]["result"]["dataset_drift"]
    
    if drift_detected:
        # Call the retrain function
        retrain_url = "https://clv-retrain-trigger-cpvl5opmca-uc.a.run.app"
        response = requests.post(retrain_url)
        return {"status": "drift_detected", "retrain_triggered": True, "retrain_response": response.status_code}
    
    return {"status": "no_drift", "retrain_triggered": False}
