
import functions_framework
from google.cloud import aiplatform

@functions_framework.http
def trigger_retrain(request):
    """HTTP Cloud Function to trigger pipeline retraining."""
    
    # Initialize Vertex AI
    aiplatform.init(
        project="clv-predictions-mlops",
        location="us-central1"
    )
    
    # Submit pipeline job
    job = aiplatform.PipelineJob(
        display_name="clv-pipeline-retrain",
        template_path="gs://clv-prediction-data/pipeline_root/clv_pipeline.json",
        pipeline_root="gs://clv-prediction-data/pipeline_root",
        parameter_values={
            "bucket_name": "clv-prediction-data",
            "project_id": "clv-predictions-mlops",
            "region": "us-central1",
            "units_1": 201,
            "units_2": 74,
            "dropout": 0.2478,
            "learning_rate": 0.0027,
            "mae_threshold": 2500.0
        }
    )
    
    job.submit()
    
    return {"status": "Pipeline triggered", "job_name": job.display_name}
