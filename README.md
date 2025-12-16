# Customer Lifetime Value Prediction - MLOps Pipeline

**[Live Demo](https://clv-demo-674754622820.us-central1.run.app/)** | End-to-end MLOps system for predicting customer lifetime value on Google Cloud

---

## The Problem

E-commerce companies allocate marketing spend inefficiently without accurate predictions of which customers will generate long-term revenue. This leads to:
- **Wasted acquisition spend** → targeting low-value customers
- **Missed opportunities** → under-investing in high-value segments
- **Reactive strategies** → no forward-looking customer valuation

## The Solution

Built a hybrid deep neural network combining RFM behavioral features with Hugging Face text embeddings from product descriptions, predicting 12-month CLV. Implemented a production MLOps pipeline with automated drift detection, retraining triggers, and A/B testing infrastructure - all deployed on Google Cloud Platform.

---

## Results

| Metric | Value |
|--------|-------|
| R² Score | 0.735 |
| Median Absolute Error | $429 |
| Predictions within $1,000 | 68.6% |
| Tuning Improvement | 27% MAE reduction |

---

## Key Findings & Learnings

### 1. Hybrid Features Outperformed RFM Alone
Adding Hugging Face embeddings from product descriptions captured purchase preference patterns that traditional RFM metrics missed. The 384-dimensional embeddings contributed meaningful signal to the final predictions.

### 2. Robust Metrics Tell a Better Story
MAE ($1,449) was inflated by extreme outliers - customers with $280K+ CLV. Median AE ($429) and R² (0.735) better represent model performance for typical customers. Choosing the right metric matters for stakeholder communication.

### 3. Vizier Tuning Delivered Significant Gains
Vertex AI Vizier reduced MAE by 27% (from $1,987 to $1,449) through 15 trials of random search. Best hyperparameters: 201/74 units, 0.25 dropout, 0.0027 learning rate.

### 4. Demo Apps Need Realistic Baselines
Exposing only 6 of 396 features in the Streamlit demo required using median customer values for the remaining 390 features (including all embeddings). Without this baseline, predictions were nonsensical.

---

## Architecture
```
                           CLV PREDICTION MLOPS PIPELINE

==============================================================================
                                 DATA LAYER
==============================================================================

  +-----------+     +-----------+     +-----------+     +-----------+
  |  BigQuery |---->|  Dataproc |---->|  Hugging  |---->|    GCS    |
  | (Raw Data)|     | (PySpark) |     |   Face    |     |(Features) |
  +-----------+     +-----------+     +-----------+     +-----+-----+
                                                              |
                                                              v
==============================================================================
                                  ML LAYER
==============================================================================

  +-----------+     +-----------+     +-----------+     +-----------+
  |  Vertex   |---->|  Vertex   |---->|   Model   |---->|    GKE    |
  |  Vizier   |     | Pipeline  |     | Registry  |     |  (Prod)   |
  +-----------+     +-----+-----+     +-----+-----+     +-----------+
                          |                 |
                          |                 |           +-----------+
                          |                 +---------->| Cloud Run |
                          |                             |(Demo+A/B) |
                          |                             +-----+-----+
                          |                                   |
                          | Retrain                           |
                          |                                   v
==============================================================================
                             MONITORING LAYER
==============================================================================

  +-----------+     +-----------+     +-----------+     +-----------+
  |   Cloud   |---->| Evidently |---->|   Cloud   |<----|  BigQuery |
  | Scheduler |     |  (Drift)  |     | Function  |     |  (Logs)   |
  +-----------+     +-----------+     +-----+-----+     +-----------+
                                            |
                                            v
                                      +-----------+
                                      |  Vertex   |
                                      | Pipeline  |
                                      +-----------+
```

---

## Pipeline Components

### Vertex AI Vizier Hyperparameter Tuning
15 trials of random search optimization across learning rate, dropout, and layer sizes. Achieved 27% MAE improvement over baseline:

![Vizier Study](screenshots/vizier-study.png)

### Vertex AI Pipeline Orchestration
Four-step pipeline: Load Data → Train Model → Evaluate → Conditional Registration. Models only registered if MAE beats threshold:

![Pipeline DAG](screenshots/pipeline-dag.png)

### GKE Production Deployment
TensorFlow Serving container deployed on GKE Autopilot with LoadBalancer endpoint for production inference:

![GKE Deployment](screenshots/gke-deployment.png)

### Cloud Run Demo Application
Streamlit app with real-time predictions, model performance metrics, and architecture visualization:

![Cloud Run](screenshots/cloud-run.png)

### A/B Testing Infrastructure
Cloud Run traffic splitting enables gradual rollout of new model versions (80/20 split shown):

![A/B Testing](screenshots/ab-testing.png)

### Evidently AI Drift Monitoring
Monitors feature distributions using Wasserstein distance. Compares production data against training baseline:

![Evidently Drift](screenshots/evidently-drift.png)

### Cloud Functions Retraining Trigger
HTTP-triggered function that submits pipeline jobs. Called by drift check when distribution shift detected:

![Drift Check Function](screenshots/drift-check.png)

### Cloud Scheduler Automation
Scheduled weekly drift checks that trigger the Evidently analysis and conditional retraining:

![Cloud Scheduler](screenshots/cloud-scheduler.png)

### BigQuery Prediction Logging
All predictions logged with features and timestamps for monitoring and drift detection:

![BigQuery Logs](screenshots/bigquery-logs.png)

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Modeling** | TensorFlow, Hybrid NN, Hugging Face Embeddings, Integrated Gradients |
| **Data Processing** | PySpark, Dataproc, BigQuery |
| **Orchestration** | Vertex AI Pipelines, Vertex AI Vizier |
| **Deployment** | GKE, Cloud Run, A/B Traffic Splitting |
| **Monitoring** | Evidently AI, BigQuery Logging, Cloud Scheduler |
| **Automation** | Cloud Functions, Cloud Build CI/CD |

---

## Project Structure
```
clv-prediction-mlops/
├── notebooks/
│   ├── 01-data-preparation.ipynb      # BigQuery → PySpark → Hugging Face
│   ├── 02-nn-training.ipynb           # Baseline hybrid NN
│   ├── 03-hyperparameter-tuning.ipynb # Vizier optimization
│   ├── 04-pipeline-orchestration.ipynb # Vertex AI Pipeline
│   └── 05-monitoring-deployment.ipynb  # GKE, Cloud Run, Evidently
├── cloud_functions/
│   ├── retrain_trigger/               # Triggers pipeline retraining
│   └── drift_check/                   # Runs Evidently drift detection
├── streamlit_app/
│   ├── app.py                         # Demo application
│   ├── Dockerfile
│   └── requirements.txt
├── screenshots/
├── cloudbuild.yaml                    # CI/CD configuration
└── README.md
```

---

## Dataset

**UCI Online Retail II** - Real transactional data from a UK online retailer (2009-2011).

| Attribute | Value |
|-----------|-------|
| Customers | 4,266 |
| Features | 396 (12 numerical + 384 embeddings) |
| Target | 12-month CLV |

**Feature Engineering:**
- RFM metrics (recency, frequency, monetary)
- Behavioral features (tenure, unique products, orders per month)
- Hugging Face `all-MiniLM-L6-v2` embeddings from product descriptions

---

## The Journey

This project demonstrates production ML engineering beyond model training - automated pipelines, model versioning, drift detection, and multi-environment deployment.

Some unexpected learnings along the way:
- **Outliers dominate CLV distributions.** A few whale customers with $100K+ CLV skew MAE dramatically. Median AE and percentile-based metrics give a clearer picture of typical model performance.
- **PySpark on Dataproc is overkill for 4K customers** but demonstrates the pattern for scaling to millions. The same code would work on a 100M customer dataset.
- **Demo apps are harder than they look.** Exposing partial features while maintaining realistic predictions required careful baseline imputation with median values.

**Production considerations:**
- Pipeline MAE threshold ($2,000) was exceeded due to NN randomness; production would compare against current champion model
- Cloud Scheduler deleted after screenshot to avoid costs; in production would run weekly
- GKE cluster deleted after deployment proof; Cloud Run sufficient for demo traffic

---

## Author

**Arion Farhi** - [GitHub](https://github.com/arion-farhi) | [LinkedIn](https://linkedin.com/in/arion-farhi)
