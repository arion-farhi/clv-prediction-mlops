
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import joblib
from google.cloud import storage, bigquery
from datetime import datetime
import os

st.set_page_config(page_title="CLV Prediction MLOps", layout="wide")

# Initialize BigQuery client for logging
@st.cache_resource
def get_bq_client():
    try:
        return bigquery.Client(project="clv-predictions-mlops")
    except:
        return None

bq_client = get_bq_client()

def log_prediction(features_dict, prediction):
    """Log prediction to BigQuery for monitoring"""
    if bq_client is None:
        return
    try:
        table_id = "clv-predictions-mlops.retail_data.prediction_logs"
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "recency": features_dict.get("recency", 0),
            "frequency": features_dict.get("frequency", 0),
            "monetary": features_dict.get("monetary", 0),
            "prediction": float(prediction)
        }
        errors = bq_client.insert_rows_json(table_id, [row])
    except Exception as e:
        pass

# Load model, scaler, and baseline features from GCS
@st.cache_resource
def load_artifacts():
    try:
        client = storage.Client()
        bucket = client.bucket("clv-prediction-data")
        
        # Download model
        bucket.blob("models/clv_model_tuned.keras").download_to_filename("/tmp/model.keras")
        model = tf.keras.models.load_model("/tmp/model.keras")
        
        # Download scaler
        bucket.blob("models/clv_scaler.pkl").download_to_filename("/tmp/scaler.pkl")
        scaler = joblib.load("/tmp/scaler.pkl")
        
        # Download feature data for baseline
        bucket.blob("features/clv_features.parquet").download_to_filename("/tmp/features.parquet")
        df = pd.read_parquet("/tmp/features.parquet")
        feature_cols = [c for c in df.columns if c not in ["customer_id", "target_clv"]]
        baseline = df[feature_cols].median().values
        
        return model, scaler, baseline, feature_cols
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None, None, None, None

model, scaler, baseline_features, feature_cols = load_artifacts()

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.metric("Model Type", "Hybrid NN")
st.sidebar.metric("Median AE", "$429")
st.sidebar.metric("R²", "0.735")

st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
st.sidebar.markdown("""
- 12 numerical features (RFM+)
- 384 text embeddings (Hugging Face)
- Tuned via Vertex AI Vizier
""")

# Main content
st.title("Customer Lifetime Value Prediction")
st.markdown("**MLOps Pipeline Demo** - Hybrid Neural Network with Hugging Face Embeddings")

tab1, tab2, tab3, tab4 = st.tabs(["Predict CLV", "Model Performance", "Feature Importance", "Architecture"])

with tab1:
    st.header("Predict Customer Lifetime Value")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Features")
        
        recency = st.slider("Recency (days since last purchase)", 0, 365, 30)
        frequency = st.slider("Frequency (number of orders)", 1, 50, 25)
        avg_order = st.number_input("Average Order Value $", 0, 5000, 1000)
        tenure = st.slider("Customer Tenure (days)", 0, 365, 180)
        unique_products = st.slider("Unique Products Purchased", 1, 100, 50)
        
        # Calculate monetary from frequency * avg_order
        monetary = frequency * avg_order
        st.caption(f"Calculated Monetary (Frequency × Avg Order): ${monetary:,}")
    
    with col2:
        st.subheader("Prediction")
        
        if st.button("Predict CLV", type="primary"):
            if model is not None and baseline_features is not None:
                # Start with median customer baseline (includes real embeddings)
                features = baseline_features.copy()
                
                # Override with slider values
                if "recency_days" in feature_cols:
                    features[feature_cols.index("recency_days")] = recency
                if "frequency" in feature_cols:
                    features[feature_cols.index("frequency")] = frequency
                if "monetary" in feature_cols:
                    features[feature_cols.index("monetary")] = monetary
                if "avg_order_value" in feature_cols:
                    features[feature_cols.index("avg_order_value")] = avg_order
                if "customer_tenure_days" in feature_cols:
                    features[feature_cols.index("customer_tenure_days")] = tenure
                if "unique_products" in feature_cols:
                    features[feature_cols.index("unique_products")] = unique_products
                
                # Scale and predict with ACTUAL model
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled, verbose=0)[0][0]
                prediction = max(0, prediction)
                
                # Log to BigQuery
                log_prediction({
                    "recency": recency,
                    "frequency": frequency,
                    "monetary": monetary
                }, prediction)
                
                st.metric("Predicted 12-Month CLV", f"${prediction:,.0f}")
                
                # Customer segment
                if prediction > 5000:
                    st.success("🌟 High-Value Customer")
                elif prediction > 1000:
                    st.info("📈 Medium-Value Customer")
                else:
                    st.warning("📊 Low-Value Customer")
            else:
                st.error("Model not loaded")

with tab2:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Baseline vs Tuned MAE comparison
        tuning_data = {
            "Model": ["Baseline", "Tuned"],
            "MAE ($)": [1987, 1449]
        }
        fig = px.bar(tuning_data, x="Model", y="MAE ($)", 
                     title="Hyperparameter Tuning Impact (27% Improvement)",
                     color="Model", color_discrete_map={"Baseline": "gray", "Tuned": "green"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Performance Metrics")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Median Absolute Error | $429 |
        | R² Score | 0.735 |
        | Predictions within $1,000 | 68.6% |
        """)
        
        st.markdown("### Vizier Tuning")
        st.markdown("""
        - **Trials**: 15
        - **Algorithm**: Random Search
        - **Best params**: 201/74 units, 0.25 dropout, 0.0027 lr
        """)

with tab3:
    st.header("Feature Importance (Integrated Gradients)")
    
    importance_data = {
        "Feature": ["monetary", "unique_purchase_days", "frequency", "orders_per_month", 
                   "total_items_purchased", "unique_products", "avg_order_value", 
                   "customer_tenure_days", "emb_115", "emb_279"],
        "Attribution": [1900, 1350, 1300, 1150, 1100, 1000, 250, 220, 150, 140]
    }
    
    fig = px.bar(importance_data, x="Attribution", y="Feature", orientation="h",
                 title="Top Features Driving CLV Predictions",
                 color="Attribution", color_continuous_scale="greens")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Key Insights:**
    - **Monetary** (past spend) is the strongest predictor
    - **Engagement metrics** (purchase frequency, unique days) rank highly
    - **Text embeddings** (emb_*) contribute - what customers buy matters
    """)

with tab4:
    st.header("MLOps Architecture")
    
    st.code("""
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
    """, language=None)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Layer**")
        st.markdown("- BigQuery (storage)")
        st.markdown("- PySpark (processing)")
        st.markdown("- Hugging Face (embeddings)")
    
    with col2:
        st.markdown("**Deployment**")
        st.markdown("- GKE (production)")
        st.markdown("- Cloud Run (demo + A/B)")
        st.markdown("- Cloud Build CI/CD")
        st.markdown("- Model Registry")
    
    with col3:
        st.markdown("**Monitoring**")
        st.markdown("- Cloud Scheduler")
        st.markdown("- Evidently AI (drift)")
        st.markdown("- Cloud Functions (retrain)")
        st.markdown("- BigQuery (logs)")

# Footer
st.markdown("---")
st.caption("📝 Note: This demo exposes 5 of 396 features. The model uses median customer values as baseline (including 384 Hugging Face text embeddings) and overrides with slider inputs for real-time prediction.")
st.markdown("**Project by Arion Farhi** | [GitHub](https://github.com/arion-farhi)")
