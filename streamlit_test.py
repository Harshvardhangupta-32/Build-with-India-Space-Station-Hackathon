import streamlit as st
import pandas as pd
import time
import random

# ----- Mock Data -----
@st.cache_data
def load_mock_data():
    return pd.DataFrame({
        'Timestamp': pd.date_range(start='2025-07-01', periods=10, freq='D'),
        'mAP@0.5': [random.uniform(0.6, 0.85) for _ in range(10)],
        'Recall': [random.uniform(0.7, 0.9) for _ in range(10)]
    })

def show_header():
    st.title("🛰️ Model Drift Monitor Dashboard")
    st.subheader("A Falcon-integrated Auto-Retraining System")

def show_metrics(data):
    st.metric("Latest mAP@0.5", f"{data['mAP@0.5'].iloc[-1]:.2f}")
    st.metric("Latest Recall", f"{data['Recall'].iloc[-1]:.2f}")

def show_performance_chart(data):
    st.line_chart(data.set_index('Timestamp'))

def show_drift_detection(data):
    threshold = 0.65
    drift_detected = data['mAP@0.5'].iloc[-1] < threshold
    
    if drift_detected:
        st.error("⚠️ Model Drift Detected! mAP dropped below threshold.")
        st.info("🔁 Initiating Auto-Retraining using Falcon...")
        show_retraining_pipeline()
    else:
        st.success("✅ Model is performing well. No retraining needed.")

def show_retraining_pipeline():
    with st.expander("🔧 Retraining Flow"):
        st.markdown("""
        1. **Detect Drift**: Performance metrics drop below acceptable levels.
        2. **Log Scenarios**: Send logs and images to Falcon.
        3. **Regenerate Synthetic Data**: Falcon creates new variations (e.g., new extinguisher design).
        4. **Augment Dataset**: Add new samples.
        5. **Auto-Retrain YOLOv8 Model**.
        6. **Re-deploy Updated Model**.
        """)

        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Model_training_pipeline.png/800px-Model_training_pipeline.png", caption="Conceptual Retraining Cycle")

# ----- Streamlit App Layout -----
def main():
    show_header()
    data = load_mock_data()
    show_metrics(data)
    show_performance_chart(data)
    show_drift_detection(data)

if __name__ == '__main__':
    main()
