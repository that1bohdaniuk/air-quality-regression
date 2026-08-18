import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌡️", layout="wide")

device = torch.device("cpu")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(APP_DIR, "assets", "weights")

# --- Architectures ---
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class AirQualityMLP(nn.Module):
    def __init__(self, input_dim, h1, h2):
        super(AirQualityMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.outputlayer = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.outputlayer(x)

# --- Model & Scaler Loading ---
@st.cache_resource
def load_models():
    scaler_raw = joblib.load(os.path.join(weight_path, "scaler_raw.joblib"))
    scaler_poly = joblib.load(os.path.join(weight_path, "scaler_poly.joblib"))
    poly = joblib.load(os.path.join(weight_path, "poly.joblib"))

    gradientboost_model = joblib.load(os.path.join(weight_path, "gradientboostweights.joblib"))
    
    linear_model = torch.load(
        os.path.join(weight_path, "linearregression.pth"),
        map_location=device,
        weights_only=False
    )
    linear_model.eval()

    deepml_model = torch.load(
        os.path.join(weight_path, "deepml.pth"),
        map_location=device,
        weights_only=False
    )
    deepml_model.eval()
    
    return scaler_raw, scaler_poly, poly, gradientboost_model, linear_model, deepml_model

scaler_raw, scaler_poly, poly, gb_model, lr_model, mlp_model = load_models()

# --- Streamlit UI ---
st.title("🌡️ Air Quality Multi-Model Temperature Benchmark")
st.write("Enter sensor readings below to run predictions through all 3 algorithms simultaneously.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Sensor Input Readings")
    co_gt = st.number_input("CO(GT)", value=2.6)
    pt08_s1 = st.number_input("PT08.S1(CO)", value=1360.0)
    c6h6_gt = st.number_input("C6H6(GT)", value=11.9)
    pt08_s2 = st.number_input("PT08.S2(NMHC)", value=1046.0)
    nox_gt = st.number_input("NOx(GT)", value=166.0)
    pt08_s3 = st.number_input("PT08.S3(NOx)", value=1056.0)
    no2_gt = st.number_input("NO2(GT)", value=113.0)
    pt08_s4 = st.number_input("PT08.S4(NO2)", value=1692.0)
    pt08_s5 = st.number_input("PT08.S5(O3)", value=1268.0)
    rh = st.number_input("Relative Humidity RH (%)", value=48.9)
    
    predict_btn = st.button("🚀 Predict Temperature Across All Models", type="primary")

with col2:
    st.subheader("Model Performance & Benchmark Results")
    
    if predict_btn:
        raw_features = np.array([[
            co_gt, pt08_s1, c6h6_gt, pt08_s2, nox_gt, 
            pt08_s3, no2_gt, pt08_s4, pt08_s5, rh
        ]])
        results = []

        # 1. HistGradientBoosting
        t0 = time.perf_counter()
        pred_gb = gb_model.predict(raw_features)[0]
        lat_gb = (time.perf_counter() - t0) * 1000
        results.append({
            "Algorithm": "HistGradientBoosting",
            "Predicted Temp (°C)": f"{pred_gb:.2f}",
            "Latency (ms)": f"{lat_gb:.3f}"
        })

        # 2. PyTorch Linear Regression
        t0 = time.perf_counter()
        poly_features = poly.transform(raw_features)
        scaled_poly_features = scaler_poly.transform(poly_features)
        tensor_poly = torch.tensor(scaled_poly_features, dtype=torch.float32)
        with torch.no_grad():
            pred_lr = lr_model(tensor_poly).item()
        lat_lr = (time.perf_counter() - t0) * 1000
        results.append({
            "Algorithm": "PyTorch Polynomial Linear Reg",
            "Predicted Temp (°C)": f"{pred_lr:.2f}",
            "Latency (ms)": f"{lat_lr:.3f}"
        })

        # 3. PyTorch DeepML MLP
        t0 = time.perf_counter()
        scaled_raw_features = scaler_raw.transform(raw_features)
        tensor_scaled = torch.tensor(scaled_raw_features, dtype=torch.float32)
        with torch.no_grad():
            pred_mlp = mlp_model(tensor_scaled).item()
        lat_mlp = (time.perf_counter() - t0) * 1000
        results.append({
            "Algorithm": "PyTorch DeepML MLP",
            "Predicted Temp (°C)": f"{pred_mlp:.2f}",
            "Latency (ms)": f"{lat_mlp:.3f}"
        })

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True, hide_index=True)