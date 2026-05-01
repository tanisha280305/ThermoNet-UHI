# ThermoNet-UHI 🛰️

AI-powered Urban Heat Island (UHI) detection & forecasting using MODIS satellite data.  
Combines CNN-LSTM deep learning with Physics-Informed Neural Networks (PINN).

---

## 📂 Dataset
- **Source:** MODIS MOD11A1 Land Surface Temperature (LST)
- **Coverage:** India (Lat: 20°–30°N, Lon: 72°–84°E)
- **Cities:** Delhi, Mumbai, Kolkata
- **Period:** 2022–2025
- **Resolution:** 150×150 pixel grid per day

---

## 🔬 Project Workflow

### Phase 1 — Data Loading & Preprocessing
- Loaded daily MODIS CSV files (LST Kelvin → °C)
- Mapped lat/lon coordinates to 150×150 pixel grid
- Handled NaN values via column mean imputation
- Vectorised grid filling (no iterrows)

### Phase 2 — Urban Heat Island Analysis
- Defined urban centers: Delhi, Mumbai, Kolkata
- Created urban/rural pixel masks (radius = 0.5°)
- Computed **UHI Intensity** = mean(urban LST) − mean(rural LST)
- Hotspot classification:
  - 🟢 Normal < 35°C
  - 🟡 High ≥ 35°C
  - 🟠 Severe ≥ 45°C
  - 🔴 Extreme ≥ 55°C

### Phase 3 — Visualization
- Interactive LST heatmaps with hotspot overlay
- ΔT difference maps (date vs date)
- Temporal trend charts (Mean / Max / Min LST)
- **Heat Flux Vector maps** (Fourier's Law: q = −k∇T)
- Location-specific LST trend charts

### Phase 4 — CNN-LSTM + PINN Model
- **CNN** — extracts spatial features from each daily LST grid
  - 2 Conv layers + BatchNorm + ReLU
  - 1×1 bottleneck + AdaptiveAvgPool
- **LSTM** — learns temporal patterns across days
  - 2 layers, hidden size 256, dropout 0.2
- **PINN Loss** enforces heat equation physics:
  - Heat diffusion residual (∂T/∂t = α∇²T)
  - Neumann boundary conditions (zero flux at edges)
  - Urban source penalty
  - Radiative loss term
- Metrics: **MAE, RMSE, Similarity Score**

### Phase 5 — Forecasting
- Multi-step future prediction for any target date
- Auto-regressive rollout (predict → feed back as input)
- Inverse MinMax scaling to °C

---

## 🚀 Quick Start
```bash
git clone https://github.com/tanisha280305/ThermoNet-UHI
cd ThermoNet-UHI
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ Add MODIS `.csv` files to `data_MOD11A1_2022_2025/data/` folder before running.

---

## 📦 Tech Stack
| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Deep Learning | PyTorch (CNN-LSTM) |
| Physics Model | PINN (Heat Equation) |
| Satellite Data | MODIS MOD11A1 LST |
| Visualization | Plotly |
| Data Processing | NumPy, Pandas |
| Python | 3.10+ |

## ⚠️ Disclaimer
For research & educational purposes only.
