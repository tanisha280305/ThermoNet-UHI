import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

SAMPLE = 150

try:
    import torch, torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if TORCH_OK:
    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1), nn.ReLU(),   # 1x1 bottleneck
            nn.AdaptiveAvgPool2d(6),
        )
            self.lstm = nn.LSTM(32*6*6, 256, num_layers=2,
                            batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        def forward(self, x):
            B, T, H, W = x.shape
            feats = [self.cnn(x[:, t:t+1]).view(B, -1) for t in range(T)]
            _, (h, _) = self.lstm(torch.stack(feats, 1))
            return self.head(h[-1])

def pinn_loss(pred, target, grids):
    mse = nn.MSELoss()(pred, target)
    if grids.shape[1] < 2:
        return mse
    alpha = 0.02
    dt = 1.0
    dT_dt = (grids[:, -1] - grids[:, -2]) / dt
    T = grids[:, -1]

    # Interior Laplacian (unchanged)
    lap = (
        torch.roll(T, 1, -1) + torch.roll(T, -1, -1) +
        torch.roll(T, 1, -2) + torch.roll(T, -1, -2) - 4 * T
    )

    # ✅ ADD: Neumann BC — zero normal flux at boundaries
    # Forces ∂T/∂n = 0 at domain edges (insulated boundary assumption)
    bc_loss = (
        (T[:, :, 0] - T[:, :, 1]).pow(2).mean() +      # left edge
        (T[:, :, -1] - T[:, :, -2]).pow(2).mean() +     # right edge
        (T[:, 0, :] - T[:, 1, :]).pow(2).mean() +       # top edge
        (T[:, -1, :] - T[:, -2, :]).pow(2).mean()       # bottom edge
    )

    heat_residual = (dT_dt - alpha * lap) ** 2
    urban_mask = (T > T.mean()).float()
    source_penalty = (urban_mask * torch.relu(-dT_dt)).mean()

    T_norm = T / T.max().clamp(min=1e-6)
    rad_loss = (urban_mask * (T_norm**4 - dT_dt.abs())).clamp(min=0).mean()

    return mse + 0.1*heat_residual.mean() + 0.05*source_penalty + 0.02*bc_loss + 0.01*rad_loss


@st.cache_resource(show_spinner="🤖 Training CNN-LSTM + PINN...")
def train_model(arrays_tuple):
    if not TORCH_OK or len(arrays_tuple) < 3:
        return None, None, None

    arrays = np.stack(arrays_tuple)
    # Replace NaN with column mean before scaling
    col_means = np.nanmean(arrays.reshape(arrays.shape[0], -1), axis=1, keepdims=True)
    flat = arrays.reshape(arrays.shape[0], -1)
    for i in range(flat.shape[0]):
        flat[i] = np.where(np.isnan(flat[i]), col_means[i], flat[i])

    T = arrays.shape[0]
    scaler = MinMaxScaler()
    flat_n = scaler.fit_transform(flat.T).T
    arr_n  = flat_n.reshape(T, SAMPLE, SAMPLE)

    X      = torch.tensor(arr_n[:-1][None], dtype=torch.float32)
    y_true = torch.tensor([[arr_n[-1].mean()]], dtype=torch.float32)

    model = CNNLSTM()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    bar   = st.progress(0, "Training…")
    for ep in range(120):
        opt.zero_grad()
        loss = pinn_loss(model(X), y_true, X)
        loss.backward(); opt.step()
        if ep % 12 == 0:
            bar.progress((ep+1)/120, f"Epoch {ep+1}/120 · loss={loss.item():.4f}")
    bar.empty()

    with torch.no_grad():
        pred      = model(X)
        pred_norm = pred.item()
        true_norm = y_true.item()

        # Inverse to °C for readable accuracy
        pred_c = inverse_predict(pred_norm, scaler)
        true_c = inverse_predict(true_norm, scaler)

        mae  = abs(pred_c - true_c)
        rmse = mae  # single sample, same value
        r2 = max(0.0, 1 - ((true_c - pred_c)**2) / (np.var([true_c]) + 1e-8))
        # For single sample, report as "similarity" not R²:
        c1, c2, c3 = st.columns(3)
        c1.metric("📉 MAE",  f"{mae:.3f} °C")
        c2.metric("📉 RMSE", f"{rmse:.3f} °C")
        c3.metric("📈 Similarity", f"{1 - abs(pred_c-true_c)/(abs(true_c)+1e-8):.4f}")

    return model, scaler, pred_norm


def inverse_predict(pred_norm, scaler):
    pred_norm = float(pred_norm)
    if np.isnan(pred_norm):
        raise ValueError("Model output is NaN")
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, 0] = pred_norm
    return float(scaler.inverse_transform(dummy)[0, 0])


def predict_for_date(model, base_sequence, scaler, base_date, target_date):
    steps = (target_date - base_date).days
    if steps <= 0:
        raise ValueError("Target date must be after last observed date")
    model.eval()
    seq = torch.tensor(base_sequence[None], dtype=torch.float32)
    final_pred = None
    with torch.no_grad():
        for _ in range(steps):
            pred_norm  = model(seq).item()
            next_frame = torch.full((1,1,SAMPLE,SAMPLE), pred_norm)
            seq        = torch.cat([seq[:,1:], next_frame], dim=1)
            final_pred = pred_norm
    return inverse_predict(final_pred, scaler)