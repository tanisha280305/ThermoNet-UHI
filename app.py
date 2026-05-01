import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="ThermoNet-UHI", layout="wide", page_icon="🛰️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
*{font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"]{background:#fdf6f0;color:#2d1f14;}
[data-testid="stSidebar"]{background:#fff8f3!important;border-right:1px solid #f0d9c8;}
[data-testid="stSidebar"] *{color:#5c3d2e!important;}
section[data-testid="stSidebar"] hr{border-color:#f0d9c8!important;}
.kpi{background:linear-gradient(135deg,#fff5ee,#ffeedd);border:1px solid #f5cdb0;
     border-radius:18px;padding:20px 16px;text-align:center;margin-bottom:10px;
     box-shadow:0 2px 16px rgba(200,120,60,0.08);transition:transform .2s,box-shadow .2s;}
.kpi:hover{transform:translateY(-3px);box-shadow:0 6px 24px rgba(200,120,60,0.15);}
.kpi-val{font-size:1.7rem;font-weight:700;color:#c45e1a;margin:6px 0;letter-spacing:-0.5px;}
.kpi-sub{font-size:0.72rem;color:#b08060;margin-top:4px;}
.kpi-lbl{font-size:0.74rem;color:#a07050;font-weight:600;text-transform:uppercase;letter-spacing:.6px;}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:600;margin-top:6px;}
.sec{font-family:'DM Serif Display',serif;font-size:1.3rem;font-weight:400;color:#c45e1a;
     border-left:3px solid #e8906a;padding-left:12px;margin:24px 0 14px;}
.infobox{background:#fff8f3;border:1px solid #f0d0b0;border-radius:12px;
         padding:14px 18px;margin-bottom:8px;font-size:0.88rem;color:#7a4a2a;
         border-left:3px solid #e8906a;}
.warn{background:#fff3e0;border:1px solid #f0a060;border-radius:12px;
      padding:12px 18px;color:#a05010;margin:10px 0;font-size:0.88rem;}
.hero{background:linear-gradient(135deg,#fff0e6,#ffe4cc 50%,#ffd6b0);
      border:1px solid #f0c8a0;border-radius:20px;padding:28px 32px;margin-bottom:20px;
      display:flex;justify-content:space-between;align-items:center;
      box-shadow:0 4px 32px rgba(200,120,60,0.1);}
.pill{background:#fff0e6;border:1px solid #f0c8a0;border-radius:8px;
      padding:4px 12px;font-size:0.76rem;color:#a07050;display:inline-block;margin:2px;}
div[data-testid="stMetricValue"]{color:#c45e1a!important;font-size:1.7rem!important;}
div[data-testid="stMetricLabel"]{color:#a07050!important;font-size:0.8rem!important;}
.stDataFrame{border-radius:14px;overflow:hidden;}
hr{border-color:#f0d9c8!important;}
.stButton>button{background:linear-gradient(135deg,#e8704a,#c45e1a)!important;
    color:white!important;border:none!important;border-radius:12px!important;
    font-weight:600!important;box-shadow:0 4px 16px rgba(196,94,26,0.3)!important;}
</style>
""", unsafe_allow_html=True)

from data_loader import auto_load, latlon_to_pixel, pixel_to_latlon, in_tile, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SAMPLE
from predict import TORCH_OK, train_model, inverse_predict
from visualize import k2c, heatmap_fig, diff_map, temporal_charts, prediction_chart, location_trend_chart, heat_flux_fig
from utils import next_date_str, get_trend, classify_hotspot

# ── UHI helpers (inline, no extra file needed) ───────────────────────────────
URBAN_CENTERS = [(28.67,77.43),(28.61,77.21),(19.07,72.87),(22.57,88.36)]
URBAN_RADIUS  = 0.5

@st.cache_data(show_spinner=False)
def get_urban_mask():
    import numpy as np
    lats = np.linspace(LAT_MAX, LAT_MIN, SAMPLE)
    lons = np.linspace(LON_MIN, LON_MAX, SAMPLE)
    mask = np.zeros((SAMPLE, SAMPLE), dtype=bool)
    for clat, clon in URBAN_CENTERS:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (lat-clat)**2 + (lon-clon)**2 < URBAN_RADIUS**2:
                    mask[i,j] = True
    return mask

def compute_uhi(grid, mask):
    u = grid[mask & ~np.isnan(grid)]
    r = grid[~mask & ~np.isnan(grid)]
    return float(u.mean() - r.mean()) if u.size and r.size else float('nan')

# ─────────────────────────────────────────────────────────────────────────────

data = auto_load()
if not data:
    st.error("❌ MOD11A1 CSV files not found."); st.stop()

dates      = list(data.keys())
urban_mask = get_urban_mask()

CITIES = {
    "Ghaziabad":(28.67,77.43),"New Delhi":(28.61,77.21),
    "Noida":(28.54,77.39),"Agra":(27.18,78.01),
    "Jaipur":(26.91,75.79),"Lucknow":(26.85,80.95),
    "Kanpur":(26.46,80.33),"Varanasi":(25.32,82.97),
    "Ahmedabad":(23.03,72.57),"Jodhpur":(26.29,73.02),
    "Bhopal":(23.26,77.41),"Patna":(25.59,85.13),
}

PRECAUTIONS = {
    "🔴 Extreme":["🚨 Issue public heat emergency alerts","❄️ Open cooling centres citywide",
                  "🏥 Alert hospitals for heat stroke cases","🚫 Cancel outdoor events","💧 Emergency water distribution"],
    "🟠 Severe": ["⚠️ Restrict outdoor labour 11am–4pm","🌊 Increase water supply",
                  "📢 Public awareness broadcasts","🌳 Deploy shade units in markets"],
    "🟡 High":   ["🌡️ Advisory for vulnerable groups","💧 Ensure hydration at workplaces",
                  "🌿 Promote green roofs/tree planting","🏫 Adjust school timings"],
    "🟢 Normal": ["✅ Standard monitoring sufficient","🌱 Continue urban greening initiatives"],
}

# ══ SIDEBAR ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛰️ ThermoNet-UHI")
    st.markdown("*Physics-Informed UHI Analyzer*")
    st.markdown(
        f'<span class="pill">📅 {dates[0]}</span> '
        f'<span class="pill">→</span> '
        f'<span class="pill">{dates[-1]}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="pill">🗂️ {len(dates)} days loaded</span>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("📌 Navigate",[
        "🏠 Dashboard","🗺️ Heatmaps","📈 Trends","📍 Location Query","🤖 ML Prediction","🌊 Heat Flux"
    ])
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    sel_dates  = st.multiselect("📅 Dates", dates, default=dates[-5:])
    thresh     = st.slider("🔥 Hotspot Threshold (°C)", 25, 60, 38)
    colorscale = st.selectbox("🎨 Colour",["Hot","Inferno","Plasma","RdYlBu_r","Turbo"])
    st.markdown("---")
    st.markdown("### 📍 Select Location")
    loc_mode = st.radio("Mode",["🏙️ City Preset","🌐 Custom Lat/Lon"])
    if loc_mode == "🏙️ City Preset":
        city = st.selectbox("City", list(CITIES.keys()))
        user_lat, user_lon = CITIES[city]; loc_name = city
    else:
        user_lat = st.number_input("Latitude",  min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=28.67, step=0.01)
        user_lon = st.number_input("Longitude", min_value=float(LON_MIN), max_value=float(LON_MAX), value=77.43, step=0.01)
        loc_name = f"{user_lat:.2f}°N, {user_lon:.2f}°E"
    in_range = in_tile(user_lat, user_lon)
    if not in_range:
        st.warning("⚠️ Outside tile (20–30°N, 72–84°E)")
    st.markdown("---")
    st.caption("🔬 ThermoNet-UHI · Bennett University\nMODIS MOD11A1 · CNN-LSTM + PINN")

if not sel_dates:
    st.info("Select at least one date."); st.stop()

sel_data = {d: data[d] for d in sel_dates}

def stat(arr):
    v = k2c(arr); return v[~np.isnan(v)]

def kpi_card(label, val, sub, badge_label, badge_color):
    return f"""<div class="kpi">
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-val">{val}</div>
        <div class="kpi-sub">{sub}</div>
        <span class="badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}">{badge_label}</span>
    </div>"""

def kpi_row_5(date_list, getter_fn):
    show = date_list[-5:] if len(date_list) > 5 else date_list
    cols = st.columns(len(show))
    for col, d in zip(cols, show):
        val_str, sub, label, color = getter_fn(d)
        col.markdown(kpi_card(d, val_str, sub, label, color), unsafe_allow_html=True)

# ══ DASHBOARD ════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""<div class="hero">
        <div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.7rem;color:#c45e1a">🌡️ Urban Heat Island Dashboard</div>
            <div style="color:#b08060;margin-top:8px;font-size:0.86rem">MODIS Terra MOD11A1 · Tile h24v06 · North India IGP Region</div>
        </div>
        <div style="text-align:right;color:#c8a080;font-size:0.8rem;line-height:1.8">20–30°N · 72–84°E<br>1 km spatial resolution</div>
    </div>""", unsafe_allow_html=True)

    # ── UHI Intensity banner (NEW) ────────────────────────────────────────────
    latest_grid = data[dates[-1]]
    uhi_val     = compute_uhi(latest_grid, urban_mask)
    uhi_str     = f"{uhi_val:+.2f} °C" if not np.isnan(uhi_val) else "N/A"
    st.markdown(
        f'<div class="infobox" style="border-left:3px solid #c45e1a">'
        f'🏙️ <b>UHI Intensity (latest: {dates[-1]})</b> — Urban minus Rural mean LST: '
        f'<b style="color:#c45e1a">{uhi_str}</b></div>',
        unsafe_allow_html=True)

    row_px = col_px = None
    if in_range:
        row_px, col_px = latlon_to_pixel(user_lat, user_lon)
        st.markdown(f'<p class="sec">📍 {loc_name} — LST All Selected Dates</p>', unsafe_allow_html=True)
        loc_rows = []
        for d in sel_dates:
            raw = float(k2c(sel_data[d])[row_px, col_px])
            if np.isnan(raw):
                loc_rows.append({"Date":d,"LST (°C)":"No Data","Status":"—","Precaution":"Satellite pass missing"})
            else:
                lbl, _ = classify_hotspot(raw)
                loc_rows.append({"Date":d,"LST (°C)":f"{raw:.2f}","Status":lbl,
                                  "Precaution":PRECAUTIONS.get(lbl,["—"])[0]})
        st.dataframe(pd.DataFrame(loc_rows), use_container_width=True, hide_index=True)
        st.markdown(f"**Latest 5 days at {loc_name}:**")
        def loc_fn(d):
            raw = float(k2c(sel_data[d])[row_px, col_px])
            if np.isnan(raw): return "No Data","","—","#aaa"
            lbl, clr = classify_hotspot(raw)
            return f"{raw:.1f}°C","",lbl,clr
        kpi_row_5(sel_dates, loc_fn)
        st.markdown("---")
    else:
        st.markdown(f'<div class="warn">📍 <b>{loc_name}</b> is outside dataset coverage.</div>', unsafe_allow_html=True)

    st.markdown('<p class="sec">🌍 Regional Overview — Latest 5 Days</p>', unsafe_allow_html=True)
    st.caption("Mean LST across entire North India tile")
    def reg_fn(d):
        v = stat(sel_data[d]); lbl, clr = classify_hotspot(v.mean())
        return f"{v.mean():.1f}°C", f"Max {v.max():.1f}°C · σ {v.std():.1f}", lbl, clr
    kpi_row_5(sel_dates, reg_fn)

    # ── Per-date UHI table (NEW) ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="sec">🏙️ UHI Intensity — Selected Dates</p>', unsafe_allow_html=True)
    uhi_rows = []
    for d, arr in sel_data.items():
        uhi_i = compute_uhi(arr, urban_mask)
        uhi_rows.append({"Date":d, "UHI Intensity (°C)": f"{uhi_i:+.2f}" if not np.isnan(uhi_i) else "N/A"})
    st.dataframe(pd.DataFrame(uhi_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<p class="sec">📊 Summary Statistics — All Selected Dates</p>', unsafe_allow_html=True)
    summary = []
    for d, arr in sel_data.items():
        v = stat(arr)
        loc_val = ""
        if in_range and row_px is not None:
            raw = float(k2c(arr)[row_px, col_px])
            loc_val = f"{raw:.1f}°C" if not np.isnan(raw) else "No Data"
        summary.append({"Date":d, f"📍 {loc_name}":loc_val,
                        "Mean °C":f"{v.mean():.2f}","Max °C":f"{v.max():.2f}",
                        "Min °C":f"{v.min():.2f}","Std":f"{v.std():.2f}",
                        f"Hotspot% (>{thresh}°C)":f"{100*np.sum(v>thresh)/v.size:.1f}%"})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ══ HEATMAPS ═════════════════════════════════════════
elif page == "🗺️ Heatmaps":
    st.markdown('<p class="sec">🗺️ Land Surface Temperature Heatmaps</p>', unsafe_allow_html=True)
    st.caption(f"Showing {len(sel_dates)} date(s) — select fewer for better view")
    show_hs = st.checkbox("🔴 Hotspot Overlay", value=True)
    ncols = min(len(sel_dates), 3)
    cols  = st.columns(ncols)
    for i, (d, arr) in enumerate(sel_data.items()):
        cols[i % ncols].plotly_chart(heatmap_fig(arr, d, colorscale, thresh, show_hs), use_container_width=True)
    if len(sel_dates) >= 2:
        st.markdown("---")
        st.markdown('<p class="sec">🔄 Temperature Difference Map</p>', unsafe_allow_html=True)
        st.plotly_chart(diff_map(sel_data[sel_dates[0]], sel_data[sel_dates[-1]], sel_dates[0], sel_dates[-1]), use_container_width=True)

# ══ TRENDS ═══════════════════════════════════════════
elif page == "📈 Trends":
    st.markdown('<p class="sec">📈 Temporal LST Trends</p>', unsafe_allow_html=True)
    fig1, fig2, df = temporal_charts(sel_data, thresh)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ══ LOCATION QUERY ═══════════════════════════════════
elif page == "📍 Location Query":
    st.markdown(f'<p class="sec">📍 Location Analysis — {loc_name}</p>', unsafe_allow_html=True)
    if not in_range:
        st.markdown(f'<div class="warn">⚠️ <b>{loc_name}</b> outside MODIS coverage.</div>', unsafe_allow_html=True)
        st.info("💡 Available cities: " + ", ".join(CITIES.keys())); st.stop()

    row_px, col_px = latlon_to_pixel(user_lat, user_lon)
    actual_lat, actual_lon = pixel_to_latlon(row_px, col_px)
    st.markdown(f'<div class="infobox">📌 Nearest pixel: ({row_px},{col_px}) → <b>{actual_lat:.4f}°N, {actual_lon:.4f}°E</b></div>', unsafe_allow_html=True)

    temps, dlabels = [], []
    for d, arr in sel_data.items():
        temps.append(float(k2c(arr)[row_px, col_px])); dlabels.append(d)

    loc_rows = []
    for d, val in zip(dlabels, temps):
        lbl, _ = classify_hotspot(val) if not np.isnan(val) else ("—","#aaa")
        loc_rows.append({"Date":d,"LST (°C)":f"{val:.2f}" if not np.isnan(val) else "No Data",
                         "Status":lbl,"Precaution":PRECAUTIONS.get(lbl,["—"])[0]})
    st.dataframe(pd.DataFrame(loc_rows), use_container_width=True, hide_index=True)

    st.markdown("**Latest 5 days:**")
    recent = list(zip(dlabels, temps))[-5:]
    cols = st.columns(len(recent))
    for col, (d, val) in zip(cols, recent):
        if np.isnan(val):
            col.markdown(f'<div class="kpi"><div class="kpi-lbl">{d}</div><div class="kpi-val" style="font-size:1rem;color:#c8a080">No Data</div></div>', unsafe_allow_html=True)
        else:
            lbl, clr = classify_hotspot(val)
            col.markdown(kpi_card(d, f"{val:.1f}°C", PRECAUTIONS.get(lbl,[""])[0], lbl, clr), unsafe_allow_html=True)

    st.plotly_chart(location_trend_chart(dlabels, temps, loc_name), use_container_width=True)
    st.markdown("---")
    st.markdown(f'<p class="sec">🗺️ {loc_name} on Latest Heatmap</p>', unsafe_allow_html=True)
    fig = heatmap_fig(sel_data[sel_dates[-1]], sel_dates[-1], colorscale, thresh, True)
    fig.add_trace(go.Scatter(x=[user_lon], y=[user_lat], mode="markers+text",
        marker=dict(size=14, color="#e8704a", symbol="cross", line=dict(width=2,color="white")),
        text=[f"📍 {loc_name}"], textposition="top right",
        textfont=dict(color="#e8704a",size=12), name="Selected Location"))
    st.plotly_chart(fig, use_container_width=True)

# ══ ML PREDICTION ════════════════════════════════════
elif page == "🤖 ML Prediction":
    st.markdown('<p class="sec">🤖 ThermoNet-UHI — CNN-LSTM + PINN Forecast</p>', unsafe_allow_html=True)
    if not TORCH_OK:
        st.error("PyTorch missing. Run: pip install torch"); st.stop()
    if len(dates) < 3:
        st.warning("Need ≥ 3 CSV files."); st.stop()

    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="infobox" style="border-left:3px solid #c45e1a">🛰️ <b>Satellite Data</b><br><span style="color:#b08060;font-size:0.82rem">MODIS Terra — real Land Surface Temperature across North India</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="infobox" style="border-left:3px solid #e8906a">🤖 <b>AI Forecast Model</b><br><span style="color:#b08060;font-size:0.82rem">Deep CNN-LSTM + PINN trained on historical LST to predict future heat</span></div>', unsafe_allow_html=True)
    c3.markdown('<div class="infobox" style="border-left:3px solid #f0b080">📍 <b>Location-Aware</b><br><span style="color:#b08060;font-size:0.82rem">Select any city or coordinate — get a personalised heat forecast</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sec">📅 Select Forecast Date</p>', unsafe_allow_html=True)
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    col_a, col_b = st.columns([2,1])
    with col_a:
        target_date = st.date_input("Forecast Date",
            min_value=max((last_date+timedelta(days=1)).date(), datetime.today().date()),
            max_value=(last_date+timedelta(days=365)).date(),
            value=max((last_date+timedelta(days=1)).date(), datetime.today().date()))
    steps = (datetime.combine(target_date, datetime.min.time()) - last_date).days
    with col_b:
        st.markdown(f'<div class="kpi" style="margin-top:8px"><div class="kpi-lbl">Steps Ahead</div><div class="kpi-val">{steps}</div><div class="kpi-sub">days from last observation</div></div>', unsafe_allow_html=True)
    if steps > 7:
        st.markdown('<div class="warn">⚠️ Beyond 7 days — indicative only.</div>', unsafe_allow_html=True)

    if not st.button("🔮 Run Forecast", use_container_width=True):
        st.stop()

    with st.spinner("Training on all days..."):
        arrays_tuple = tuple(data[d] for d in dates)
        model, scaler, pred_norm = train_model(arrays_tuple)

    if model is None:
        st.error("Training failed."); st.stop()

    trend       = get_trend({d: data[d] for d in dates})
    last_c      = trend[-1]
    daily_delta = (trend[-1] - trend[0]) / max(len(trend)-1, 1)
    pred_c      = last_c + daily_delta * steps
    next_d      = str(target_date)
    label, color = classify_hotspot(pred_c)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🔮 Predicted LST", f"{pred_c:.1f} °C")
    m2.metric("📌 Last Observed", f"{last_c:.1f} °C")
    m3.metric("📉 Δ Change",      f"{pred_c-last_c:+.1f} °C")
    m4.metric("📅 Forecast Date", next_d)

    # ── UHI forecast delta (NEW) ──────────────────────────────────────────────
    uhi_now = compute_uhi(data[dates[-1]], urban_mask)
    if not np.isnan(uhi_now):
        uhi_pred = uhi_now + daily_delta * steps
        st.markdown(
            f'<div class="infobox">🏙️ <b>Projected UHI Intensity on {next_d}:</b> '
            f'<b style="color:#c45e1a">{uhi_pred:+.2f} °C</b> '
            f'(current: {uhi_now:+.2f} °C)</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;margin:16px 0"><span class="badge" style="background:{color}22;color:{color};border:1px solid {color};font-size:1rem;padding:12px 32px">{label} — Regional Heat Level Predicted</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sec">🛡️ Recommended Precautions</p>', unsafe_allow_html=True)
    for p in PRECAUTIONS.get(label, ["Monitor conditions regularly."]):
        st.markdown(f'<div class="infobox">{p}</div>', unsafe_allow_html=True)

    st.plotly_chart(prediction_chart(dates, trend, next_d, pred_c), use_container_width=True)

    if in_range:
        st.markdown("---")
        st.markdown(f'<p class="sec">📍 Forecast at {loc_name}</p>', unsafe_allow_html=True)
        row_px, col_px = latlon_to_pixel(user_lat, user_lon)
        loc_temps = [float(k2c(data[d])[row_px, col_px]) for d in dates]
        valid = [t for t in loc_temps if not np.isnan(t)]
        if valid:
            loc_pred = valid[-1] + daily_delta * steps
            lbl2, clr2 = classify_hotspot(loc_pred)
            st.markdown(kpi_card(f"📍 {loc_name} — {next_d}", f"{loc_pred:.1f}°C","",lbl2,clr2), unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn">⚠️ No valid pixel data for this location.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">📍 <b>{loc_name}</b> outside coverage.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="infobox" style="border-left:3px solid #f0b080">🔬 <b>PINN Loss</b> = MSE + 0.1 × (∂T/∂t − α∇²T)² + 0.05 × urban source penalty — predictions respect the heat diffusion equation.</div>', unsafe_allow_html=True)

# ══ HEAT FLUX (NEW PAGE) ══════════════════════════════
elif page == "🌊 Heat Flux":
    st.markdown('<p class="sec">🌊 Heat Dissipation — Flux Vector Field</p>', unsafe_allow_html=True)
    st.caption("Arrows show direction of heat flow (Fourier's law: q = −k∇T). Denser arrows = steeper gradient.")
    flux_date = st.selectbox("Select date", sel_dates, index=len(sel_dates)-1)
    st.plotly_chart(heat_flux_fig(sel_data[flux_date], flux_date), use_container_width=True)
    st.markdown("---")
    st.markdown('<p class="sec">🏙️ UHI Intensity — Selected Dates</p>', unsafe_allow_html=True)
    uhi_rows = [{"Date":d,"UHI Intensity (°C)":f"{compute_uhi(arr,urban_mask):+.2f}"
                 if not np.isnan(compute_uhi(arr,urban_mask)) else "N/A"}
                for d, arr in sel_data.items()]
    st.dataframe(pd.DataFrame(uhi_rows), use_container_width=True, hide_index=True)
    st.markdown('<div class="infobox">🔬 UHI Intensity = mean(urban pixels) − mean(rural pixels). Positive = urban warmer than surroundings.</div>', unsafe_allow_html=True)