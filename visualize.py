import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SAMPLE

BG   = "#fdf6f0"
BG2  = "#fff8f3"
FONT = "#5c3d2e"
ACC1 = "#c45e1a"
ACC2 = "#e8704a"
ACC3 = "#2ec4b6"

def k2c(arr):
    return np.array(arr, dtype=np.float32)  # already °C

def get_stats(arr):
    v = arr[~np.isnan(arr)]
    return dict(mean=v.mean(), std=v.std(), mx=v.max(), mn=v.min(), count=v.size)

def heatmap_fig(arr, date, colorscale, thresh, show_hotspot):
    lats = np.linspace(LAT_MAX, LAT_MIN, SAMPLE)
    lons = np.linspace(LON_MIN, LON_MAX, SAMPLE)
    fig  = go.Figure()
    fig.add_trace(go.Heatmap(
        z=arr, x=lons, y=lats,
        colorscale=colorscale,
        zmin=np.nanpercentile(arr,2), zmax=np.nanpercentile(arr,98),
        colorbar=dict(title="°C", thickness=12),
    ))
    if show_hotspot:
        mask = np.where(arr > thresh, 1.0, np.nan)
        fig.add_trace(go.Heatmap(
            z=mask, x=lons, y=lats,
            colorscale=[[0,"rgba(255,80,30,0.4)"],[1,"rgba(255,80,30,0.4)"]],
            showscale=False,
        ))
    fig.update_layout(
        title=dict(text=f"LST — {date}", font=dict(size=13, color=ACC1)),
        xaxis_title="Longitude", yaxis_title="Latitude",
        height=360, margin=dict(l=10,r=10,t=40,b=30),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(color=FONT),
    )
    return fig

def diff_map(arr1, arr2, d1, d2):
    diff = arr2 - arr1
    lats = np.linspace(LAT_MAX, LAT_MIN, SAMPLE)
    lons = np.linspace(LON_MIN, LON_MAX, SAMPLE)
    fig  = go.Figure(go.Heatmap(
        z=diff, x=lons, y=lats,
        colorscale="RdBu_r", colorbar=dict(title="ΔT °C"), zmid=0,
    ))
    fig.update_layout(
        title=f"ΔT: {d2} − {d1}",
        xaxis_title="Longitude", yaxis_title="Latitude",
        height=400, paper_bgcolor=BG2, plot_bgcolor=BG2, font=dict(color=FONT),
    )
    return fig

def temporal_charts(sel_data, thresh):
    rows = []
    for d, arr in sel_data.items():
        v = arr[~np.isnan(arr)]
        rows.append({"Date":d,"Mean":round(v.mean(),2),"Max":round(v.max(),2),
                     "Min":round(v.min(),2),"Hotspot %":round(100*np.sum(v>thresh)/v.size,1)})
    df = pd.DataFrame(rows)
    fig1 = go.Figure()
    for col, clr in [("Mean",ACC1),("Max",ACC2),("Min",ACC3)]:
        fig1.add_trace(go.Scatter(x=df["Date"],y=df[col],mode="lines+markers",
                                  name=col,line=dict(color=clr,width=2),marker=dict(size=8)))
    fig1.update_layout(title="LST Trend Over Time", yaxis_title="°C",
                       paper_bgcolor=BG, plot_bgcolor=BG2,
                       font=dict(color=FONT), legend=dict(bgcolor=BG2), height=380)
    fig2 = px.area(df, x="Date", y="Hotspot %", title=f"Hotspot Coverage % (>{thresh}°C)",
                   color_discrete_sequence=[ACC2])
    fig2.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=FONT), height=320)
    return fig1, fig2, df

def location_trend_chart(dates, temps, loc_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=temps, mode="lines+markers+text",
                             text=[f"{t:.1f}°C" for t in temps],
                             textposition="top center",
                             line=dict(color=ACC1,width=3),
                             marker=dict(size=10,color=ACC2)))
    fig.update_layout(title=f"LST at {loc_name}", yaxis_title="°C",
                      paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=FONT), height=360)
    return fig

def prediction_chart(dates, trend, next_date, pred_c):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=trend, mode="lines+markers",
                             name="Observed", line=dict(color=ACC1,width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[dates[-1],next_date], y=[trend[-1],pred_c],
                             mode="lines+markers", name="PINN Forecast",
                             line=dict(color=ACC2,dash="dash",width=2),
                             marker=dict(size=10,symbol="star")))
    fig.add_hrect(y0=40, y1=70, fillcolor="red", opacity=0.07,
                  annotation_text="Extreme Heat Zone", annotation_font_color=ACC2)
    fig.update_layout(title="Observed + Predicted LST", yaxis_title="°C",
                      paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=FONT), height=400)
    return fig

def heat_flux_fig(arr, date):
    """Visualize heat dissipation gradient vectors"""
    lats = np.linspace(LAT_MAX, LAT_MIN, SAMPLE)
    lons = np.linspace(LON_MIN, LON_MAX, SAMPLE)

    # Gradient = heat flux direction (Fourier's law: q = -k∇T)
    arr_filled = np.where(np.isnan(arr), np.nanmean(arr), arr)
    dy, dx = np.gradient(arr_filled)  # dy=lat gradient, dx=lon gradient

    # Subsample for readability
    step = 10
    xi = np.arange(0, SAMPLE, step)
    yi = np.arange(0, SAMPLE, step)
    X, Y = np.meshgrid(xi, yi)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=arr, x=lons, y=lats,
                             colorscale="Hot", colorbar=dict(title="°C")))
    # Quiver arrows showing heat flow direction
    for i in yi:
        for j in xi:
            fig.add_annotation(
                x=lons[j], y=lats[i],
                ax=lons[min(j+1,SAMPLE-1)]-dx[i,j]*3,
                ay=lats[min(i+1,SAMPLE-1)]+dy[i,j]*3,
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowsize=0.8,
                arrowcolor='rgba(255,255,100,0.5)', arrowwidth=1
            )
    fig.update_layout(title=f"Heat Flux Vectors — {date}",
                      height=450, paper_bgcolor=BG2, font=dict(color=FONT))
    return fig

