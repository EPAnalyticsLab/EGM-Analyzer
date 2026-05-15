"""
Cardiac Electrophysiology Analysis Application  –  Light mode
"""

import math, re, base64, io, json
import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from import_data.extract_dxl_data import extract_dxl_data
from utils.filters import bandpass_filter, notch_filter
# Core signal-processing primitives live in signal_processing.py so that the
# unit tests under /tests/ and the validation scripts under /validation/ can
# import them without spinning up the Dash app.
from signal_processing import (
    compute_lat,
    compute_vpp,
    compute_omnipolar,
    label_to_grid,
)

GEOMETRY = None
SIGNALS  = None

BG      = "#eaecef"
PANEL   = "#f3f4f6"
BORDER  = "#d0d7de"
TEXT    = "#24292f"
MUTED   = "#57606a"
ACCENT  = "#0969da"
DANGER  = "#cf222e"
SIG_COL = "#000000"
GEO_BG  = "#ffffff"
GEO_GRID= "#aaaaaa"

# ──────────────────────────────────────────────────────────────────────────────
# Colormaps. Defaults are perceptually uniform and colorblind-safe.
#   - "viridis" for sequential metrics (LAT, Vpp, ROR)
#   - "RdBu_r"  for divergent bipolar traces (zero-centred)
# Reviewer R1.16 (SoftwareX): the default colormap for 3D parameter maps must
# be perceptually uniform and colorblind-safe. "jet", "rainbow" and "turbo"
# are explicitly avoided here.
# ──────────────────────────────────────────────────────────────────────────────
CMAP_SEQUENTIAL = "Viridis"   # LAT, Vpp uni/bip/omni, ROR
CMAP_DIVERGENT  = "RdBu_r"    # signed bipolar traces / electric field

def compute_lat_legacy(signal):  # pragma: no cover — kept only as a reference
    """Legacy in-line copy of :func:`signal_processing.compute_lat`. Unused."""
    return compute_lat(signal)


def get_filter_func(fv, fs):
    if fv == "bp-2-100": return lambda x: bandpass_filter(x, 2, 100, fs)
    if fv == "notch-50": return lambda x: notch_filter(x, 50, fs)
    return lambda x: x


COORDS_RE = re.compile(r"([A-D])([1-4])")

def apply_interval_to_rov(rov, interval, signals):
    """Truncate signal columns to [t0_ms, t1_ms].
    Returns a NEW DataFrame with fewer sample columns — not NaN-padded.
    This avoids All-NaN warnings and keeps compute fast.
    """
    t0_ms = (interval or {}).get("t0")
    t1_ms = (interval or {}).get("t1")
    if t0_ms is None or t1_ms is None:
        return rov
    fs_hz = float(signals["data_table"]["Sample rate"].dropna().unique()[0])
    s0 = max(0, int(t0_ms * fs_hz / 1000.0))
    s1 = int(t1_ms * fs_hz / 1000.0)
    sig_cols = [c for c in rov.columns if c not in ("label","x","y")]
    keep_sig = sig_cols[s0:s1]          # just the slice, no NaN padding
    meta_cols = [c for c in ("label","x","y") if c in rov.columns]
    return rov[meta_cols + keep_sig]


def get_group_data(group_id, filter_val="None"):
    dt = SIGNALS["data_table"]
    # Coerce both sides to the same type to avoid silent mismatches
    # (Dash serialises selector values as int/float via JSON)
    col = dt["pt number"]
    try:
        gid = type(col.iloc[0])(group_id)   # cast group_id to same type as column
    except Exception:
        gid = group_id
    rows = dt[col == gid]
    if rows.empty:
        # fallback: string comparison
        rows = dt[col.astype(str) == str(group_id)]
    rov  = SIGNALS["signals"]["rov trace"].loc[rows.index]
    fs   = dt["Sample rate"].dropna().unique()[0]
    return rov, get_filter_func(filter_val, fs)

def get_signal_array(row):
    return row.drop(["label","x","y"], errors="ignore").astype(float).values

def compute_group_metrics(rov, filt):
    res = {}
    for idx, row in rov.iterrows():
        lbl  = str(row["label"])
        sig  = get_signal_array(row)
        sigf = filt(sig)
        r, c = label_to_grid(lbl)
        res[lbl] = {"lat":compute_lat(sigf),"vpp_uni":compute_vpp(sigf),
                    "signal":sigf,"row":r,"col":c}
    return res

def compute_bipolars(metrics):
    h_bip, v_bip, lmap = {}, {}, {}
    for lbl, d in metrics.items():
        if d["row"] is not None:
            lmap[(d["row"], d["col"])] = lbl
    for (r, c), lbl in lmap.items():
        if (r, c+1) in lmap:
            l2 = lmap[(r,c+1)]
            h_bip[f"{lbl}-{l2}"] = metrics[lbl]["signal"] - metrics[l2]["signal"]
        if (r+1, c) in lmap:
            l2 = lmap[(r+1,c)]
            v_bip[f"{lbl}-{l2}"] = metrics[lbl]["signal"] - metrics[l2]["signal"]
    return h_bip, v_bip, lmap

# L-shape configs: (name, corner_offsets_needed, bip_h_pair, bip_v_pair)
# Each config defines: for a corner at (r,c), which 3 electrodes are needed,
# and how to form bip_h and bip_v.
# Convention: positive = first - second
L_CONFIGS = {
    "└": dict(
        needs=lambda r,c: [(r,c),(r,c+1),(r-1,c)],
        bip_h=lambda sm,lm,r,c: sm[lm[(r,c)]]["signal"] - sm[lm[(r,c+1)]]["signal"],
        bip_v=lambda sm,lm,r,c: sm[lm[(r-1,c)]]["signal"] - sm[lm[(r,c)]]["signal"],
        label=lambda lm,r,c: (lm[(r-1,c)],lm[(r,c)],lm[(r,c+1)]),
    ),
    "┘": dict(
        needs=lambda r,c: [(r,c),(r,c-1),(r-1,c)],
        bip_h=lambda sm,lm,r,c: sm[lm[(r,c-1)]]["signal"] - sm[lm[(r,c)]]["signal"],
        bip_v=lambda sm,lm,r,c: sm[lm[(r-1,c)]]["signal"] - sm[lm[(r,c)]]["signal"],
        label=lambda lm,r,c: (lm[(r-1,c)],lm[(r,c)],lm[(r,c-1)]),
    ),
    "┌": dict(
        needs=lambda r,c: [(r,c),(r,c+1),(r+1,c)],
        bip_h=lambda sm,lm,r,c: sm[lm[(r,c)]]["signal"] - sm[lm[(r,c+1)]]["signal"],
        bip_v=lambda sm,lm,r,c: sm[lm[(r,c)]]["signal"] - sm[lm[(r+1,c)]]["signal"],
        label=lambda lm,r,c: (lm[(r,c)],lm[(r+1,c)],lm[(r,c+1)]),
    ),
    "┐": dict(
        needs=lambda r,c: [(r,c),(r,c-1),(r+1,c)],
        bip_h=lambda sm,lm,r,c: sm[lm[(r,c-1)]]["signal"] - sm[lm[(r,c)]]["signal"],
        bip_v=lambda sm,lm,r,c: sm[lm[(r,c)]]["signal"] - sm[lm[(r+1,c)]]["signal"],
        label=lambda lm,r,c: (lm[(r,c)],lm[(r+1,c)],lm[(r,c-1)]),
    ),
}

def compute_omnipoles_for_group(metrics, lmap):
    """Returns dict keyed by (cfg_name, corner_r, corner_c) for all valid L-shapes."""
    results = {}
    for cfg_name, cfg in L_CONFIGS.items():
        for (r, c) in list(lmap.keys()):
            needed = cfg["needs"](r, c)
            if not all(k in lmap for k in needed):
                continue
            try:
                bx = cfg["bip_h"](metrics, lmap, r, c)
                by = cfg["bip_v"](metrics, lmap, r, c)
                om, re_, ang, vec = compute_omnipolar(bx, by)
                lbls = cfg["label"](lmap, r, c)
                results[(cfg_name, r, c)] = {
                    "bip_x": bx, "bip_y": by, "omni": om, "residue": re_,
                    "angle": ang, "vector": vec,
                    "vpp_omni": compute_vpp(om), "vpp_res": compute_vpp(re_),
                    "labels": lbls, "corner": (r, c),
                }
            except Exception as e:
                pass

    # Also compute cross for each full 2x2 clique
    cross_results = {}
    for (r, c) in list(lmap.keys()):
        if not all(k in lmap for k in [(r,c),(r+1,c),(r,c+1),(r+1,c+1)]):
            continue
        A,B,C,D = lmap[(r,c)],lmap[(r+1,c)],lmap[(r,c+1)],lmap[(r+1,c+1)]
        sA,sB,sC,sD = [metrics[k]["signal"] for k in [A,B,C,D]]
        th=math.radians(45); co,si=math.cos(th),math.sin(th)
        bxc,byc=sD-sA,sB-sC
        bxr=co*bxc-si*byc; byr=si*bxc+co*byc
        om,re_,ang,vec=compute_omnipolar(bxr,byr)
        cross_results[(r,c)]={"bip_x":bxr,"bip_y":byr,"omni":om,"residue":re_,
               "angle":ang,"vector":vec,
               "vpp_omni":compute_vpp(om),"vpp_res":compute_vpp(re_),
               "labels":(A,B,C,D),"corner":(r,c)}
    return results, cross_results

def build_geometry_figure(vertices, faces, extra_traces=None):
    mesh = go.Mesh3d(
        x=vertices["x"],y=vertices["y"],z=vertices["z"],
        i=faces["v1"],j=faces["v2"],k=faces["v3"],
        opacity=0.35,color="#aaaaaa",flatshading=False,
        lighting=dict(ambient=0.9,diffuse=0.5),
        showlegend=False,showscale=False,
    )
    traces = [mesh] + (extra_traces or [])
    fig = go.Figure(data=traces)
    ax = dict(gridcolor=GEO_GRID,zerolinecolor=GEO_GRID,showbackground=True,
              backgroundcolor=GEO_BG,tickfont=dict(color=TEXT))
    fig.update_layout(
        scene=dict(bgcolor=GEO_BG,
                   xaxis=dict(title=dict(text="x (mm)",font=dict(color=TEXT)),**ax),
                   yaxis=dict(title=dict(text="y (mm)",font=dict(color=TEXT)),**ax),
                   zaxis=dict(title=dict(text="z (mm)",font=dict(color=TEXT)),**ax)),
        paper_bgcolor=PANEL,font_color=TEXT,
        margin=dict(l=0,r=0,b=0,t=30),showlegend=False,
    )
    return fig

def interpolate_on_mesh(vertices, faces, pts_xyz, values, sigma=2.0):
    vx  = vertices[["x","y","z"]].values.astype(float)
    pts = np.array(pts_xyz,dtype=float)
    vals= np.array(values,dtype=float)
    valid = np.isfinite(vals)
    if valid.sum()==0: return np.zeros(len(vx))
    tree = cKDTree(pts[valid])
    k    = min(4, valid.sum())
    dists,idxs = tree.query(vx, k=k)
    if dists.ndim==1: dists=dists[:,None]; idxs=idxs[:,None]
    w     = 1.0/(dists+1e-9)
    w_sum = w.sum(axis=1,keepdims=True)
    interp= (w * vals[valid][idxs]).sum(axis=1)/w_sum.squeeze()
    if sigma>0:
        tree_v   = cKDTree(vx)
        smoothed = np.zeros_like(interp)
        nn_lists = tree_v.query_ball_point(vx, r=sigma*3)
        for vi, nb_list in enumerate(nn_lists):
            nb = np.array(nb_list)
            d  = np.linalg.norm(vx[nb]-vx[vi],axis=1)
            w_ = np.exp(-0.5*(d/sigma)**2)
            smoothed[vi] = (w_*interp[nb]).sum()/w_.sum()
        interp = smoothed
    return interp

def empty_fig(msg="No data loaded"):
    fig=go.Figure()
    fig.update_layout(paper_bgcolor=PANEL,plot_bgcolor=BG,
        xaxis=dict(visible=False),yaxis=dict(visible=False),
        annotations=[dict(text=msg,x=0.5,y=0.5,showarrow=False,
                         font=dict(color=MUTED,size=14))])
    return fig

EMPTY_FIG = empty_fig()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "EP Analysis"
INP = {"fontSize":"11px","border":f"1px solid {BORDER}","borderRadius":"4px",
       "padding":"2px 4px","backgroundColor":PANEL,"color":TEXT}

def lspan(txt):
    return html.Span(txt,style={"fontSize":"12px","color":MUTED,
                                "marginRight":"6px","whiteSpace":"nowrap"})

app.layout = dbc.Container(fluid=True,
    style={"backgroundColor":BG,"padding":"10px"},
    children=[
    dbc.Row([
        dbc.Col(html.H4("Cardiac EP Analysis",
            style={"color":ACCENT,"margin":"0"}),width="auto"),
        dbc.Col([lspan("System:"),
            dbc.Select(id="system-select",
                options=[{"label":"Ensite Precision","value":"precision"},
                         {"label":"Ensite X","value":"ensitex"}],
                value="precision",
                style={"width":"160px","fontSize":"12px","display":"inline-block"})],
            width="auto",style={"paddingTop":"4px","display":"flex","alignItems":"center","gap":"4px"}),
        dbc.Col(dcc.Loading(type="circle",color="#0969da",children=[
            dcc.Upload(
                dbc.Button("📂 Load Geometry",color="primary",outline=True,size="sm"),
                id="upload-geometry",multiple=False),
        ]),width="auto",style={"paddingTop":"2px"}),
        # Precision: file picker (shown when system=precision)
        dbc.Col(dcc.Loading(type="circle",color="#0969da",children=[
            dcc.Upload(
                dbc.Button("📂 Load Signals",color="success",outline=True,size="sm"),
                id="upload-signals",multiple=True),
        ]),width="auto",style={"paddingTop":"2px"},id="col-upload-signals"),
        # EnsiteX: path inputs (shown when system=ensitex)
        dbc.Col(html.Div(id="ensitex-path-container", style={"display":"none"},
            children=[
                dcc.Input(id="ensitex-wave-path",
                    placeholder="Ruta completa a Wave_rov.csv",
                    debounce=False,
                    style={**INP,"width":"300px"}),
                dcc.Input(id="ensitex-lat-path",
                    placeholder="Ruta completa a Map_LAT_uni.csv",
                    debounce=False,
                    style={**INP,"width":"300px","marginLeft":"4px"}),
                dbc.Button("Load",id="btn-ensitex-load",n_clicks=0,
                    size="sm",color="success",outline=True,
                    style={"marginLeft":"4px"}),
            ]),width="auto",style={"paddingTop":"2px"}),
        dbc.Col([
            dbc.Checklist(
                options=[{"label":" Estimate missing electrodes","value":"estimate"}],
                value=[], id="estimate-missing-toggle", inline=True,
                style={"fontSize":"12px","color":MUTED}),
        ],width="auto",style={"paddingTop":"6px"}),
        dbc.Col(dcc.Loading(type="circle",color="#0969da",children=[
            dcc.Upload(
                dbc.Button("💾 Load Session (.pkl)",color="warning",outline=True,size="sm"),
                id="upload-pkl",multiple=False),
        ]),width="auto",style={"paddingTop":"2px"}),
        dbc.Col([
            dbc.Button("📤 Export",id="btn-export-open",n_clicks=0,
                       size="sm",color="secondary",outline=True),
        ],width="auto",style={"paddingTop":"2px"}),
        dbc.Col(html.Div(id="load-status",
            style={"fontSize":"12px","color":MUTED,"paddingTop":"6px"})),
    ],align="center",style={"marginBottom":"8px","paddingBottom":"8px",
                             "borderBottom":f"1px solid {BORDER}"}),

    # Export modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("📤 Export data and figures")),
        dbc.ModalBody([
            dbc.Checklist(id="export-checklist",
                options=[
                    {"label":"Vpp Unipolar (CSV)",     "value":"vpp_uni"},
                    {"label":"Vpp Bipolar (CSV)",      "value":"vpp_bip"},
                    {"label":"Vpp Omnipolar + ROR (CSV)", "value":"vpp_omni"},
                    {"label":"Signals Unipolar (CSV)", "value":"sig_uni"},
                    {"label":"Signals Bipolar (CSV)",  "value":"sig_bip"},
                    {"label":"Signals Omnipolar (CSV)","value":"sig_omni"},
                    {"label":"3D mesh screenshot (PNG)", "value":"fig_3d"},
                    {"label":"Signal panels (PNG: uni / bip / omni)", "value":"fig_signals"},
                ],
                value=[], style={"fontSize":"13px"}),
            html.Div("CSV files contain numeric data; PNG files are static "
                     "screenshots of the currently displayed figures. All "
                     "selected outputs are bundled into a single ZIP archive.",
                     style={"marginTop":"8px","fontSize":"11px","color":MUTED,
                            "fontStyle":"italic"}),
            html.Div(id="export-status",
                     style={"marginTop":"8px","fontSize":"12px","color":MUTED}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Export selected",id="btn-export-run",
                       n_clicks=0,color="primary",size="sm"),
            dbc.Button("Close",id="btn-export-close",
                       n_clicks=0,color="secondary",outline=True,size="sm"),
            dcc.Download(id="download-export"),
        ]),
    ],id="export-modal",is_open=False),

dbc.Row([
    # ========== LEFT COLUMN: Geometry (top) + Unipolar (bottom) ==========
    dbc.Col([
        # TOP-LEFT: geometry
html.Div(
    style={"backgroundColor":PANEL,"padding":"8px","borderRadius":"8px",
           "border":f"1px solid {BORDER}","marginBottom":"8px"},
    children=[
        # Fila 1: Botones principales
        dbc.ButtonGroup([
            dbc.Button("LAT",      id="btn-lat",      n_clicks=0,size="sm",color="primary",outline=True),
            dbc.Button("Vpp Uni",  id="btn-vpp-uni",  n_clicks=0,size="sm",color="primary",outline=True),
            dbc.Button("Vpp Bip",  id="btn-vpp-bip",  n_clicks=0,size="sm",color="primary",outline=True),
            dbc.Button("Vpp Omni", id="btn-vpp-omni", n_clicks=0,size="sm",color="primary",outline=True),
            dbc.Button("ROR",      id="btn-ror",       n_clicks=0,size="sm",color="primary",outline=True),
            dbc.Button("🎬 Video", id="btn-video-map", n_clicks=0,size="sm",color="info",   outline=True),
            dbc.Button("Clear",    id="btn-clear-map", n_clicks=0,size="sm",color="danger",  outline=True),
        ],style={"marginBottom":"4px","flexWrap":"wrap"}),

        # Fila 2: Controles compactos
        html.Div(style={"display":"flex","alignItems":"center","flexWrap":"wrap",
                        "marginBottom":"4px","gap":"8px","fontSize":"11px"},
        children=[
            html.Div(id="omni-type-container",
                style={"display":"none"},
                children=[lspan("Omni:"),
                    dbc.Select(id="omni-type-select",
                        options=[{"label":"Tri","value":"tri"},{"label":"Cross","value":"cross"}],
                        value="tri",size="sm",style={"width":"80px","fontSize":"11px"})]),
            
            lspan("Cbar:"),
            dcc.Input(id="cbar-min",type="number",placeholder="Min",
                      style={**INP,"width":"55px","fontSize":"11px"}),
            dcc.Input(id="cbar-max",type="number",placeholder="Max",
                      style={**INP,"width":"55px","fontSize":"11px"}),
            dbc.Button("Apply",id="btn-apply-cbar",n_clicks=0,size="sm",color="warning",outline=True),
            dbc.Button("Auto", id="btn-auto-cbar", n_clicks=0,size="sm",color="secondary",outline=True),
            
            html.Span("|",style={"color":BORDER,"margin":"0 4px"}),
            
            dbc.Checklist(options=[{"label":" Interp","value":"interp"}],
                value=[],id="interp-toggle",inline=True,style={"fontSize":"11px","color":TEXT}),
            lspan("σ:"),
            html.Div(dcc.Slider(id="interp-sigma",min=0.5,max=10.0,step=0.5,value=2.0,
                marks=None,
                tooltip={"placement":"top","always_visible":True}),
                style={"width":"100px"}),
            
            html.Span("|",style={"color":BORDER,"margin":"0 4px"}),
            
            lspan("Group:"),
            dbc.Select(id="freeze-group-select",options=[],disabled=True,
                       style={"width":"100px","fontSize":"11px"}),
            
            html.Span("|",style={"color":BORDER,"margin":"0 4px"}),
            
            lspan("Opacity:"),
            dcc.Input(id="scatter-opacity",type="number",value=1.0,min=0.0,max=1.0,step=0.05,
                      style={**INP,"width":"50px","fontSize":"11px"}),
            
            html.Div(id="electrode-grid-inline",
                style={"lineHeight":"0","marginLeft":"8px"},
                children=[
                    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,18px)",
                                    "gridTemplateRows":"repeat(4,18px)","gap":"1px"},
                    id="electrode-grid-cells",children=[])
                ]),
        ]),

        html.Div(id="video-controls-container",
            style={"display":"none","marginBottom":"4px","flexWrap":"wrap",
                   "alignItems":"center","gap":"6px","fontSize":"11px"},
            children=[
                dbc.Button("▶",  id="btn-video-play",  n_clicks=0,size="sm",color="success",outline=True),
                dbc.Button("⏸", id="btn-video-pause", n_clicks=0,size="sm",color="warning",outline=True),
                dbc.Button("⏮", id="btn-video-reset", n_clicks=0,size="sm",color="secondary",outline=True),
                lspan("Speed:"),
                dbc.Select(id="video-speed",
                    options=[{"label":"0.25×","value":"250"},{"label":"0.5×","value":"125"},
                             {"label":"1×","value":"60"},{"label":"2×","value":"30"},{"label":"4×","value":"15"}],
                    value="60",style={"width":"70px","fontSize":"11px"}),
                lspan("Interp:"),
                dbc.Checklist(options=[{"label":"","value":"interp"}],
                    value=[],id="video-interp-toggle",inline=True),
                lspan("σ:"),
                html.Div(dcc.Slider(id="video-sigma",min=0.5,max=10.0,step=0.5,value=2.0,
                    tooltip={"placement":"top","always_visible":False}),style={"width":"80px"}),
                lspan("cmin:"),
                dcc.Input(id="video-cmin",type="number",value=-3.0,style={**INP,"width":"50px","fontSize":"11px"}),
                lspan("cmax:"),
                dcc.Input(id="video-cmax",type="number",value=3.0,style={**INP,"width":"50px","fontSize":"11px"}),
                dbc.Button("⚙",id="btn-video-compute",n_clicks=0,size="sm",color="info",outline=True),
                html.Span(id="video-compute-status",style={"fontSize":"10px","color":MUTED}),
            ]),

        dcc.Loading(type="circle", color="#0969da", children=[
        dcc.Graph(id="geo-graph",figure=EMPTY_FIG,
                  style={"height":"52vh"},config={"scrollZoom":True}),
        ]),
        dcc.Store(id="store-map-type",   data=None),
        dcc.Store(id="store-cbar-range", data={"min":None,"max":None}),
    ]),
        
        # BOTTOM-LEFT: unipolar
        html.Div(
            style={"backgroundColor":PANEL,"padding":"8px","borderRadius":"8px",
                   "border":f"1px solid {BORDER}"},
            children=[
                html.Div(style={"display":"flex","alignItems":"center",
                                "marginBottom":"4px","gap":"6px"},
                children=[lspan("Filter:"),
                    dbc.Select(id="uni-filter",
                        options=[{"label":"None","value":"None"},
                                 {"label":"Band Pass 2-100Hz","value":"bp-2-100"},
                                 {"label":"Notch 50Hz","value":"notch-50"}],
                        value="None",
                        style={"width":"160px","fontSize":"12px"}),
                ]),
                dcc.Loading(type="circle", color="#0969da", children=[
                dcc.Graph(id="uni-graph",figure=EMPTY_FIG,
                          style={"height":"55vh"},config={"scrollZoom":True}),
                ]),
            ]),
    ],width=6),

    # ========== RIGHT COLUMN: Bipolar (top) + Omnipolar (bottom) ==========
    dbc.Col([
        # TOP-RIGHT: bipolar
html.Div(
    style={"backgroundColor":PANEL,"padding":"8px","borderRadius":"8px",
           "border":f"1px solid {BORDER}","marginBottom":"8px"},
    children=[
        html.Div(style={"display":"flex","alignItems":"center",
                        "flexWrap":"wrap","marginBottom":"4px","gap":"8px"},
        children=[lspan("Bipolar:"),
            dbc.RadioItems(id="bip-direction",
                options=[{"label":"Horizontal","value":"h"},
                         {"label":"Vertical",  "value":"v"}],
                value="h",inline=True,
                style={"fontSize":"12px","color":TEXT}),
            dbc.Button("+ Custom bipolar",id="btn-add-bip",n_clicks=0,
                       size="sm",color="secondary",outline=True),
        ]),
        html.Div(id="custom-bip-container",
            style={"display":"none","marginBottom":"4px",
                   "alignItems":"center","gap":"4px","flexWrap":"wrap"},
            children=[
                dcc.Input(id="custom-bip-1",placeholder="Electrode 1 (A1)",
                          style={**INP,"width":"110px"}),
                dcc.Input(id="custom-bip-2",placeholder="Electrode 2 (B2)",
                          style={**INP,"width":"110px"}),
                dbc.Button("Compute",id="btn-compute-custom-bip",
                           n_clicks=0,size="sm",color="success",outline=True),
            ]),
        dcc.Loading(type="circle", color="#0969da", children=[
        html.Div(style={"height":"39vh","overflowY":"scroll"},
            children=[
                dcc.Graph(id="bip-graph",figure=EMPTY_FIG,
                          config={"scrollZoom":True}),
            ]),
        ]),
    ]),
        
        # BOTTOM-RIGHT: omnipolar
        html.Div(
            style={"backgroundColor":PANEL,"padding":"8px","borderRadius":"8px",
                   "border":f"1px solid {BORDER}"},
            children=[
                html.Div(style={"display":"flex","alignItems":"center",
                                "marginBottom":"4px","gap":"6px","flexWrap":"wrap"},
                children=[lspan("Omnipolar config:"),
                    dbc.Select(id="omni-plot-type",
                        options=[{"label":"Triangular","value":"tri"},
                                 {"label":"Cross","value":"cross"}],
                        value="tri",
                        style={"width":"130px","fontSize":"12px"}),
                    html.Div(id="tri-config-container",
                        style={"display":"inline-flex","alignItems":"center","gap":"6px"},
                        children=[lspan("Config:"),
                            dbc.Select(id="tri-config-select",
                                options=[
                                    {"label":"└","value":"└"},
                                    {"label":"┘","value":"┘"},
                                    {"label":"┌","value":"┌"},
                                    {"label":"┐","value":"┐"},
                                ],
                                value="└",
                                style={"width":"160px","fontSize":"12px"}),
                        ]),
                    dbc.Button("Select interval",id="btn-omni-interval",n_clicks=0,
                               size="sm",color="warning",outline=True),
                    html.Span(id="omni-interval-label",
                              style={"fontSize":"11px","color":MUTED}),
                ]),
                dcc.Loading(type="circle", color="#0969da", children=[
                dcc.Graph(id="omni-signals-graph",figure=EMPTY_FIG,
                          style={"height":"30vh","marginTop":"5px"},config={"scrollZoom":True}),
                dcc.Graph(id="omni-loops-graph",figure=EMPTY_FIG,
                          style={"height":"40vh","marginTop":"80px"},config={"scrollZoom":True}),
                ]),
            ]),
    ],width=6),
]),

    # Interval selection store: {t0, t1} in ms, None = full signal
    dcc.Store(id="store-omni-interval", data={"t0": None, "t1": None}),

    # Interval selector modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select temporal interval for omnipolar")),
        dbc.ModalBody([
            # Toggle: unipolars / bipolars
            html.Div(style={"display":"flex","gap":"8px","marginBottom":"8px"},
                children=[
                    dbc.Button("Unipolars", id="btn-interval-uni", n_clicks=0,
                               size="sm", color="primary", outline=False),
                    dbc.Button("Bipolars",  id="btn-interval-bip", n_clicks=0,
                               size="sm", color="primary", outline=True),
                    html.Span("Drag on the plot to select the interval, then click Apply.",
                              style={"fontSize":"11px","color":MUTED,
                                     "alignSelf":"center","marginLeft":"8px"}),
                ]),
            dcc.Graph(id="interval-preview-graph",
                      figure=EMPTY_FIG,
                      style={"height":"320px"},
                      config={"scrollZoom":True,
                              "modeBarButtonsToAdd":["select2d"],
                              "displayModeBar":True}),
            html.Div(style={"display":"flex","gap":"12px","marginTop":"8px",
                            "alignItems":"center","flexWrap":"wrap"},
                children=[
                    lspan("t₀ (ms):"),
                    dcc.Input(id="interval-t0",type="number",placeholder="auto",
                              style={**INP,"width":"90px"}),
                    lspan("t₁ (ms):"),
                    dcc.Input(id="interval-t1",type="number",placeholder="auto",
                              style={**INP,"width":"90px"}),
                    dbc.Button("Apply",id="btn-interval-apply",n_clicks=0,
                               color="primary",size="sm"),
                    dbc.Button("Reset (full signal)",id="btn-interval-reset",n_clicks=0,
                               color="secondary",outline=True,size="sm"),
                ]),
            html.Div(id="interval-status",
                     style={"fontSize":"11px","color":MUTED,"marginTop":"4px"}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close",id="btn-interval-close",n_clicks=0,
                       color="secondary",outline=True,size="sm")),
    ], id="omni-interval-modal", is_open=False, size="xl"),

    dcc.Store(id="store-custom-bips",data=[]),
    dcc.Store(id="store-video-state",data={"playing":False,"frame":0,"max_frame":100}),
    dcc.Interval(id="video-interval",interval=60,n_intervals=0,disabled=True),
])


# ── Load geometry ──────────────────────────────────────────────────────────────
@app.callback(
    Output("geo-graph","figure",allow_duplicate=True),
    Output("load-status","children",allow_duplicate=True),
    Input("upload-geometry","contents"),
    State("upload-geometry","filename"),
    prevent_initial_call=True,
)
def load_geometry(contents, filename):
    global GEOMETRY
    if not contents: return no_update, no_update
    try:
        decoded = base64.b64decode(contents.split(",")[1]).decode("utf-8")
        if filename.endswith(".xml"):
            import xml.etree.ElementTree as ET
            root  = ET.fromstring(decoded)
            verts = root.find("DIFBody/Volumes/Volume/Vertices").text
            polys = root.find("DIFBody/Volumes/Volume/Polygons").text
            vertices = pd.read_csv(io.StringIO(verts),sep=" ",header=None)
            faces    = pd.read_csv(io.StringIO(polys),sep=" ",header=None)
            vertices = vertices.dropna(axis=1,how="all").dropna(axis=0,how="all")
            faces    = faces.dropna(axis=1,how="all").dropna(axis=0,how="all").astype(int)
            vertices.columns=["x","y","z"]; faces.columns=["v1","v2","v3"]
            faces -= 1
        elif filename.endswith(".html"):
            m = re.search(r'Plotly\.newPlot\([^,]+,\s*(\[.*?\])\s*,',decoded,re.DOTALL)
            if not m: return no_update,f"❌ Cannot parse {filename}"
            dj   = json.loads(m.group(1))
            mesh = next((d for d in dj if d.get("type")=="mesh3d"),None)
            if not mesh: return no_update,f"❌ No mesh3d in {filename}"
            vertices=pd.DataFrame({"x":mesh["x"],"y":mesh["y"],"z":mesh["z"]})
            faces   =pd.DataFrame({"v1":mesh["i"],"v2":mesh["j"],"v3":mesh["k"]})
        else:
            return no_update,f"❌ Unsupported: {filename}"
        GEOMETRY={"vertices":vertices,"faces":faces}
        return build_geometry_figure(vertices,faces), \
               f"✅ Geometry: {len(vertices)} vertices"
    except Exception as e:
        return no_update,f"❌ {e}"


def parse_ensitex_from_paths(wave_path, lat_path):
    """Read EnsiteX files directly from disk — no browser upload, no OOM.
    MATLAB equivalent:
      T        = readtable(wave_path, HeaderLines=59)
      u.group  = T{1:end-1, 2}        -> col index 1 (0-based)
      u.EG     = T{1:end-1, 6:end-1}  -> cols 5..-2 (0-based)
      coords   = LATs{1:end-1, 8:10}  -> cols 7:10  (0-based)
    """
    import io as _io, os as _os

    wave_path = wave_path.strip().strip('"').strip("'")
    lat_path  = lat_path.strip().strip('"').strip("'")

    if not _os.path.isfile(wave_path):
        raise FileNotFoundError(f"No encontrado: {wave_path}")
    if not _os.path.isfile(lat_path):
        raise FileNotFoundError(f"No encontrado: {lat_path}")

    print(f"EnsiteX: leyendo {wave_path}")
    wave_df = pd.read_csv(wave_path, skiprows=59, header=0, low_memory=False)
    print(f"EnsiteX: leyendo {lat_path}")
    lat_df  = pd.read_csv(lat_path,  skiprows=59, header=0, low_memory=False)

    n_rows = min(len(wave_df) - 1, len(lat_df) - 1)
    if n_rows <= 0:
        raise ValueError(f"Sin filas de datos (wave={len(wave_df)}, lat={len(lat_df)})")

    groups  = wave_df.iloc[:n_rows, 1].astype(float).astype(int).values
    eg_data = wave_df.iloc[:n_rows, 5:-1].astype(float).values
    coords  = lat_df.iloc[:n_rows, 7:10].astype(float).values

    print(f"EnsiteX: {n_rows} electrodos, {eg_data.shape[1]} muestras, "
          f"grupos={sorted(set(groups.tolist()))[:10]}")

    if eg_data.shape[1] == 0:
        raise ValueError("Sin columnas de señal en Wave_rov")
    if coords.shape[1] < 3:
        raise ValueError(f"Coordenadas no encontradas — Map_LAT_uni tiene {lat_df.shape[1]} cols")

    order   = np.argsort(groups, kind="stable")
    groups  = groups[order];  eg_data = eg_data[order];  coords = coords[order]

    group_counter: dict = {}
    labels: list = []
    for g in groups:
        cnt = group_counter.get(g, 0)
        group_counter[g] = cnt + 1
        row_letter = chr(ord("A") + min(cnt // 4, 3))
        col_num    = (cnt % 4) + 1
        labels.append(f"{row_letter}{col_num}")

    n_elec = len(groups)
    idx = list(range(n_elec))   # shared integer index for both dataframes

    dt = pd.DataFrame({
        "pt number":   groups,
        "roving x":    coords[:, 0],
        "roving y":    coords[:, 1],
        "roving z":    coords[:, 2],
        "Sample rate": [2000.0] * n_elec,
        "rov LAT":     [0.0]    * n_elec,
        "peak2peak":   [0.0]    * n_elec,
    }, index=idx)

    sample_cols = [f"s{i}" for i in range(eg_data.shape[1])]
    rov_df = pd.DataFrame(eg_data, columns=sample_cols, index=idx)
    rov_df.insert(0, "label", labels)
    rov_df.insert(1, "x", coords[:, 0])
    rov_df.insert(2, "y", coords[:, 1])

    # Sanity check
    print(f"EnsiteX: dt shape={dt.shape}, rov shape={rov_df.shape}, "
          f"grupos únicos={dt['pt number'].nunique()}")

    return {"data_table": dt, "signals": {"rov trace": rov_df}}


def estimate_missing_electrodes(signals):
    """For each freeze group, interpolate missing A1-D4 electrode signals
    from the available ones using their 3D coordinates (IDW).
    Returns an augmented SIGNALS dict."""
    import copy
    dt  = signals["data_table"].copy()
    rov = signals["signals"]["rov trace"].copy()

    rows_alphabet = ["A","B","C","D"]
    cols_num      = [1,2,3,4]
    augmented_dt_rows  = []
    augmented_rov_rows = []

    for group_id in dt["pt number"].unique():
        mask    = dt["pt number"] == group_id
        g_dt    = dt[mask]
        g_rov   = rov.loc[g_dt.index]
        present = {}   # (r,c) -> idx
        for idx, row in g_rov.iterrows():
            lbl = str(row["label"])
            m   = COORDS_RE.search(lbl)
            if m:
                r = ord(m.group(1)) - ord("A")
                c = int(m.group(2)) - 1
                present[(r,c)] = idx

        if len(present) < 3:
            continue  # not enough to interpolate

        # Source coords and signals
        src_idx  = list(present.values())
        src_xyz  = np.column_stack([
            dt.loc[src_idx,"roving x"].astype(float).values,
            dt.loc[src_idx,"roving y"].astype(float).values,
            dt.loc[src_idx,"roving z"].astype(float).values,
        ])
        # signal matrix (n_src × n_samples)
        sig_cols = [c for c in rov.columns if c not in ("label","x","y")]
        src_sigs = g_rov.loc[src_idx, sig_cols].astype(float).values  # (n_src, T)

        # Build IDW weights for missing electrodes
        # Estimate missing position by averaging known neighbour positions
        tree = cKDTree(src_xyz)
        new_dt_rows  = []
        new_rov_rows = []

        for ri, r_letter in enumerate(rows_alphabet):
            for ci, c_num in enumerate(cols_num):
                if (ri,ci) in present: continue
                # Estimate position: IDW from all present neighbours
                # Use nearest known neighbours
                k = min(4, len(src_idx))
                dists, idxs = tree.query(np.zeros((1,3)), k=k)
                # Better: use grid neighbours if available
                nbr_pos = []
                nbr_xyz = []
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nb = (ri+dr, ci+dc)
                    if nb in present:
                        nb_idx = present[nb]
                        nbr_pos.append(nb_idx)
                        nbr_xyz.append([
                            float(dt.loc[nb_idx,"roving x"]),
                            float(dt.loc[nb_idx,"roving y"]),
                            float(dt.loc[nb_idx,"roving z"]),
                        ])
                if not nbr_xyz:
                    continue  # can't estimate - no direct neighbours
                nbr_xyz = np.array(nbr_xyz)
                est_xyz = nbr_xyz.mean(axis=0)  # simple mean of neighbours

                # IDW signal interpolation from nearby electrodes
                dists_s, ids_s = tree.query(est_xyz.reshape(1,3), k=min(4,len(src_idx)))
                dists_s = dists_s.flatten(); ids_s = ids_s.flatten()
                w = 1.0/(dists_s + 1e-9)
                w /= w.sum()
                est_sig = (w[:,None] * src_sigs[ids_s]).sum(axis=0)

                lbl = f"{r_letter}{c_num} (est)"
                # dt row
                new_dt_rows.append({
                    "pt number":  group_id,
                    "roving x":   float(est_xyz[0]),
                    "roving y":   float(est_xyz[1]),
                    "roving z":   float(est_xyz[2]),
                    "Sample rate": float(dt.loc[src_idx[0],"Sample rate"]),
                    "rov LAT":    0.0,
                    "peak2peak":  0.0,
                })
                # rov row
                rov_row = {"label": lbl, "x": float(est_xyz[0]), "y": float(est_xyz[1])}
                for k_, v_ in zip(sig_cols, est_sig):
                    rov_row[k_] = v_
                new_rov_rows.append(rov_row)

        augmented_dt_rows.extend(new_dt_rows)
        augmented_rov_rows.extend(new_rov_rows)

    if augmented_dt_rows:
        extra_dt  = pd.DataFrame(augmented_dt_rows)
        extra_rov = pd.DataFrame(augmented_rov_rows)
        dt  = pd.concat([dt,  extra_dt],  ignore_index=True)
        rov = pd.concat([rov, extra_rov], ignore_index=True)

    return {"data_table": dt, "signals": {"rov trace": rov}}


# ── Load signals ───────────────────────────────────────────────────────────────
@app.callback(
    Output("freeze-group-select","options"),
    Output("freeze-group-select","disabled"),
    Output("load-status","children",allow_duplicate=True),
    Input("upload-signals","contents"),
    State("upload-signals","filename"),
    State("estimate-missing-toggle","value"),
    prevent_initial_call=True,
)
def load_signals(contents, filenames, estimate_missing):
    global SIGNALS
    if not contents: return [],True,no_update
    try:
        _,dt,raw_sigs,_ = extract_dxl_data(filenames,contents)
        sigs = {"data_table":dt,"signals":raw_sigs}
        if estimate_missing and "estimate" in estimate_missing:
            sigs = estimate_missing_electrodes(sigs)
        SIGNALS = sigs
        dt     = SIGNALS["data_table"]
        groups = dt["pt number"].unique()
        opts   = [{"label":str(g),"value":g} for g in groups]
        est    = " + estimated" if (estimate_missing and "estimate" in estimate_missing) else ""
        return opts, False, f"✅ Ensite Precision{est}: {len(groups)} freeze groups"
    except Exception as e:
        import traceback; traceback.print_exc()
        return [], True, f"❌ {e}"

# ── Load previous session (.pkl) via Dash upload ──────────────────────────────
# The application is a fully browser-based Dash app and does NOT depend on any
# OS-native windowing toolkit. Session resumption is handled entirely through
# the standard dcc.Upload component, ensuring identical behaviour on Windows,
# macOS, and Linux.
@app.callback(
    Output("freeze-group-select","options",  allow_duplicate=True),
    Output("freeze-group-select","disabled", allow_duplicate=True),
    Output("load-status","children",         allow_duplicate=True),
    Input("upload-pkl","contents"),
    State("upload-pkl","filename"),
    prevent_initial_call=True,
)
def load_pkl_session(contents, filename):
    global SIGNALS, GEOMETRY
    if not contents:
        return no_update, no_update, no_update
    try:
        import base64 as _b64, io, pickle
        _, b64data = contents.split(",", 1)
        raw = _b64.b64decode(b64data)
        data = pickle.load(io.BytesIO(raw))
        if isinstance(data, dict) and "signals" in data:
            if "geometry" in data and data["geometry"] is not None:
                GEOMETRY = data["geometry"]
            SIGNALS = data["signals"]
        else:
            SIGNALS = data
        dt = SIGNALS["data_table"]
        groups = dt["pt number"].unique()
        opts = [{"label": str(g), "value": g} for g in groups]
        return opts, False, f"✅ Session loaded from {filename}: {len(groups)} freeze groups"
    except Exception as e:
        import traceback; traceback.print_exc()
        return [], True, f"❌ PKL load error: {e}"


# ── Omni type selector + video controls visibility ────────────────────────────
@app.callback(
    Output("omni-type-container","style"),
    Output("video-controls-container","style"),
    Input("btn-vpp-omni","n_clicks"),Input("btn-lat","n_clicks"),
    Input("btn-vpp-uni","n_clicks"),Input("btn-vpp-bip","n_clicks"),
    Input("btn-ror","n_clicks"),Input("btn-clear-map","n_clicks"),
    Input("btn-video-map","n_clicks"),
    prevent_initial_call=True,
)
def toggle_panels(*_):
    tid = ctx.triggered_id
    omni_vis  = {"display":"block","marginBottom":"4px"} if tid=="btn-vpp-omni" else {"display":"none","marginBottom":"4px"}
    video_vis = {"display":"flex","marginBottom":"4px","flexWrap":"wrap","alignItems":"center","gap":"6px"} if tid=="btn-video-map" else {"display":"none"}
    return omni_vis, video_vis


# ── Toggle triangular sub-config visibility ──────────────────────────────────
@app.callback(
    Output("tri-config-container","style"),
    Input("omni-plot-type","value"),
    prevent_initial_call=False,
)
def toggle_tri_config(omni_type):
    if omni_type == "tri":
        return {"display":"inline-flex","alignItems":"center","gap":"6px"}
    return {"display":"none"}


# ── Toggle precision upload vs EnsiteX path inputs ───────────────────────────
@app.callback(
    Output("col-upload-signals","style"),
    Output("ensitex-path-container","style"),
    Input("system-select","value"),
    prevent_initial_call=False,
)
def toggle_system_ui(system):
    if system == "ensitex":
        return {"display":"none"},                {"display":"flex","alignItems":"center","gap":"2px","paddingTop":"2px"}
    return {"paddingTop":"2px"}, {"display":"none"}


# ── EnsiteX: load from disk paths ─────────────────────────────────────────────
@app.callback(
    Output("freeze-group-select","options",  allow_duplicate=True),
    Output("freeze-group-select","disabled", allow_duplicate=True),
    Output("load-status","children",         allow_duplicate=True),
    Input("btn-ensitex-load","n_clicks"),
    State("ensitex-wave-path","value"),
    State("ensitex-lat-path","value"),
    State("estimate-missing-toggle","value"),
    prevent_initial_call=True,
)
def load_ensitex_from_path(n_clicks, wave_path, lat_path, estimate_missing):
    global SIGNALS
    if not n_clicks: return no_update, no_update, no_update
    if not wave_path or not lat_path:
        return [], True, "❌ Introduce ambas rutas de fichero"
    try:
        sigs = parse_ensitex_from_paths(wave_path, lat_path)
        if estimate_missing and "estimate" in estimate_missing:
            sigs = estimate_missing_electrodes(sigs)
        SIGNALS = sigs
        dt     = SIGNALS["data_table"]
        groups = dt["pt number"].unique()
        opts   = [{"label":str(g),"value":g} for g in groups]
        est    = " + estimated" if (estimate_missing and "estimate" in estimate_missing) else ""
        return opts, False, f"✅ EnsiteX{est}: {len(groups)} freeze groups"
    except Exception as e:
        import traceback; traceback.print_exc()
        return [], True, f"❌ {e}"


# ── Store map type ─────────────────────────────────────────────────────────────
@app.callback(
    Output("store-map-type","data"),
    Input("btn-lat","n_clicks"),Input("btn-vpp-uni","n_clicks"),
    Input("btn-vpp-bip","n_clicks"),Input("btn-vpp-omni","n_clicks"),
    Input("btn-ror","n_clicks"),Input("btn-clear-map","n_clicks"),
    prevent_initial_call=True,
)
def store_map_type(*_):
    return {"btn-lat":"lat","btn-vpp-uni":"vpp_uni","btn-vpp-bip":"vpp_bip",
            "btn-vpp-omni":"vpp_omni","btn-ror":"ror",
            "btn-clear-map":None}.get(ctx.triggered_id)


# ── Colorbar store ─────────────────────────────────────────────────────────────
@app.callback(
    Output("store-cbar-range","data"),
    Input("btn-apply-cbar","n_clicks"),Input("btn-auto-cbar","n_clicks"),
    State("cbar-min","value"),State("cbar-max","value"),
    prevent_initial_call=True,
)
def update_cbar(_,__,cmin,cmax):
    return {"min":None,"max":None} if ctx.triggered_id=="btn-auto-cbar" \
           else {"min":cmin,"max":cmax}


# ── Toggle custom bip panel ────────────────────────────────────────────────────
@app.callback(
    Output("custom-bip-container","style"),
    Input("btn-add-bip","n_clicks"),
    State("custom-bip-container","style"),
    prevent_initial_call=True,
)
def toggle_custom(_, style):
    if style.get("display")=="none":
        return {"display":"flex","marginBottom":"4px",
                "alignItems":"center","gap":"4px","flexWrap":"wrap"}
    return {"display":"none","marginBottom":"4px",
            "alignItems":"center","gap":"4px","flexWrap":"wrap"}


# ── Geometry map ───────────────────────────────────────────────────────────────
@app.callback(
    Output("geo-graph","figure"),
    Input("store-map-type","data"),
    Input("store-cbar-range","data"),
    Input("freeze-group-select","value"),
    Input("interp-toggle","value"),
    Input("interp-sigma","value"),
    Input("omni-type-select","value"),
    Input("scatter-opacity","value"),
    Input("store-omni-interval","data"),
    prevent_initial_call=True,
)
def render_geo(map_type, cbar_range, group_id, interp_toggle, sigma, omni_type, scatter_opacity, interval):
    if GEOMETRY is None: return no_update
    verts=GEOMETRY["vertices"]; faces=GEOMETRY["faces"]
    extra=[]

    if SIGNALS is not None and group_id is not None:
        try:
            dt=SIGNALS["data_table"]
            grows=dt[dt["pt number"]==group_id]
            extra.append(go.Scatter3d(
                x=grows["roving x"].astype(float),
                y=grows["roving y"].astype(float),
                z=grows["roving z"].astype(float),
                mode="markers+text",
                marker=dict(size=7,color=SIG_COL,symbol="circle",opacity=scatter_opacity,
                            line=dict(color="#555555",width=1)),
                text=list(SIGNALS["signals"]["rov trace"].loc[grows.index,"label"].astype(str)),
                textposition="top center",
                textfont=dict(size=8,color=SIG_COL),
                showlegend=False,
            ))
        except: pass

    if map_type is None or SIGNALS is None:
        return build_geometry_figure(verts, faces, extra)

    try:
        dt  = SIGNALS["data_table"]
        rov = apply_interval_to_rov(
                  SIGNALS["signals"]["rov trace"], interval, SIGNALS)
        filt = lambda s: s
        ex=dt["roving x"].astype(float).values
        ey=dt["roving y"].astype(float).values
        ez=dt["roving z"].astype(float).values
        pts_xyz=np.column_stack([ex,ey,ez])

        values=[]; colorscale=CMAP_DIVERGENT; title=""

        if map_type=="lat":
            for idx in dt.index:
                try: values.append(compute_lat(filt(get_signal_array(rov.loc[idx]))))
                except: values.append(np.nan)
            # Convert sample index to ms at FS=2000 Hz
            fs_hz = float(SIGNALS["data_table"]["Sample rate"].dropna().unique()[0])
            values = [v * 1000.0/fs_hz if np.isfinite(v) else np.nan for v in values]
            title,colorscale="LAT (ms)",CMAP_SEQUENTIAL

        elif map_type=="vpp_uni":
            for idx in dt.index:
                try: values.append(compute_vpp(filt(get_signal_array(rov.loc[idx]))))
                except: values.append(np.nan)
            title,colorscale="Vpp Unipolar (mV)",CMAP_SEQUENTIAL

        elif map_type=="vpp_bip":
            for idx in dt.index:
                try:
                    g=dt.loc[idx,"pt number"]; gr=dt[dt["pt number"]==g]
                    met=compute_group_metrics(rov.loc[gr.index],filt)
                    h,v,_=compute_bipolars(met); lbl=str(rov.loc[idx,"label"])
                    vpps=[compute_vpp(s) for k,s in {**h,**v}.items() if lbl in k]
                    values.append(max(vpps) if vpps else np.nan)
                except: values.append(np.nan)
            title,colorscale="Vpp Bipolar max (mV)",CMAP_SEQUENTIAL

        elif map_type=="vpp_omni":
            for idx in dt.index:
                try:
                    g=dt.loc[idx,"pt number"]; gr=dt[dt["pt number"]==g]
                    met=compute_group_metrics(rov.loc[gr.index],filt)
                    _,_,lmap=compute_bipolars(met)
                    tri_od, cross_od = compute_omnipoles_for_group(met,lmap)
                    vpps=[]
                    if omni_type=="tri":
                        for d in tri_od.values(): vpps.append(d["vpp_omni"])
                    else:
                        for d in cross_od.values(): vpps.append(d["vpp_omni"])
                    values.append(max(vpps) if vpps else np.nan)
                except: values.append(np.nan)
            title=f"Vpp Omnipolar ({omni_type}) (mV)"; colorscale=CMAP_SEQUENTIAL

        elif map_type=="ror":
            for idx in dt.index:
                try:
                    g=dt.loc[idx,"pt number"]; gr=dt[dt["pt number"]==g]
                    met=compute_group_metrics(rov.loc[gr.index],filt)
                    _,_,lmap=compute_bipolars(met)
                    tri_od, cross_od = compute_omnipoles_for_group(met,lmap)
                    rors=[]
                    cfgs_iter = tri_od.values() if omni_type=="tri" else cross_od.values()
                    for v in cfgs_iter:
                        if v["vpp_omni"]>1e-9: rors.append(v["vpp_res"]/v["vpp_omni"])
                    values.append(np.nanmean(rors) if rors else np.nan)
                except: values.append(np.nan)
            title="ROR (residue/omnipolar)"; colorscale=CMAP_SEQUENTIAL
        else:
            return build_geometry_figure(verts,faces,extra)

        values=np.array(values,dtype=float)
        vmin=cbar_range.get("min") if cbar_range else None
        vmax=cbar_range.get("max") if cbar_range else None
        if vmin is None: vmin=float(np.nanmin(values))
        if vmax is None: vmax=float(np.nanmax(values))

        do_interp = interp_toggle and "interp" in interp_toggle
        ax = dict(gridcolor=GEO_GRID,zerolinecolor=GEO_GRID,showbackground=True,
                  backgroundcolor=GEO_BG,tickfont=dict(color=TEXT))

        if do_interp:
            vc = interpolate_on_mesh(verts,faces,pts_xyz,values,sigma=sigma)
            mesh_col = go.Mesh3d(
                x=verts["x"],y=verts["y"],z=verts["z"],
                i=faces["v1"],j=faces["v2"],k=faces["v3"],
                intensity=vc, intensitymode="vertex",
                colorscale=colorscale, cmin=vmin, cmax=vmax,
                opacity=float(scatter_opacity) if scatter_opacity is not None else 0.90, showscale=True,
                colorbar=dict(title=dict(text=title,font=dict(size=10)),thickness=14,
                              nticks=6,tickfont=dict(color=TEXT,size=9)),
                showlegend=False,
            )
            wire = go.Mesh3d(
                x=verts["x"],y=verts["y"],z=verts["z"],
                i=faces["v1"],j=faces["v2"],k=faces["v3"],
                opacity=0.10,color="#aaaaaa",showlegend=False,showscale=False,
            )
            fig=go.Figure(data=[wire,mesh_col]+extra)
            fig.update_layout(
                scene=dict(bgcolor=GEO_BG,
                           xaxis=dict(title=dict(text="x (mm)",font=dict(color="white")),**ax),
                           yaxis=dict(title=dict(text="y (mm)",font=dict(color="white")),**ax),
                           zaxis=dict(title=dict(text="z (mm)",font=dict(color="white")),**ax)),
                paper_bgcolor=PANEL,font_color=TEXT,
                margin=dict(l=0,r=0,b=0,t=30),showlegend=False,
                title=dict(text=title,font=dict(color=TEXT)),
            )
        else:
            _op = float(scatter_opacity) if scatter_opacity is not None else 1.0
            scatter=go.Scatter3d(
                x=ex,y=ey,z=ez,mode="markers",
                marker=dict(size=6,color=values,colorscale=colorscale,opacity=_op,
                            cmin=vmin,cmax=vmax,
                            colorbar=dict(title=dict(text=title,font=dict(size=10)),thickness=14,
                                          nticks=6,tickfont=dict(color=TEXT,size=9))),
                showlegend=False,
            )
            fig=build_geometry_figure(verts,faces,extra+[scatter])
            fig.update_layout(title=dict(text=title,font=dict(color=TEXT)))
        return fig
    except Exception as e:
        import traceback; traceback.print_exc()
        return build_geometry_figure(verts,faces,extra)


# ── Unipolar ───────────────────────────────────────────────────────────────────
@app.callback(
    Output("uni-graph","figure"),
    Input("freeze-group-select","value"),
    Input("uni-filter","value"),
    Input("store-omni-interval","data"),
    prevent_initial_call=True,
)
def render_uni(group_id, filter_val, interval):
    if SIGNALS is None or group_id is None: return EMPTY_FIG
    try:
        rov,filt=get_group_data(group_id,filter_val)
        rov = apply_interval_to_rov(rov, interval, SIGNALS)
        n=len(rov)
        # Collect all signals to compute global y range
        all_sigs=[]
        rows_data=[]
        for idx,row in rov.iterrows():
            lbl=str(row["label"]); sig=filt(get_signal_array(row))
            all_sigs.append(sig); rows_data.append((lbl,sig))
        sig_min=float(np.nanmin([s.min() for s in all_sigs]))
        sig_max=float(np.nanmax([s.max() for s in all_sigs]))
        # Time axis in ms
        fs_uni = float(SIGNALS["data_table"]["Sample rate"].dropna().unique()[0])
        n_pts=len(all_sigs[0]) if all_sigs else 1
        t=[i*1000.0/fs_uni for i in range(n_pts)]
        fig=make_subplots(rows=n,cols=1,shared_xaxes=True,vertical_spacing=0.002)
        for i,(lbl,sig) in enumerate(rows_data):
            fig.add_trace(go.Scatter(x=t,y=sig,mode="lines",
                  line=dict(color=SIG_COL,width=0.9),
                  name=lbl,showlegend=False),row=i+1,col=1)
            show_ticks = (i==n-1)
            fig.update_yaxes(range=[sig_min,sig_max],showticklabels=False,showgrid=False,
                     row=i+1,col=1)
            fig.update_xaxes(showticklabels=show_ticks,showgrid=False,
                     title_text="Time (ms)" if show_ticks else "",
                     row=i+1,col=1)
            # Añadir etiqueta como anotación horizontal
            yaxis_name = "y" if i == 0 else f"y{i+1}"
            fig.add_annotation(
                text=lbl,
                xref="paper", yref=f"{yaxis_name} domain",
                x=-0.02,y=0.5,
                xanchor="right", yanchor="middle",
                showarrow=False,
                font=dict(size=11, color=MUTED),
            )

        fig.update_layout(height=max(35*n,280),paper_bgcolor=PANEL,plot_bgcolor=BG,
            font_color=TEXT,margin=dict(l=100,r=5,b=20,t=25),dragmode="pan",
            title=dict(text=f"Unipolar – Group {group_id}",
               font=dict(color=TEXT,size=13)))
        return fig
    except Exception as e:
        print(f"Uni: {e}"); return EMPTY_FIG


# ── Bipolar ────────────────────────────────────────────────────────────────────
@app.callback(
    Output("bip-graph","figure"),
    Output("store-custom-bips","data"),
    Input("freeze-group-select","value"),
    Input("bip-direction","value"),
    Input("uni-filter","value"),
    Input("btn-compute-custom-bip","n_clicks"),
    State("custom-bip-1","value"),
    State("custom-bip-2","value"),
    State("store-custom-bips","data"),
    Input("store-omni-interval","data"),
    prevent_initial_call=True,
)
def render_bip(group_id,direction,filter_val,_nc,cbip1,cbip2,custom_bips,interval):
    if SIGNALS is None or group_id is None: return EMPTY_FIG,custom_bips or []
    try:
        rov,filt = get_group_data(group_id,filter_val)
        rov      = apply_interval_to_rov(rov, interval, SIGNALS)   # truncate to interval
        fs_hz    = float(SIGNALS["data_table"]["Sample rate"].dropna().unique()[0])
        metrics  = compute_group_metrics(rov,filt)
        h_bip,v_bip,_ = compute_bipolars(metrics)

        if ctx.triggered_id=="btn-compute-custom-bip" and cbip1 and cbip2:
            def _resolve(inp):
                s=inp.strip().upper()
                if s in metrics: return s
                for lbl in metrics:
                    if lbl.upper().endswith(s) or lbl.upper().endswith(" "+s): return lbl
                return None
            l1,l2 = _resolve(cbip1),_resolve(cbip2)
            if l1 and l2 and l1 in metrics and l2 in metrics:
                key = f"{l1}-{l2} (custom)"
                custom_bips = (custom_bips or [])+[{
                    "key":key,"group":group_id,
                    "signal":(metrics[l1]["signal"]-metrics[l2]["signal"]).tolist()}]

        std_bips = dict(h_bip if direction=="h" else v_bip)
        custom_for_group = {cb["key"]: np.array(cb["signal"])
                            for cb in (custom_bips or []) if cb["group"]==group_id}

        std_list    = list(std_bips.items())
        custom_list = list(custom_for_group.items())
        all_list    = std_list + custom_list
        n = len(all_list)
        if n==0: return EMPTY_FIG,custom_bips

        all_sigs_b = [sig for _,sig in all_list]
        bip_min = float(np.nanmin([np.nanmin(s) for s in all_sigs_b]))
        bip_max = float(np.nanmax([np.nanmax(s) for s in all_sigs_b]))
        n_pts_b = len(all_sigs_b[0]) if all_sigs_b else 1
        t_b = [i*1000.0/fs_hz for i in range(n_pts_b)]

        fig=make_subplots(rows=n,cols=1,shared_xaxes=True,vertical_spacing=0.002)
        for i,(lbl,sig) in enumerate(all_list):
            show_ticks_b = (i==n-1)
            is_custom    = lbl in custom_for_group
            line_style   = dict(color=SIG_COL,width=1.0,dash="dash") if is_custom else dict(color=SIG_COL,width=0.9)
            fig.add_trace(go.Scatter(x=t_b,y=sig,mode="lines",
                  line=line_style,name=lbl,showlegend=False),row=i+1,col=1)
            fig.update_yaxes(range=[bip_min,bip_max],showticklabels=False,showgrid=False,
                     row=i+1,col=1)
            fig.update_xaxes(showticklabels=show_ticks_b,showgrid=False,
                     title_text="Time (ms)" if show_ticks_b else "",
                     row=i+1,col=1)
            # Añadir etiqueta como anotación horizontal
            yaxis_name = "y" if i == 0 else f"y{i+1}"
            fig.add_annotation(
                text=lbl,
                xref="paper", yref=f"{yaxis_name} domain",
                x=-0.02, y=0.5,
                xanchor="right", yanchor="middle",
                showarrow=False,
                font=dict(size=11, color=MUTED),
            )

        dir_lbl   = "Horizontal" if direction=="h" else "Vertical"
        n_custom  = len(custom_list)
        custom_note = f" + {n_custom} custom" if n_custom else ""
        fig.update_layout(height=max(35*n,280),paper_bgcolor=PANEL,plot_bgcolor=BG,
            font_color=TEXT,margin=dict(l=130,r=5,b=20,t=25),dragmode="pan",
            title=dict(text=f"Bipolar ({dir_lbl}){custom_note} – Group {group_id}",
               font=dict(color=TEXT,size=13)))
        return fig,custom_bips
    except Exception as e:
        print(f"Bip: {e}"); import traceback; traceback.print_exc()
        return EMPTY_FIG,custom_bips or []


# ── Omnipolar ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("omni-signals-graph","figure"),
    Output("omni-loops-graph","figure"),
    Input("freeze-group-select","value"),
    Input("omni-plot-type","value"),
    Input("uni-filter","value"),
    Input("tri-config-select","value"),
    Input("store-omni-interval","data"),
    prevent_initial_call=True,
)
def render_omni(group_id, omni_type, filter_val, tri_cfg, interval):
    if SIGNALS is None or group_id is None: return EMPTY_FIG,EMPTY_FIG
    try:
        rov,filt = get_group_data(group_id,filter_val)
        rov      = apply_interval_to_rov(rov, interval, SIGNALS)  # truncate, no NaN
        metrics  = compute_group_metrics(rov,filt)
        _,_,lmap=compute_bipolars(metrics)
        tri_data, cross_data = compute_omnipoles_for_group(metrics,lmap)

        fs_hz = float(SIGNALS["data_table"]["Sample rate"].dropna().unique()[0])

        if omni_type == "cross":
            # cross_data keyed by (r,c) corner of 2x2 clique
            active = {k: v for k,v in cross_data.items()}
            clique_keys = list(active.keys())  # (r,c) tuples
        else:
            # tri_data keyed by (cfg_name, r, c)
            cfg = tri_cfg or "└"
            active = {k: v for k,v in tri_data.items() if k[0]==cfg}
            clique_keys = [(k[1],k[2]) for k in active.keys()]

        active_list = list(active.values())

        if not active_list:
            nf=empty_fig(f"No valid L-shapes found for config {omni_type}/{tri_cfg}")
            return nf, nf

        n_c = len(active_list)

        # ── TOP: omnipolar & residue signals ──────────────────────────────
        sig_fig=make_subplots(rows=n_c,cols=1,shared_xaxes=True,
                      vertical_spacing=0.02)
        for ci, od in enumerate(active_list):
            n_om = len(od["omni"])
            t=[_i*1000.0/fs_hz for _i in range(n_om)]
            row_i=ci+1
            lbls=od["labels"]
            lbl_str = "-".join(str(l) for l in lbls)

            sig_fig.add_trace(go.Scatter(x=t,y=od["omni"],mode="lines",
                line=dict(color=SIG_COL,width=1),
                name="Omnipolar",showlegend=(ci==0),legendgroup="omni"),
                row=row_i,col=1)
            sig_fig.add_trace(go.Scatter(x=t,y=od["residue"],mode="lines",
                line=dict(color=SIG_COL,width=1,dash="dash"),
                name="Residue",showlegend=(ci==0),legendgroup="res"),
                row=row_i,col=1)

            show_t=(ci==n_c-1)
            sig_fig.update_yaxes(showticklabels=False,showgrid=False,
                row=row_i,col=1)
            sig_fig.update_xaxes(showticklabels=show_t,showgrid=False,
                title_text="Time (ms)" if show_t else "",
                row=row_i,col=1)
    
            # Añadir etiqueta como anotación horizontal
            yaxis_name = "y" if ci == 0 else f"y{ci+1}"
            sig_fig.add_annotation(
                text=lbl_str,
                xref="paper", yref=f"{yaxis_name} domain",
                x=-0.02, y=0.5,
                xanchor="right", yanchor="middle",
            showarrow=False,
                font=dict(size=11, color=MUTED),
            )

        sig_fig.update_layout(
            height=max(45*n_c,160),paper_bgcolor=PANEL,plot_bgcolor=BG,
            font_color=TEXT,margin=dict(l=140,r=5,b=40,t=90),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,font=dict(size=10)),
            title=dict(
                text=f"Omnipolar (─) & Residue (- -) | {omni_type} {tri_cfg or ''} | Group {group_id}",
                font=dict(color=TEXT,size=12)),
        )

        # ── BOTTOM: 3×3 loop grid (always fixed 3×3, empty cells show ✕) ────
        corner_map = {}  # (r,c) -> omni_data
        if omni_type == "cross":
            for k,v in active.items(): corner_map[k] = v
        else:
            for k,v in active.items(): corner_map[(k[1],k[2])] = v

        all_loop_vals = []
        for od in active_list:
            all_loop_vals.extend(od["bip_x"].tolist())
            all_loop_vals.extend(od["bip_y"].tolist())
            all_loop_vals.extend([0, float(od["vector"][0]), float(od["vector"][1])])
        if all_loop_vals:
            loop_abs = max(abs(np.nanmin(all_loop_vals)), abs(np.nanmax(all_loop_vals)))
            loop_abs = loop_abs * 1.15 if loop_abs > 0 else 1.0
            loop_range = [-loop_abs, loop_abs]
        else:
            loop_range = [-1, 1]

        GRID_N = 3  # always 3×3
        loop_fig = make_subplots(rows=GRID_N, cols=GRID_N,
                                 vertical_spacing=0.04, horizontal_spacing=0.04)

        # Map available (r,c) keys into 3×3 positions by insertion order
        # So the first 9 active cliques fill row-major cells (0,0)→(2,2)
        pos_map = {}  # grid position (gr,gc) -> omni_data
        for idx_k, (k, od) in enumerate(corner_map.items()):
            gr = idx_k // GRID_N
            gc = idx_k %  GRID_N
            if gr < GRID_N:
                pos_map[(gr, gc)] = (k, od)

        for gr in range(GRID_N):
            for gc in range(GRID_N):
                rp = gr + 1; cp = gc + 1
                sp_idx = gr * GRID_N + gc
                total_sp = GRID_N * GRID_N
                xref = f"x{sp_idx+1} domain" if sp_idx > 0 else "x domain"
                yref = f"y{sp_idx+1} domain" if sp_idx > 0 else "y domain"

                if (gr, gc) in pos_map:
                    orig_k, od = pos_map[(gr, gc)]
                    bx, by = od["bip_x"], od["bip_y"]
                    vec = od["vector"]
                    # electrode label from original key
                    if omni_type == "cross":
                        cell_lbl = f"{chr(ord('A')+orig_k[0])}{orig_k[1]+1}"
                    else:
                        cell_lbl = f"{chr(ord('A')+orig_k[0])}{orig_k[1]+1}"
                    loop_fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                        line=dict(color=SIG_COL, width=1.2), showlegend=False),
                        row=rp, col=cp)
                    loop_fig.add_trace(go.Scatter(
                        x=[0, float(vec[0])], y=[0, float(vec[1])],
                        mode="lines+markers",
                        line=dict(color=DANGER, width=2),
                        marker=dict(size=[0,6], color=DANGER), showlegend=False),
                        row=rp, col=cp)
                    loop_fig.update_yaxes(range=loop_range, showticklabels=False, showgrid=True,
                        gridcolor=BORDER, zeroline=True, zerolinecolor=BORDER,
                        title_text=cell_lbl, title_font=dict(size=7, color=MUTED),
                        row=rp, col=cp)
                else:
                    loop_fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                        marker=dict(size=0, opacity=0), showlegend=False),
                        row=rp, col=cp)
                    loop_fig.add_annotation(
                        text="✕", x=0.5, y=0.5,
                        xref=xref, yref=yref,
                        showarrow=False,
                        font=dict(size=16, color=BORDER))
                    loop_fig.update_yaxes(range=loop_range, showticklabels=False, showgrid=False,
                        row=rp, col=cp)

                loop_fig.update_xaxes(range=loop_range, showticklabels=False, showgrid=True,
                    gridcolor=BORDER, zeroline=True, zerolinecolor=BORDER,
                    row=rp, col=cp)

        loop_fig.update_layout(
            height=350, paper_bgcolor=PANEL, plot_bgcolor=BG,
            font_color=TEXT, margin=dict(l=30, r=5, b=5, t=80),
            title=dict(
                text=f"Bipolar loops | {tri_cfg or omni_type} | "
                     f"<span style='color:{DANGER}'>→ propagation dir.</span>",
                font=dict(color=TEXT, size=11)),
        )
        return sig_fig,loop_fig

    except Exception as e:
        import traceback; traceback.print_exc()
        return EMPTY_FIG,EMPTY_FIG


# ── Electrode grid (HTML div cells) ──────────────────────────────────────────
@app.callback(
    Output("electrode-grid-cells","children"),
    Input("freeze-group-select","value"),
    prevent_initial_call=True,
)
def render_electrode_grid(group_id):
    if SIGNALS is None or group_id is None: return []
    try:
        dt  = SIGNALS["data_table"]
        rov = SIGNALS["signals"]["rov trace"]
        col = dt["pt number"]
        try:
            gid = type(col.iloc[0])(group_id)
        except Exception:
            gid = group_id
        rows_g = dt[col == gid]
        if rows_g.empty:
            rows_g = dt[col.astype(str) == str(group_id)]
        available = set()
        for idx in rows_g.index:
            try:
                lbl = str(rov.loc[idx,"label"])
                m   = COORDS_RE.search(lbl)
                if m: available.add((m.group(1), int(m.group(2))))
            except: pass

        cells = []
        for r in ["A","B","C","D"]:
            for c in [1,2,3,4]:
                exists = (r,c) in available
                cells.append(html.Div(
                    f"{r}{c}" if exists else "",
                    style={
                        "width":"22px","height":"22px",
                        "display":"flex","alignItems":"center","justifyContent":"center",
                        "fontSize":"8px","fontWeight":"bold","color":"white",
                        "backgroundColor":"#2ecc71" if exists else "#e0e0e0",
                        "border":"1px solid " + ("#27ae60" if exists else "#bbb"),
                        "borderRadius":"2px",
                        "cursor":"default",
                    }
                ))
        return cells
    except Exception as e:
        print(f"Grid: {e}"); return []



# ── Interval modal: open/close + uni/bip toggle ───────────────────────────────
@app.callback(
    Output("omni-interval-modal","is_open"),
    Output("interval-preview-graph","figure"),
    Output("btn-interval-uni","outline"),
    Output("btn-interval-bip","outline"),
    Input("btn-omni-interval","n_clicks"),
    Input("btn-interval-close","n_clicks"),
    Input("btn-interval-uni","n_clicks"),
    Input("btn-interval-bip","n_clicks"),
    State("omni-interval-modal","is_open"),
    State("freeze-group-select","value"),
    State("uni-filter","value"),
    State("store-omni-interval","data"),
    State("omni-plot-type","value"),
    State("tri-config-select","value"),
    prevent_initial_call=True,
)
def toggle_interval_modal(open_n, close_n, uni_n, bip_n,
                          is_open, group_id, filter_val, interval,
                          omni_type, tri_cfg):
    tid = ctx.triggered_id
    if tid == "btn-interval-close":
        return False, no_update, no_update, no_update

    show_bip = (tid == "btn-interval-bip")
    if tid in ("btn-interval-uni", "btn-interval-bip"):
        # Just switch the preview without reopening
        open_modal = True
    elif tid == "btn-omni-interval":
        open_modal = True
        show_bip = False
    else:
        return is_open, no_update, no_update, no_update

    uni_outline = show_bip      # uni button outlined when bipolars active
    bip_outline = not show_bip  # bip button outlined when unipolars active

    if SIGNALS is None or group_id is None:
        return open_modal, empty_fig("No data loaded"), uni_outline, bip_outline

    try:
        rov, filt = get_group_data(group_id, filter_val)
        fs_hz = float(SIGNALS["data_table"]["Sample rate"].dropna().unique()[0])
        fig = go.Figure()

        if not show_bip:
            # ── Unipolars: all signals overlaid in black ─────────────────────
            for _, row in rov.iterrows():
                sig = get_signal_array(row)
                sigf = filt(sig)
                t = [i * 1000.0 / fs_hz for i in range(len(sigf))]
                fig.add_trace(go.Scatter(
                    x=t, y=sigf, mode="lines",
                    line=dict(color="black", width=0.8),
                    showlegend=False, hoverinfo="skip"))
            ytitle = "Amplitude (mV)"
        else:
            # ── Bipolars: bip_x and bip_y for each L-shape, overlaid ─────────
            metrics = compute_group_metrics(rov, filt)
            _, _, lmap = compute_bipolars(metrics)
            tri_data, cross_data = compute_omnipoles_for_group(metrics, lmap)
            cfg = tri_cfg or "└"
            if omni_type == "tri":
                active = {k: v for k, v in tri_data.items() if k[0] == cfg}
            else:
                active = cross_data
            n_samps = None
            for od in active.values():
                n_samps = len(od["bip_x"])
                break
            if n_samps is None:
                fig.add_annotation(text="No bipolar data", x=0.5, y=0.5,
                                   xref="paper", yref="paper", showarrow=False)
            else:
                t = [i * 1000.0 / fs_hz for i in range(n_samps)]
                for od in active.values():
                    fig.add_trace(go.Scatter(
                        x=t, y=od["bip_x"], mode="lines",
                        line=dict(color="black", width=0.8),
                        showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(
                        x=t, y=od["bip_y"], mode="lines",
                        line=dict(color="black", width=0.8, dash="dot"),
                        showlegend=False, hoverinfo="skip"))
            ytitle = "Bipolar (mV)  — solid=X  dashed=Y"

        # Mark current interval
        t0 = (interval or {}).get("t0")
        t1 = (interval or {}).get("t1")
        if t0 is not None and t1 is not None:
            fig.add_vrect(x0=t0, x1=t1,
                fillcolor=ACCENT, opacity=0.15, layer="below", line_width=0)
            fig.add_vline(x=t0, line=dict(color=ACCENT, width=1.5, dash="dot"))
            fig.add_vline(x=t1, line=dict(color=ACCENT, width=1.5, dash="dot"))

        fig.update_layout(
            paper_bgcolor=PANEL, plot_bgcolor=BG, font_color=TEXT,
            margin=dict(l=40, r=10, b=40, t=10),
            xaxis=dict(title="Time (ms)", showgrid=True, gridcolor=BORDER),
            yaxis=dict(title=ytitle, showgrid=True, gridcolor=BORDER),
            dragmode="select",
        )
        return open_modal, fig, uni_outline, bip_outline

    except Exception as e:
        import traceback; traceback.print_exc()
        return open_modal, empty_fig(str(e)), uni_outline, bip_outline


@app.callback(
    Output("store-omni-interval","data"),
    Output("interval-t0","value"),
    Output("interval-t1","value"),
    Output("interval-status","children"),
    Output("omni-interval-label","children"),
    Output("omni-interval-modal","is_open", allow_duplicate=True),
    Input("btn-interval-apply","n_clicks"),
    Input("btn-interval-reset","n_clicks"),
    State("interval-t0","value"),
    State("interval-t1","value"),
    State("interval-preview-graph","selectedData"),
    prevent_initial_call=True,
)
def apply_interval(apply_n, reset_n, t0_input, t1_input, selected_data):
    tid = ctx.triggered_id
    if tid == "btn-interval-reset":
        return {"t0": None, "t1": None}, None, None,                "Interval reset — using full signal", "", no_update

    # Prefer box/lasso selection on graph over manual inputs
    t0, t1 = t0_input, t1_input
    if selected_data and "range" in selected_data:
        rng = selected_data["range"]["x"]
        t0, t1 = min(rng), max(rng)
    elif selected_data and "points" in selected_data and selected_data["points"]:
        xs = [p["x"] for p in selected_data["points"]]
        t0, t1 = min(xs), max(xs)

    if t0 is None or t1 is None:
        return no_update, t0_input, t1_input,                "❌ Set t₀ and t₁ or drag-select on the plot", no_update, no_update
    if t0 >= t1:
        return no_update, t0_input, t1_input,                "❌ t₀ must be less than t₁", no_update, no_update

    label = f"[{t0:.1f} – {t1:.1f} ms]"
    # close modal and update store — render_omni fires automatically
    return ({"t0": float(t0), "t1": float(t1)},
            float(t0), float(t1),
            f"✅ Interval applied: {label}",
            label,
            False)


# ── Video: precompute all frames as Plotly animation  ──────────────────────────
@app.callback(
    Output("geo-graph","figure", allow_duplicate=True),
    Output("video-compute-status","children"),
    Input("btn-video-compute","n_clicks"),
    State("video-interp-toggle","value"),
    State("video-sigma","value"),
    State("video-cmin","value"),
    State("video-cmax","value"),
    State("video-speed","value"),
    prevent_initial_call=True,
)
def precompute_video(n_clicks, interp_toggle, sigma, cmin, cmax, speed_val):
    if not n_clicks or SIGNALS is None or GEOMETRY is None:
        return no_update, "Load geometry + signals first"
    try:
        verts = GEOMETRY["vertices"]
        faces = GEOMETRY["faces"]
        dt    = SIGNALS["data_table"]
        rov   = SIGNALS["signals"]["rov trace"]
        fs_hz = float(dt["Sample rate"].dropna().unique()[0])

        # Collect electrode positions and all signals
        ex, ey, ez, all_sigs = [], [], [], []
        for idx in dt.index:
            try:
                sig = get_signal_array(rov.loc[idx])
                ex.append(float(dt.loc[idx,"roving x"]))
                ey.append(float(dt.loc[idx,"roving y"]))
                ez.append(float(dt.loc[idx,"roving z"]))
                all_sigs.append(sig.astype(float))
            except: pass
        if not ex:
            return no_update, "No electrode data found"

        pts_xyz = np.column_stack([ex, ey, ez])
        n_frames = len(all_sigs[0])
        vmin = float(cmin) if cmin is not None else -3.0
        vmax = float(cmax) if cmax is not None else  3.0
        do_interp = interp_toggle and "interp" in interp_toggle
        sig_arr = np.array(all_sigs)  # shape (n_elec, n_samples)
        ax = dict(gridcolor=GEO_GRID, zerolinecolor=GEO_GRID, showbackground=True,
                  backgroundcolor=GEO_BG, tickfont=dict(color=TEXT))

        # Step: subsample to max 200 frames for performance
        step = max(1, n_frames // 200)
        frame_indices = list(range(0, n_frames, step))

        # Build base figure (frame 0)
        vals0 = sig_arr[:, 0]
        colorbar_cfg = dict(title=dict(text="mV", font=dict(size=10)),
                            thickness=14, nticks=6,
                            tickfont=dict(color=TEXT, size=9))

        if do_interp:
            vc0 = interpolate_on_mesh(verts, faces, pts_xyz, vals0,
                                      sigma=sigma if sigma else 2.0)
            base_data = [
                go.Mesh3d(x=verts["x"], y=verts["y"], z=verts["z"],
                          i=faces["v1"], j=faces["v2"], k=faces["v3"],
                          opacity=0.10, color="#aaaaaa",
                          showlegend=False, showscale=False),
                go.Mesh3d(x=verts["x"], y=verts["y"], z=verts["z"],
                          i=faces["v1"], j=faces["v2"], k=faces["v3"],
                          intensity=vc0, intensitymode="vertex",
                          colorscale=CMAP_DIVERGENT, cmin=vmin, cmax=vmax,
                          opacity=0.90, showscale=True,
                          colorbar=colorbar_cfg, showlegend=False),
            ]
        else:
            base_data = [
                go.Mesh3d(x=verts["x"], y=verts["y"], z=verts["z"],
                          i=faces["v1"], j=faces["v2"], k=faces["v3"],
                          opacity=0.25, color="#aaaaaa",
                          showlegend=False, showscale=False),
                go.Scatter3d(x=ex, y=ey, z=ez, mode="markers",
                             marker=dict(size=7, color=vals0.tolist(),
                                         colorscale=CMAP_DIVERGENT, cmin=vmin, cmax=vmax,
                                         colorbar=colorbar_cfg),
                             showlegend=False),
            ]

        # Build animation frames
        anim_frames = []
        slider_steps = []
        interval_ms = int(speed_val) if speed_val else 60

        for fi in frame_indices:
            vals_f = sig_arr[:, fi]
            t_ms   = fi * 1000.0 / fs_hz
            if do_interp:
                vc = interpolate_on_mesh(verts, faces, pts_xyz, vals_f,
                                         sigma=sigma if sigma else 2.0)
                frame_data = [
                    go.Mesh3d(x=verts["x"], y=verts["y"], z=verts["z"],
                              i=faces["v1"], j=faces["v2"], k=faces["v3"],
                              opacity=0.10, color="#aaaaaa",
                              showlegend=False, showscale=False),
                    go.Mesh3d(x=verts["x"], y=verts["y"], z=verts["z"],
                              i=faces["v1"], j=faces["v2"], k=faces["v3"],
                              intensity=vc.tolist(), intensitymode="vertex",
                              colorscale=CMAP_DIVERGENT, cmin=vmin, cmax=vmax,
                              opacity=0.90, showscale=True,
                              colorbar=colorbar_cfg, showlegend=False),
                ]
            else:
                frame_data = [
                    go.Mesh3d(x=verts["x"], y=verts["y"], z=verts["z"],
                              i=faces["v1"], j=faces["v2"], k=faces["v3"],
                              opacity=0.25, color="#aaaaaa",
                              showlegend=False, showscale=False),
                    go.Scatter3d(x=ex, y=ey, z=ez, mode="markers",
                                 marker=dict(size=7, color=vals_f.tolist(),
                                             colorscale=CMAP_DIVERGENT, cmin=vmin, cmax=vmax,
                                             colorbar=colorbar_cfg),
                                 showlegend=False),
                ]
            frame_name = f"f{fi}"
            anim_frames.append(go.Frame(data=frame_data, name=frame_name,
                                        layout=go.Layout(title_text=f"t = {t_ms:.1f} ms")))
            slider_steps.append(dict(
                args=[[frame_name],
                      {"frame":{"duration":interval_ms,"redraw":True},
                       "mode":"immediate","transition":{"duration":0}}],
                label=f"{t_ms:.0f}",
                method="animate",
            ))

        fig = go.Figure(data=base_data, frames=anim_frames)
        fig.update_layout(
            scene=dict(bgcolor=GEO_BG,
                       xaxis=dict(title=dict(text="x",font=dict(color=TEXT)),**ax),
                       yaxis=dict(title=dict(text="y",font=dict(color=TEXT)),**ax),
                       zaxis=dict(title=dict(text="z",font=dict(color=TEXT)),**ax)),
            paper_bgcolor=PANEL, font_color=TEXT,
            margin=dict(l=0,r=0,b=60,t=35), showlegend=False,
            title=dict(text="Unipolar potential – t = 0.0 ms", font=dict(color=TEXT,size=12)),
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.05, y=0.02, xanchor="left", yanchor="bottom",
                buttons=[
                    dict(label="▶",
                         method="animate",
                         args=[None, {"frame":{"duration":interval_ms,"redraw":True},
                                      "fromcurrent":True,"transition":{"duration":0}}]),
                    dict(label="⏸",
                         method="animate",
                         args=[[None], {"frame":{"duration":0,"redraw":False},
                                        "mode":"immediate","transition":{"duration":0}}]),
                ],
                bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT, size=12),
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="t (ms): ", font=dict(color=TEXT,size=11)),
                pad=dict(b=5,t=5),
                steps=slider_steps,
                bgcolor=BORDER, font=dict(color=TEXT,size=9),
            )],
        )
        n_computed = len(frame_indices)
        return fig, f"✅ {n_computed} frames computed"

    except Exception as e:
        import traceback; traceback.print_exc()
        return no_update, f"❌ {e}"


# ── Video: legacy interval play/pause (unused but keeps IDs alive) ─────────────
@app.callback(
    Output("video-interval","disabled"),
    Output("store-video-state","data"),
    Input("btn-video-play","n_clicks"),
    Input("btn-video-pause","n_clicks"),
    Input("btn-video-reset","n_clicks"),
    State("store-video-state","data"),
    prevent_initial_call=True,
)
def control_video(play_n, pause_n, reset_n, state):
    state = state or {"playing":False,"frame":0,"max_frame":100}
    tid = ctx.triggered_id

    if tid == "btn-video-play":
        state["playing"] = True
        return False, state

    if tid == "btn-video-pause":
        state["playing"] = False
        return True, state

    if tid == "btn-video-reset":
        state["playing"] = False
        state["frame"]   = 0
        if SIGNALS is not None:
            dt  = SIGNALS["data_table"]
            rov = SIGNALS["signals"]["rov trace"]
            if len(dt):
                idx0 = dt.index[0]
                sig  = get_signal_array(rov.loc[idx0])
                state["max_frame"] = len(sig) - 1
        return True, state

    return no_update, state














# ── Export modal open/close ────────────────────────────────────────────────────
@app.callback(
    Output("export-modal","is_open"),
    Input("btn-export-open","n_clicks"),
    Input("btn-export-close","n_clicks"),
    State("export-modal","is_open"),
    prevent_initial_call=True,
)
def toggle_export_modal(n_open, n_close, is_open):
    return not is_open


# ── Export to CSV + PNG (bundled into a ZIP archive) ──────────────────────────
# Reviewer R1.17 / R2.8 (SoftwareX): export formats must be explicit. Parameter
# maps are written as CSV; figures are exported as PNG via Plotly + kaleido.
# All selected outputs are packaged into a single ZIP archive for convenience.
@app.callback(
    Output("download-export","data"),
    Output("export-status","children"),
    Input("btn-export-run","n_clicks"),
    State("export-checklist","value"),
    State("geo-graph","figure"),
    State("uni-graph","figure"),
    State("bip-graph","figure"),
    State("omni-signals-graph","figure"),
    State("omni-loops-graph","figure"),
    prevent_initial_call=True,
)
def run_export(n_clicks, selected, fig_geo, fig_uni, fig_bip,
               fig_omni_sig, fig_omni_loops):
    if not n_clicks or not selected:
        return no_update, "Select at least one option"
    if SIGNALS is None:
        return no_update, "❌ No signals loaded"

    try:
        import io as _io
        import zipfile as _zip

        dt  = SIGNALS["data_table"]
        rov = SIGNALS["signals"]["rov trace"]

        def sig_cols(rov_row):
            return rov_row.drop(["label","x","y"], errors="ignore").astype(float).values

        # Build aligned arrays using dt.index as the source of truth
        # so positional index i always matches rov.loc[idx]
        dt_aligned  = dt.copy()
        rov_aligned = rov.loc[dt_aligned.index]

        xs     = dt_aligned["roving x"].astype(float).values
        ys     = dt_aligned["roving y"].astype(float).values
        zs     = dt_aligned["roving z"].astype(float).values
        labels = rov_aligned["label"].astype(str).values
        groups = dt_aligned["pt number"].values
        filt_fn = lambda s: s

        # ────────────────────────────────────────────────────────────────
        # Build CSV/PNG payloads in memory
        # ────────────────────────────────────────────────────────────────
        csv_files = {}   # filename -> bytes
        png_files = {}   # filename -> bytes
        report    = []   # short tag per produced output

        # ── Parameter maps (CSV) ───────────────────────────────────────
        if "vpp_uni" in selected:
            rows = []
            for i, idx in enumerate(dt_aligned.index):
                try:
                    v = compute_vpp(sig_cols(rov_aligned.loc[idx]))
                except Exception:
                    v = np.nan
                rows.append({
                    "Group": groups[i], "Label": labels[i],
                    "x": xs[i], "y": ys[i], "z": zs[i],
                    "Vpp_uni(mV)": v,
                })
            csv_files["vpp_unipolar.csv"] = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
            report.append("Vpp_uni")

        if "vpp_bip" in selected:
            rows = []
            for g in dt_aligned["pt number"].unique():
                gmask   = dt_aligned["pt number"] == g
                g_rov   = rov_aligned.loc[dt_aligned[gmask].index]
                metrics = compute_group_metrics(g_rov, filt_fn)
                h, v_b, _ = compute_bipolars(metrics)
                for lbl, sig in {**h, **v_b}.items():
                    rows.append({"Group": g, "Label": lbl, "Vpp_bip(mV)": compute_vpp(sig)})
            csv_files["vpp_bipolar.csv"] = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
            report.append("Vpp_bip")

        if "vpp_omni" in selected:
            rows = []
            for g in dt_aligned["pt number"].unique():
                gmask   = dt_aligned["pt number"] == g
                g_rov   = rov_aligned.loc[dt_aligned[gmask].index]
                metrics = compute_group_metrics(g_rov, filt_fn)
                _, _, lmap = compute_bipolars(metrics)
                tri_od, cross_od = compute_omnipoles_for_group(metrics, lmap)
                for k, v in {**tri_od, **cross_od}.items():
                    cfg_name = k[0] if isinstance(k, tuple) and len(k) == 3 else "cross"
                    rows.append({
                        "Group": g, "Config": cfg_name,
                        "Labels": str(v["labels"]),
                        "Vpp_omni(mV)": v["vpp_omni"],
                        "ROR": v["vpp_res"] / v["vpp_omni"] if v["vpp_omni"] > 1e-9 else np.nan,
                    })
            csv_files["vpp_omnipolar_ROR.csv"] = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
            report.append("Vpp_omni+ROR")

        # ── Raw signals (CSV, wide format: one row per electrode/clique) ─
        if "sig_uni" in selected:
            sigs_arr = []
            for idx in dt_aligned.index:
                try:
                    sigs_arr.append(sig_cols(rov_aligned.loc[idx]))
                except Exception:
                    sigs_arr.append(np.full(100, np.nan))
            n_samp = max(len(s) for s in sigs_arr) if sigs_arr else 0
            sig_data = {
                "Group": groups, "Label": labels,
                "x": xs, "y": ys, "z": zs,
            }
            for j in range(n_samp):
                sig_data[f"t{j}"] = [s[j] if j < len(s) else np.nan for s in sigs_arr]
            csv_files["signals_unipolar.csv"] = pd.DataFrame(sig_data).to_csv(index=False).encode("utf-8")
            report.append("Sig_uni")

        if "sig_bip" in selected:
            rows_meta = []
            sigs_list = []
            for g in dt_aligned["pt number"].unique():
                gmask   = dt_aligned["pt number"] == g
                g_rov   = rov_aligned.loc[dt_aligned[gmask].index]
                metrics = compute_group_metrics(g_rov, filt_fn)
                h, v_b, _ = compute_bipolars(metrics)
                for lbl, sig in {**h, **v_b}.items():
                    rows_meta.append({"Group": g, "Label": lbl})
                    sigs_list.append(sig)
            if sigs_list:
                n_samp = max(len(s) for s in sigs_list)
                df_meta = pd.DataFrame(rows_meta)
                for j in range(n_samp):
                    df_meta[f"t{j}"] = [s[j] if j < len(s) else np.nan for s in sigs_list]
                csv_files["signals_bipolar.csv"] = df_meta.to_csv(index=False).encode("utf-8")
                report.append("Sig_bip")

        if "sig_omni" in selected:
            rows_meta = []
            sigs_omni = []
            sigs_res  = []
            for g in dt_aligned["pt number"].unique():
                gmask   = dt_aligned["pt number"] == g
                g_rov   = rov_aligned.loc[dt_aligned[gmask].index]
                metrics = compute_group_metrics(g_rov, filt_fn)
                _, _, lmap = compute_bipolars(metrics)
                tri_od, cross_od = compute_omnipoles_for_group(metrics, lmap)
                for k, v in {**tri_od, **cross_od}.items():
                    cfg_name = k[0] if isinstance(k, tuple) and len(k) == 3 else "cross"
                    rows_meta.append({"Group": g, "Config": cfg_name, "Labels": str(v["labels"])})
                    sigs_omni.append(v["omni"])
                    sigs_res.append(v["residue"])
            if sigs_omni:
                n_samp = max(len(s) for s in sigs_omni)
                df_o = pd.DataFrame(rows_meta)
                for j in range(n_samp):
                    df_o[f"omni_t{j}"] = [s[j] if j < len(s) else np.nan for s in sigs_omni]
                    df_o[f"res_t{j}"]  = [s[j] if j < len(s) else np.nan for s in sigs_res]
                csv_files["signals_omnipolar.csv"] = df_o.to_csv(index=False).encode("utf-8")
                report.append("Sig_omni")

        # ── Figures (PNG via Plotly + kaleido) ────────────────────────
        figure_map = {
            "fig_3d":  [("mesh_3d.png",        fig_geo)],
            "fig_signals": [
                ("signals_unipolar.png",   fig_uni),
                ("signals_bipolar.png",    fig_bip),
                ("signals_omnipolar.png",  fig_omni_sig),
                ("bipolar_loops.png",      fig_omni_loops),
            ],
        }
        kaleido_warning = None
        for key, items in figure_map.items():
            if key not in selected:
                continue
            for fname, fig_dict in items:
                if not fig_dict:
                    continue
                try:
                    fig_obj = go.Figure(fig_dict)
                    png_bytes = fig_obj.to_image(format="png", scale=2)
                    png_files[fname] = png_bytes
                    report.append(fname.replace(".png", ""))
                except Exception as e:
                    # kaleido may not be installed locally; report once
                    kaleido_warning = (
                        "PNG export requires the `kaleido` package. "
                        f"Install with `pip install -U kaleido`. ({e})"
                    )

        # Nothing produced
        if not csv_files and not png_files:
            return no_update, (kaleido_warning or
                               "Nothing exported — please select at least one option.")

        # ── Bundle everything into a single ZIP ────────────────────────
        buf = _io.BytesIO()
        with _zip.ZipFile(buf, "w", _zip.ZIP_DEFLATED) as zf:
            for name, data in csv_files.items():
                zf.writestr(name, data)
            for name, data in png_files.items():
                zf.writestr(name, data)

        msg = f"✅ Exported: {', '.join(report)}"
        if kaleido_warning and not png_files:
            msg += f"  ⚠ {kaleido_warning}"

        return (
            dcc.send_bytes(buf.getvalue(), "egm_analyzer_export.zip"),
            msg,
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return no_update, f"❌ {e}"


if __name__ == "__main__":
    import sys, os
    # Cross-platform: use app.run() (Dash >= 2.x) with host 0.0.0.0 for Docker/Linux compatibility
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("DASH_PORT", "8050"))
    debug = "--debug" in sys.argv or os.environ.get("DASH_DEBUG", "0") == "1"
    app.run(debug=debug, host=host, port=port)
