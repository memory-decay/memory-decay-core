"""Memory Decay Experiment Dashboard.

Dash/Plotly application for visualizing 440+ memory decay experiments
across two research eras (memories_500 and LongMemEval).

Features:
- Sidebar: era selector, status multi-select filter, text search
- Leaderboard table: sortable columns, status badges, best experiment highlight
- Detail view: same-page overlay with metrics, params, hypothesis, CV, snapshots
- Phase timeline: horizontal bar chart with clickable phase bars
- Metric progression: line charts with phase background shading
- Threshold heatmap: recall/precision at thresholds 0.1-0.9
- Retention curve overlay: multi-experiment comparison
- URL state management via dcc.Location for bookmarkability
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import dash
import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash, html, dcc
from dash import ctx as callback_ctx

from dashboard.data_loader import Experiment, load_all_experiments
from dashboard import charts

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = str(PROJECT_ROOT / "experiments")

# Status badge colors (distinct, colorblind-safe)
STATUS_COLORS: dict[str, dict[str, str]] = {
    "completed": {"bg": "#d4edda", "text": "#155724", "icon": "✓"},
    "no_results": {"bg": "#e2e3e5", "text": "#383d41", "icon": "—"},
    "validation_failed": {"bg": "#f8d7da", "text": "#721c24", "icon": "✗"},
    "parse_error": {"bg": "#fff3cd", "text": "#856404", "icon": "⚠"},
    "baseline": {"bg": "#cce5ff", "text": "#004085", "icon": "●"},
    "improved": {"bg": "#d4edda", "text": "#155724", "icon": "↑"},
    "not_improved": {"bg": "#fff3cd", "text": "#856404", "icon": "→"},
    "rejected": {"bg": "#f8d7da", "text": "#721c24", "icon": "↓"},
    "accepted_cv": {"bg": "#d1ecf1", "text": "#0c5460", "icon": "★"},
    "rejected_cv": {"bg": "#e2e3e5", "text": "#383d41", "icon": "✗"},
    "recorded": {"bg": "#e2e3e5", "text": "#383d41", "icon": "●"},
}

# History-based status values (from history.jsonl status field)
HISTORY_STATUSES = [
    "baseline", "improved", "not_improved", "rejected",
    "accepted_cv", "rejected_cv", "recorded",
]

ERA_OPTIONS = [
    {"label": "All", "value": "All"},
    {"label": "memories_500", "value": "memories_500"},
    {"label": "LongMemEval", "value": "LongMemEval"},
]

# AG Grid row height and pagination
ROW_HEIGHT = 36
PAGE_SIZE = 50

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def _load_cv_data(experiments: list[Experiment]) -> dict[str, dict]:
    """Load cv_results.json for experiments that have it."""
    cv_data: dict[str, dict] = {}
    for exp in experiments:
        cv_path = os.path.join(exp.dir_path, "cv_results.json")
        if os.path.exists(cv_path):
            try:
                with open(cv_path, "r", encoding="utf-8") as f:
                    cv_data[exp.id] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    return cv_data


def _load_history_status(experiments: list[Experiment]) -> dict[str, str]:
    """Load status from history.jsonl for display."""
    history_path = os.path.join(EXPERIMENTS_DIR, "history.jsonl")
    if not os.path.exists(history_path):
        return {}

    status_map: dict[str, str] = {}
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                exp_key = entry.get("exp")
                if exp_key and entry.get("status"):
                    status_map[exp_key] = entry["status"]
    except IOError:
        pass
    return status_map


def _find_best_experiment(
    experiments: list[Experiment],
    cv_data: dict[str, dict],
) -> str | None:
    """Find best experiment: highest CV mean overall_score, fallback to overall_score."""
    best_id = None
    best_score = -1.0

    # Prefer CV mean
    for exp_id, cv in cv_data.items():
        mean = cv.get("mean", {})
        if isinstance(mean, dict):
            score = mean.get("overall_score")
        else:
            score = mean
        if score is not None and score > best_score:
            best_score = score
            best_id = exp_id

    if best_id is not None:
        return best_id

    # Fallback to overall_score
    for exp in experiments:
        if exp.overall_score is not None and exp.overall_score > best_score:
            best_score = exp.overall_score
            best_id = exp.id

    return best_id


def _build_dataframe(
    experiments: list[Experiment],
    history_status: dict[str, str],
) -> pd.DataFrame:
    """Convert experiments list to DataFrame for table display."""
    rows = []
    for exp in experiments:
        display_status = history_status.get(exp.id, exp.status)
        rows.append({
            "id": exp.id,
            "era": exp.era,
            "phase": exp.phase,
            "overall_score": exp.overall_score,
            "retrieval_score": exp.retrieval_score,
            "plausibility_score": exp.plausibility_score,
            "status": display_status,
            "raw_status": exp.status,
            "hypothesis": (exp.hypothesis[:120] + "...") if exp.hypothesis and len(exp.hypothesis) > 120 else (exp.hypothesis or ""),
            "error": exp.error,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    title="Memory Decay Experiment Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
)

# Load data at module level
_experiments = load_all_experiments(EXPERIMENTS_DIR)
_cv_data = _load_cv_data(_experiments)
_history_status = _load_history_status(_experiments)
_best_exp_id = _find_best_experiment(_experiments, _cv_data)
_df_all = _build_dataframe(_experiments, _history_status)

# All unique statuses from data
_all_status_values = sorted(set(_df_all["status"].unique()) | set(HISTORY_STATUSES))

# ---------------------------------------------------------------------------
# AG Grid column definitions
# ---------------------------------------------------------------------------

_GRID_COLUMN_DEFS = [
    {
        "headerName": "Experiment",
        "field": "id",
        "filter": True,
        "sortable": True,
        "minWidth": 160,
        "cellStyle": {"fontWeight": "600", "fontSize": "13px"},
    },
    {
        "headerName": "Overall",
        "field": "overall_score",
        "filter": "agNumberColumnFilter",
        "sortable": True,
        "minWidth": 100,
        "valueFormatter": {"function": "params.value !== null ? d3.format('.4f')(params.value) : 'N/A'"},
    },
    {
        "headerName": "Retrieval",
        "field": "retrieval_score",
        "filter": "agNumberColumnFilter",
        "sortable": True,
        "minWidth": 100,
        "valueFormatter": {"function": "params.value !== null ? d3.format('.4f')(params.value) : 'N/A'"},
    },
    {
        "headerName": "Plausibility",
        "field": "plausibility_score",
        "filter": "agNumberColumnFilter",
        "sortable": True,
        "minWidth": 110,
        "valueFormatter": {"function": "params.value !== null ? d3.format('.4f')(params.value) : 'N/A'"},
    },
    {
        "headerName": "Status",
        "field": "status",
        "filter": True,
        "sortable": True,
        "minWidth": 140,
        "cellStyle": {
            "function": (
                "const colors = {"
                "  completed: {bg: '#d4edda', text: '#155724'},"
                "  no_results: {bg: '#e2e3e5', text: '#383d41'},"
                "  validation_failed: {bg: '#f8d7da', text: '#721c24'},"
                "  baseline: {bg: '#cce5ff', text: '#004085'},"
                "  improved: {bg: '#d4edda', text: '#155724'},"
                "  not_improved: {bg: '#fff3cd', text: '#856404'},"
                "  rejected: {bg: '#f8d7da', text: '#721c24'},"
                "  accepted_cv: {bg: '#d1ecf1', text: '#0c5460'},"
                "  rejected_cv: {bg: '#e2e3e5', text: '#383d41'},"
                "  recorded: {bg: '#e2e3e5', text: '#383d41'},"
                "  parse_error: {bg: '#fff3cd', text: '#856404'},"
                "};"
                "const c = colors[params.value] || colors.completed;"
                "return {backgroundColor: c.bg, color: c.text, borderRadius: '12px', padding: '2px 8px', fontSize: '11px', fontWeight: '600', display: 'inline-block', textAlign: 'center'};"
            )
        },
    },
    {
        "headerName": "Hypothesis",
        "field": "hypothesis",
        "filter": True,
        "sortable": True,
        "minWidth": 300,
        "flex": 1,
        "cellStyle": {"color": "#6c757d", "fontSize": "12px"},
        "tooltipField": "hypothesis",
    },
]

# Best experiment row style
_GRID_ROW_CLASS_RULES = {
    "best-experiment": {
        "function": f"return params.data.id === '{_best_exp_id or ''}';"
    }
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "fontFamily": "system-ui, -apple-system, sans-serif"},
    children=[
        # URL state
        dcc.Location(id="url", refresh=False),

        # Stores
        dcc.Store(id="selected-experiment", data=None),
        dcc.Store(id="sort-state", data={"column": "overall_score", "ascending": False}),
        dcc.Store(id="selected-phase", data=None),
        dcc.Store(id="active-page", data="leaderboard"),
        dcc.Store(id="retention-selected", data=[]),
        dcc.Store(id="snapshot-selected-exp", data=None),
        dcc.Store(id="snapshot-current-tick", data=0),
        dcc.Store(id="snapshot-animation-state", data={"playing": False, "interval_id": None}),
        dcc.Store(id="cv-selected-exp", data=None),
        dcc.Store(id="compare-phase-a", data=None),
        dcc.Store(id="compare-phase-b", data=None),
        dcc.Store(id="detail-source-page", data="leaderboard"),

        # Sidebar
        html.Div(
            id="sidebar",
            style={
                "width": "280px",
                "minWidth": "280px",
                "backgroundColor": "#f8f9fa",
                "borderRight": "1px solid #dee2e6",
                "padding": "20px 16px",
                "overflowY": "auto",
                "display": "flex",
                "flexDirection": "column",
                "gap": "20px",
            },
            children=[
                # Title
                html.H2(
                    "📊 Memory Decay",
                    style={"margin": "0 0 4px 0", "fontSize": "18px", "color": "#212529"},
                ),
                html.P(
                    "Experiment Dashboard",
                    style={"margin": "0 0 16px 0", "fontSize": "12px", "color": "#6c757d"},
                ),

                # Era selector
                html.Div([
                    html.Label("Era", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px", "display": "block"}),
                    dcc.Dropdown(
                        id="era-dropdown",
                        options=ERA_OPTIONS,
                        value="All",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                ]),

                # Status filter
                html.Div([
                    html.Label(
                        "Status",
                        style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px", "display": "block"},
                    ),
                    dcc.Dropdown(
                        id="status-filter",
                        options=[{"label": s, "value": s} for s in _all_status_values],
                        value=[],
                        multi=True,
                        placeholder="Filter by status...",
                        style={"fontSize": "13px"},
                    ),
                ]),

                # Text search
                html.Div([
                    html.Label(
                        "Search",
                        style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px", "display": "block"},
                    ),
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="ID or hypothesis...",
                        style={
                            "width": "100%",
                            "padding": "6px 10px",
                            "borderRadius": "4px",
                            "border": "1px solid #ced4da",
                            "fontSize": "13px",
                            "boxSizing": "border-box",
                        },
                    ),
                ]),

                # Experiment count
                html.Div(
                    id="experiment-count",
                    style={
                        "marginTop": "auto",
                        "paddingTop": "16px",
                        "borderTop": "1px solid #dee2e6",
                        "fontSize": "12px",
                        "color": "#6c757d",
                    },
                    children=["Loading..."],
                ),

                # Status legend
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "4px",
                        "fontSize": "11px",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "6px"},
                            children=[
                                html.Span(style={"display": "inline-block", "width": "10px", "height": "10px", "borderRadius": "2px", "backgroundColor": c["bg"], "border": f"1px solid {c['text']}"}),
                                html.Span(s, style={"color": "#495057"}),
                            ]
                        )
                        for s, c in STATUS_COLORS.items()
                    ],
                ),
            ],
        ),

        # Main content
        html.Div(
            id="main-content",
            style={"flex": "1", "overflow": "auto", "position": "relative"},
            children=[
                # Navigation tabs
                html.Div(
                    style={
                        "display": "flex",
                        "borderBottom": "1px solid #dee2e6",
                        "backgroundColor": "white",
                        "padding": "0 24px",
                        "position": "sticky",
                        "top": 0,
                        "zIndex": 5,
                    },
                    children=[
                        html.Button(
                            "📊 Leaderboard",
                            id="tab-leaderboard",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid #1565C0",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "color": "#1565C0",
                                "marginRight": "4px",
                            },
                        ),
                        html.Button(
                            "📈 Timeline & Metrics",
                            id="tab-timeline",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid transparent",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "color": "#6c757d",
                                "marginRight": "4px",
                            },
                        ),
                        html.Button(
                            "🗺️ Threshold Heatmap",
                            id="tab-heatmap",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid transparent",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "color": "#6c757d",
                                "marginRight": "4px",
                            },
                        ),
                        html.Button(
                            "📉 Retention Curves",
                            id="tab-retention",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid transparent",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "color": "#6c757d",
                                "marginRight": "4px",
                            },
                        ),
                        html.Button(
                            "🧬 Parameter Sweep",
                            id="tab-params",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid transparent",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "color": "#6c757d",
                                "marginRight": "4px",
                            },
                        ),
                        html.Button(
                            "🎥 Snapshot Viewer",
                            id="tab-snapshots",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid transparent",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "color": "#6c757d",
                                "marginRight": "4px",
                            },
                        ),
                        html.Button(
                            "🧪 Forgetting & CV",
                            id="tab-analysis",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid transparent",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "color": "#6c757d",
                                "marginRight": "4px",
                            },
                        ),
                        html.Button(
                            "⚖️ Phase Compare",
                            id="tab-compare",
                            n_clicks=0,
                            style={
                                "padding": "12px 20px",
                                "border": "none",
                                "borderBottom": "2px solid transparent",
                                "backgroundColor": "transparent",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "color": "#6c757d",
                            },
                        ),
                    ],
                ),

                # Leaderboard view
                html.Div(
                    id="leaderboard-view",
                    children=[
                        html.Div(
                            style={
                                "padding": "16px 24px 0 24px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "space-between",
                            },
                            children=[
                                html.H1(
                                    "Experiment Leaderboard",
                                    style={"margin": 0, "fontSize": "22px", "color": "#212529"},
                                ),
                                html.Div(
                                    id="sort-indicator",
                                    style={"fontSize": "12px", "color": "#6c757d"},
                                    children=["Click column headers to sort"],
                                ),
                            ],
                        ),
                        html.Div(
                            id="table-container",
                            style={"padding": "12px 24px 24px 24px", "height": "calc(100vh - 60px)"},
                            children=[
                                dag.AgGrid(
                                    id="leaderboard-grid",
                                    columnDefs=_GRID_COLUMN_DEFS,
                                    rowData=[],
                                    defaultColDef={
                                        "resizable": True,
                                        "filterParams": {"buttons": ["reset", "apply"]},
                                    },
                                    rowClassRules=_GRID_ROW_CLASS_RULES,
                                    getRowId="params.data.id",
                                    style={"height": "100%"},
                                    dashGridOptions={
                                        "suppressCellFocus": True,
                                        "enableCellTextSelection": True,
                                        "domLayout": "autoHeight",
                                        "pagination": True,
                                        "paginationPageSize": PAGE_SIZE,
                                        "paginationPageSizeSelector": [25, 50, 100, 200],
                                        "animateRows": False,
                                        "rowSelection": "single",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                # Timeline & Metrics view
                html.Div(
                    id="timeline-view",
                    style={"display": "none", "padding": "24px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "16px"},
                            children=[
                                html.H1("Phase Timeline & Metric Progression", style={"margin": 0, "fontSize": "22px", "color": "#212529"}),
                                html.Div(
                                    id="phase-clear-btn-container",
                                    style={"display": "none"},
                                    children=[
                                        html.Button(
                                            "✕ Clear Phase Filter",
                                            id="phase-clear-btn",
                                            n_clicks=0,
                                            style={
                                                "padding": "6px 12px",
                                                "border": "1px solid #dee2e6",
                                                "borderRadius": "4px",
                                                "backgroundColor": "#f8f9fa",
                                                "cursor": "pointer",
                                                "fontSize": "12px",
                                                "color": "#495057",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Phase timeline chart
                        html.Div(
                            id="phase-timeline-container",
                            style={"marginBottom": "24px"},
                            children=[dcc.Graph(id="phase-timeline-graph", config={"displayModeBar": False})],
                        ),
                        # Metric progression charts
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(1, 1fr)", "gap": "16px"},
                            children=[
                                dcc.Graph(id="metric-overall-graph", config={"displayModeBar": True}),
                                dcc.Graph(id="metric-retrieval-graph", config={"displayModeBar": True}),
                                dcc.Graph(id="metric-plausibility-graph", config={"displayModeBar": True}),
                            ],
                        ),
                    ],
                ),

                # Threshold heatmap view
                html.Div(
                    id="heatmap-view",
                    style={"display": "none", "padding": "24px"},
                    children=[
                        html.H1("Threshold Metric Heatmap", style={"margin": "0 0 16px 0", "fontSize": "22px", "color": "#212529"}),
                        html.P(
                            "Recall and precision rates at each activation threshold (0.1–0.9). "
                            "Blank cells indicate missing thresholds (old-era experiments have only 0.2–0.5).",
                            style={"marginBottom": "16px", "fontSize": "13px", "color": "#6c757d"},
                        ),
                        dcc.Graph(id="heatmap-recall-graph", config={"displayModeBar": True}),
                        html.Div(style={"height": "24px"}),
                        dcc.Graph(id="heatmap-precision-graph", config={"displayModeBar": True}),
                    ],
                ),

                # Retention curve view
                html.Div(
                    id="retention-view",
                    style={"display": "none", "padding": "24px"},
                    children=[
                        html.H1("Retention Curve Overlay", style={"margin": "0 0 16px 0", "fontSize": "22px", "color": "#212529"}),
                        html.P(
                            "Compare retention curves across experiments. "
                            "Select 2–5 experiments to overlay their retention profiles at ticks 40, 80, 120, 160, 200.",
                            style={"marginBottom": "16px", "fontSize": "13px", "color": "#6c757d"},
                        ),
                        # Retention experiment selector
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
                            children=[
                                html.Div(
                                    style={"flex": "1"},
                                    children=[
                                        dcc.Dropdown(
                                            id="retention-dropdown",
                                            options=[],
                                            value=[],
                                            multi=True,
                                            placeholder="Select 2–5 experiments with retention data...",
                                            style={"fontSize": "13px"},
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Clear All",
                                    id="retention-clear-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 16px",
                                        "border": "1px solid #dee2e6",
                                        "borderRadius": "4px",
                                        "backgroundColor": "#f8f9fa",
                                        "cursor": "pointer",
                                        "fontSize": "13px",
                                        "color": "#495057",
                                    },
                                ),
                            ],
                        ),
                        # Warning container
                        html.Div(id="retention-warning", style={"marginBottom": "12px"}),
                        # Retention count display
                        html.Div(
                            id="retention-count",
                            style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "16px"},
                            children=["Select experiments to compare"],
                        ),
                        # Retention chart
                        dcc.Graph(id="retention-graph", config={"displayModeBar": True}),
                    ],
                ),

                # Parameter Sweep view
                html.Div(
                    id="params-view",
                    style={"display": "none", "padding": "24px"},
                    children=[
                        html.H1("Parameter Sweep", style={"margin": "0 0 16px 0", "fontSize": "22px", "color": "#212529"}),
                        html.P(
                            "Parallel coordinates showing parameter values colored by overall_score. "
                            "Toggle dimensions with the selector below. Click an experiment in the dropdown to view details.",
                            style={"marginBottom": "16px", "fontSize": "13px", "color": "#6c757d"},
                        ),
                        # Param toggle selector
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
                            children=[
                                html.Div(
                                    style={"flex": "1"},
                                    children=[
                                        dcc.Dropdown(
                                            id="param-dimension-selector",
                                            options=[],
                                            value=[],
                                            multi=True,
                                            placeholder="Select dimensions to display...",
                                            style={"fontSize": "13px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Experiment selector for detail navigation
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
                            children=[
                                html.Div(
                                    style={"flex": "1"},
                                    children=[
                                        dcc.Dropdown(
                                            id="param-exp-detail-selector",
                                            options=[],
                                            value=None,
                                            placeholder="Select experiment to view details...",
                                            style={"fontSize": "13px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="param-sweep-graph", config={"displayModeBar": True}),
                    ],
                ),

                # Snapshot Viewer view
                html.Div(
                    id="snapshots-view",
                    style={"display": "none", "padding": "24px"},
                    children=[
                        html.H1("Snapshot Viewer", style={"margin": "0 0 16px 0", "fontSize": "22px", "color": "#212529"}),
                        html.P(
                            "Expanded view of simulation snapshots across 11 ticks. "
                            "Shows multiple metric traces with stepping and animation controls.",
                            style={"marginBottom": "16px", "fontSize": "13px", "color": "#6c757d"},
                        ),
                        # Experiment selector
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
                            children=[
                                dcc.Dropdown(
                                    id="snapshot-exp-selector",
                                    options=[],
                                    value=None,
                                    placeholder="Select experiment...",
                                    style={"fontSize": "13px", "flex": "1"},
                                ),
                            ],
                        ),
                        # Stepping controls
                        html.Div(
                            id="snapshot-controls",
                            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "16px"},
                            children=[
                                html.Button("⏮", id="snapshot-first", n_clicks=0, title="First tick",
                                    style={"padding": "6px 12px", "border": "1px solid #dee2e6", "borderRadius": "4px", "backgroundColor": "#f8f9fa", "cursor": "pointer", "fontSize": "14px"}),
                                html.Button("◀", id="snapshot-prev", n_clicks=0, title="Previous tick",
                                    style={"padding": "6px 12px", "border": "1px solid #dee2e6", "borderRadius": "4px", "backgroundColor": "#f8f9fa", "cursor": "pointer", "fontSize": "14px"}),
                                html.Button("▶", id="snapshot-next", n_clicks=0, title="Next tick",
                                    style={"padding": "6px 12px", "border": "1px solid #dee2e6", "borderRadius": "4px", "backgroundColor": "#f8f9fa", "cursor": "pointer", "fontSize": "14px"}),
                                html.Button("⏭", id="snapshot-last", n_clicks=0, title="Last tick",
                                    style={"padding": "6px 12px", "border": "1px solid #dee2e6", "borderRadius": "4px", "backgroundColor": "#f8f9fa", "cursor": "pointer", "fontSize": "14px"}),
                                html.Button("⏯ Animate", id="snapshot-animate-btn", n_clicks=0, title="Play/Pause animation",
                                    style={"padding": "6px 16px", "border": "1px solid #1565C0", "borderRadius": "4px", "backgroundColor": "#E3F2FD", "cursor": "pointer", "fontSize": "13px", "color": "#1565C0", "marginLeft": "8px"}),
                                html.Span(
                                    id="snapshot-tick-label",
                                    style={"marginLeft": "12px", "fontSize": "13px", "color": "#495057", "fontWeight": "600"},
                                    children=["Tick: —"],
                                ),
                                dcc.Interval(id="snapshot-interval", interval=1500, n_intervals=0, disabled=True),
                            ],
                        ),
                        # Tick metric cards
                        html.Div(id="snapshot-tick-cards", style={"marginBottom": "16px"}),
                        # Snapshot chart
                        dcc.Graph(id="snapshot-viewer-graph", config={"displayModeBar": True}),
                    ],
                ),

                # Forgetting Depth & CV Analysis view
                html.Div(
                    id="analysis-view",
                    style={"display": "none", "padding": "24px"},
                    children=[
                        html.H1("Forgetting Depth & Cross-Validation Analysis", style={"margin": "0 0 16px 0", "fontSize": "22px", "color": "#212529"}),
                        html.P(
                            "Analyze forgetting depth and strict validation results across experiments. "
                            "View cross-validation fold performance for experiments with CV data.",
                            style={"marginBottom": "16px", "fontSize": "13px", "color": "#6c757d"},
                        ),
                        # Forgetting depth section
                        html.H2("Forgetting Depth & Strict Score", style={"fontSize": "18px", "color": "#212529", "margin": "24px 0 12px 0"}),
                        html.Div(
                            id="forgetting-depth-container",
                            style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px"},
                            children=[
                                dcc.Graph(id="forgetting-depth-graph", config={"displayModeBar": True}),
                                dcc.Graph(id="strict-score-graph", config={"displayModeBar": True}),
                            ],
                        ),
                        # CV section
                        html.H2("Cross-Validation Results", style={"fontSize": "18px", "color": "#212529", "margin": "24px 0 12px 0"}),
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
                            children=[
                                dcc.Dropdown(
                                    id="cv-exp-selector",
                                    options=[],
                                    value=None,
                                    placeholder="Select experiment with CV data...",
                                    style={"fontSize": "13px", "flex": "1"},
                                ),
                            ],
                        ),
                        html.Div(id="cv-results-container", children=[
                            dcc.Graph(id="cv-fold-scores-graph", config={"displayModeBar": True}),
                            html.Div(style={"height": "24px"}),
                            dcc.Graph(id="cv-fold-deltas-graph", config={"displayModeBar": True}),
                        ]),
                    ],
                ),

                # Phase Comparison view
                html.Div(
                    id="compare-view",
                    style={"display": "none", "padding": "24px"},
                    children=[
                        html.H1("Phase Comparison", style={"margin": "0 0 16px 0", "fontSize": "22px", "color": "#212529"}),
                        html.P(
                            "Side-by-side statistical comparison of two selected phases. "
                            "validation_failed experiments are excluded from all calculations.",
                            style={"marginBottom": "16px", "fontSize": "13px", "color": "#6c757d"},
                        ),
                        # Phase selectors
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px", "marginBottom": "24px"},
                            children=[
                                html.Div([
                                    html.Label("Phase A", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px", "display": "block"}),
                                    dcc.Dropdown(
                                        id="compare-phase-a-dropdown",
                                        options=[],
                                        value=None,
                                        placeholder="Select first phase...",
                                        style={"fontSize": "13px"},
                                    ),
                                ]),
                                html.Div([
                                    html.Label("Phase B", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px", "display": "block"}),
                                    dcc.Dropdown(
                                        id="compare-phase-b-dropdown",
                                        options=[],
                                        value=None,
                                        placeholder="Select second phase...",
                                        style={"fontSize": "13px"},
                                    ),
                                ]),
                            ],
                        ),
                        # Comparison results
                        html.Div(id="phase-comparison-results"),
                    ],
                ),

                # Detail view overlay
                html.Div(
                    id="detail-view",
                    style={
                        "display": "none",
                        "position": "absolute",
                        "top": 0,
                        "left": 0,
                        "right": 0,
                        "bottom": 0,
                        "backgroundColor": "white",
                        "zIndex": 100,
                        "overflow": "auto",
                    },
                    children=[],
                ),
            ],
        ),
    ],
)

# Add CSS for best experiment highlighting
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        .best-experiment {
            background-color: #fffde7 !important;
            border-left: 3px solid #ffd700 !important;
        }
        .best-experiment::after {
            content: "⭐";
            margin-left: 4px;
        }
        .ag-row {
            cursor: pointer;
        }
        .ag-header-cell-label {
            cursor: pointer;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
    <script>
    // Cross-area navigation: browser back/forward support for detail view
    (function() {
        var _pushOnNextDetail = true;

        // Monitor detail view visibility to push history entry
        var _observer = new MutationObserver(function(mutations) {
            var detailEl = document.getElementById('detail-view');
            if (!detailEl) return;
            var style = window.getComputedStyle(detailEl);
            var isVisible = style.display !== 'none';
            var expData = window.dash_clientside || {};

            if (isVisible && _pushOnNextDetail) {
                // Detail view opened: push a history entry so browser back closes it
                history.pushState({detailOpen: true}, '');
                _pushOnNextDetail = false;
            } else if (!isVisible && !_pushOnNextDetail) {
                // Detail view closed: we don't push, the history entry was already consumed
                _pushOnNextDetail = true;
            }
        });

        // Observe the detail-view style changes
        setTimeout(function() {
            var detailEl = document.getElementById('detail-view');
            if (detailEl) {
                _observer.observe(detailEl, {attributes: true, attributeFilter: ['style']});
            }
        }, 2000);

        // On browser back: if detail view is open, close it
        window.addEventListener('popstate', function(event) {
            var detailEl = document.getElementById('detail-view');
            if (!detailEl) return;
            var style = window.getComputedStyle(detailEl);
            if (style.display !== 'none') {
                // Find and click the back button to close detail view
                var backBtn = document.getElementById('back-button');
                if (backBtn) {
                    backBtn.click();
                }
                _pushOnNextDetail = true;
            }
        });
    })();
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helper functions for callbacks
# ---------------------------------------------------------------------------


def _format_score(val: float | None) -> str:
    """Format a score to 4 decimal places or return N/A."""
    if val is None:
        return "N/A"
    return f"{val:.4f}"


def _make_status_badge(status: str) -> str:
    """Create HTML badge for a status value."""
    colors = STATUS_COLORS.get(status, STATUS_COLORS.get("completed"))
    icon = colors["icon"]
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;'
        f'backgroundColor:{colors["bg"]};color:{colors["text"]};'
        f'white-space:nowrap;">{icon} {status}</span>'
    )


def _filter_dataframe(
    df: pd.DataFrame,
    era: str,
    statuses: list[str],
    search: str,
) -> pd.DataFrame:
    """Apply era, status, and search filters to DataFrame."""
    filtered = df.copy()

    # Era filter
    if era != "All":
        filtered = filtered[filtered["era"] == era]

    # Status filter (OR logic)
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]

    # Text search (AND with other filters)
    # For experiment ID: exact match (to avoid exp_lme_0008 matching exp_lme_00080)
    # For hypothesis: substring match
    if search:
        search_lower = search.lower()
        id_match = filtered["id"].str.lower() == search_lower
        hypothesis_match = filtered["hypothesis"].str.lower().str.contains(search_lower, na=False)
        filtered = filtered[id_match | hypothesis_match]

    return filtered


def _sort_dataframe(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool,
) -> pd.DataFrame:
    """Sort DataFrame by column with nulls last.

    Non-numeric values (validation_failed strings) and null metrics
    sort to the bottom regardless of ascending/descending.
    """
    df = df.copy()
    # Use pandas sort_values with na_position='last' for null handling
    # For numeric columns, na_position='last' puts NaN at bottom
    df = df.sort_values(
        by=sort_col, ascending=ascending, na_position="last", kind="mergesort"
    )
    return df


def _build_row_data(
    df: pd.DataFrame,
) -> list[dict]:
    """Convert DataFrame to AG Grid row data."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "id": row["id"],
            "era": row["era"],
            "phase": row["phase"],
            "overall_score": row["overall_score"],
            "retrieval_score": row["retrieval_score"],
            "plausibility_score": row["plausibility_score"],
            "status": row["status"],
            "hypothesis": row["hypothesis"],
        })
    return rows


def _build_metric_card(
    name: str,
    val: float | None,
    bg_color: str = "#f8f9fa",
    border_color: str = "#e9ecef",
    label_color: str = "#6c757d",
    value_color: str = "#212529",
    font_size: str = "20px",
) -> html.Div:
    """Build a single metric card for the detail view."""
    return html.Div(
        style={"padding": "12px", "backgroundColor": bg_color, "borderRadius": "6px", "border": f"1px solid {border_color}"},
        children=[
            html.Div(name, style={"fontSize": "11px", "color": label_color, "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(_format_score(val), style={"fontSize": font_size, "fontWeight": "700", "color": value_color, "marginTop": "4px"}),
        ],
    )


def _build_detail_view(exp_id: str, source_page: str = "leaderboard") -> list:
    """Build detail view content for an experiment."""
    exp = next((e for e in _experiments if e.id == exp_id), None)
    if exp is None:
        return [html.P(f"Experiment {exp_id} not found")]

    cv = _cv_data.get(exp_id)
    history_status_val = _history_status.get(exp_id)
    display_status = history_status_val or exp.status
    is_best = _best_exp_id == exp_id
    is_validation_failed = exp.status == "validation_failed"

    # Error section (shown first, prominently, for validation_failed)
    error_section = []
    if exp.error:
        error_section = [
            html.Div(
                style={"padding": "16px", "backgroundColor": "#f8d7da", "borderRadius": "6px", "border": "1px solid #f5c6cb", "marginBottom": "16px"},
                children=[
                    html.Div("⚠ Validation Error", style={"fontWeight": "700", "color": "#721c24", "marginBottom": "8px"}),
                    html.Div(exp.error, style={"color": "#721c24", "fontSize": "14px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
                ],
            ),
        ]

    # Key metrics — show N/A for all when validation_failed (VAL-TABLE-014)
    key_metrics = [
        ("Overall Score", None if is_validation_failed else exp.overall_score),
        ("Retrieval Score", None if is_validation_failed else exp.retrieval_score),
        ("Plausibility Score", None if is_validation_failed else exp.plausibility_score),
        ("Recall Mean", None if is_validation_failed else exp.recall_mean),
        ("Precision Mean", None if is_validation_failed else exp.precision_mean),
        ("MRR Mean", None if is_validation_failed else exp.mrr_mean),
        ("Correlation Score", None if is_validation_failed else exp.corr_score),
        ("Retention AUC", None if is_validation_failed else exp.retention_auc),
        ("Selectivity Score", None if is_validation_failed else exp.selectivity_score),
    ]

    metrics_html = [
        html.H3("Key Metrics", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))", "gap": "12px"},
            children=[
                _build_metric_card(name, val)
                for name, val in key_metrics
            ],
        ),
    ]

    # Later-era metrics — always show, with "N/A" for missing (VAL-TABLE-012)
    later_metrics = [
        ("Strict Score", None if is_validation_failed else exp.strict_score),
        ("Forgetting Depth", None if is_validation_failed else exp.forgetting_depth),
        ("Forgetting Score", None if is_validation_failed else exp.forgetting_score),
        ("Eval V2 Score", None if is_validation_failed else exp.eval_v2_score),
    ]
    later_html = [
        html.H3("Strict Validation & Extended Metrics", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))", "gap": "12px"},
            children=[
                _build_metric_card(
                    name, val,
                    bg_color="#fff8e1", border_color="#ffe082",
                    label_color="#856404",
                )
                for name, val in later_metrics
            ],
        ),
    ]

    # Hypothesis section (VAL-TABLE-013)
    hypothesis_section = [
        html.H3("Hypothesis", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
        html.Div(
            exp.hypothesis if exp.hypothesis else "Hypothesis not available",
            style={"padding": "16px", "backgroundColor": "#f8f9fa", "borderRadius": "6px", "border": "1px solid #e9ecef", "fontSize": "14px", "lineHeight": "1.6", "whiteSpace": "pre-wrap", "color": "#495057" if exp.hypothesis else "#adb5bd"},
        ),
    ]

    # Parameters section (VAL-TABLE-013)
    if exp.params:
        params_items = sorted(exp.params.items())
        params_section = [
            html.H3("Parameters", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div(
                style={"padding": "12px", "backgroundColor": "#f8f9fa", "borderRadius": "6px", "border": "1px solid #e9ecef"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "padding": "6px 0", "borderBottom": "1px solid #e9ecef"},
                        children=[
                            html.Span(k, style={"fontFamily": "monospace", "color": "#495057", "fontSize": "13px"}),
                            html.Span(str(v), style={"fontFamily": "monospace", "color": "#212529", "fontWeight": "600", "fontSize": "13px"}),
                        ],
                    )
                    for k, v in params_items
                ],
            ),
        ]
    else:
        params_section = [
            html.H3("Parameters", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div("Parameters not available", style={"color": "#adb5bd", "fontSize": "14px"}),
        ]

    # CV section (VAL-TABLE-015)
    cv_section = []
    if cv:
        cv_mean = cv.get("mean", {})
        cv_std = cv.get("std", {})
        cv_k = cv.get("k", "?")
        cv_worst = cv.get("worst_fold", {})
        fold_scores = cv.get("fold_scores", [])
        fold_deltas = cv.get("fold_deltas", [])

        # CV mean metrics
        cv_metric_cards = []
        if isinstance(cv_mean, dict):
            cv_metric_cards.extend([
                ("CV Mean (overall)", cv_mean.get("overall_score")),
                ("CV Mean (retrieval)", cv_mean.get("retrieval_score")),
                ("CV Mean (plausibility)", cv_mean.get("plausibility_score")),
            ])
        if isinstance(cv_std, dict):
            cv_metric_cards.extend([
                ("CV Std (overall)", cv_std.get("overall_score")),
                ("CV Std (retrieval)", cv_std.get("retrieval_score")),
            ])

        cv_section = [
            html.H3("Cross-Validation Results", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div(style={"marginBottom": "8px", "fontSize": "13px", "color": "#6c757d"}, children=[f"k = {cv_k} folds"]),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))", "gap": "12px", "marginBottom": "16px"},
                children=[
                    _build_metric_card(
                        name, val,
                        bg_color="#e8f5e9", border_color="#a5d6a7",
                        label_color="#2e7d32", font_size="18px",
                    )
                    for name, val in cv_metric_cards
                ],
            ),
        ]

        # Worst fold
        if cv_worst and isinstance(cv_worst, dict):
            worst_overall = cv_worst.get("overall_score")
            cv_section.append(
                html.Div(
                    style={"padding": "12px", "backgroundColor": "#fff3e0", "borderRadius": "6px", "border": "1px solid #ffcc80", "marginBottom": "16px"},
                    children=[
                        html.Div("Worst Fold", style={"fontSize": "11px", "color": "#e65100", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                        html.Div(_format_score(worst_overall), style={"fontSize": "18px", "fontWeight": "700", "color": "#212529", "marginTop": "4px"}),
                    ],
                ),
            )

        # Fold deltas
        if fold_deltas:
            delta_text = ", ".join(f"{d:+.4f}" for d in fold_deltas)
            cv_section.append(
                html.Div(
                    style={"padding": "12px", "backgroundColor": "#f8f9fa", "borderRadius": "6px", "border": "1px solid #e9ecef", "marginBottom": "16px"},
                    children=[
                        html.Div("Fold Deltas", style={"fontSize": "11px", "color": "#6c757d", "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "4px"}),
                        html.Div(delta_text, style={"fontFamily": "monospace", "fontSize": "13px", "color": "#495057"}),
                    ],
                ),
            )

        # Fold scores bar chart
        if fold_scores and isinstance(fold_scores, list):
            fold_overalls = []
            for i, fs in enumerate(fold_scores):
                if isinstance(fs, dict):
                    score = fs.get("overall_score")
                    if score is not None:
                        fold_overalls.append({"fold": f"Fold {i + 1}", "overall_score": score})

            if fold_overalls:
                fold_fig = go.Figure()
                fold_fig.add_trace(go.Bar(
                    x=[f["fold"] for f in fold_overalls],
                    y=[f["overall_score"] for f in fold_overalls],
                    marker_color=["#4CAF50" if s >= (cv_mean.get("overall_score") if isinstance(cv_mean, dict) else 0) else "#FF9800" for s in [f["overall_score"] for f in fold_overalls]],
                    text=[f"{s:.4f}" for s in [f["overall_score"] for f in fold_overalls]],
                    textposition="auto",
                ))
                # Add mean line
                if isinstance(cv_mean, dict) and cv_mean.get("overall_score") is not None:
                    fold_fig.add_hline(
                        y=cv_mean["overall_score"],
                        line_dash="dash", line_color="#2196F3",
                        annotation_text=f"Mean: {cv_mean['overall_score']:.4f}",
                        annotation_position="top right",
                    )
                fold_fig.update_layout(
                    title="Fold Scores (overall_score)",
                    yaxis_title="Score", yaxis={"range": [0, 1]},
                    height=250, margin={"l": 50, "r": 20, "t": 40, "b": 40},
                    template="plotly_white", font={"size": 12},
                    showlegend=False,
                )
                cv_section.append(dcc.Graph(figure=fold_fig, config={"displayModeBar": False}))

    # Snapshot mini-charts (VAL-TABLE-016, VAL-TABLE-017)
    snapshot_section = []
    if exp.snapshots:
        ticks = [s["tick"] for s in exp.snapshots if "tick" in s]
        if ticks:
            fig = go.Figure()
            trace_configs = [
                ("overall_score", "Overall Score", "#1565C0"),
                ("retrieval_score", "Retrieval Score", "#2E7D32"),
                ("plausibility_score", "Plausibility Score", "#E65100"),
            ]
            has_any_trace = False
            for key, label, color in trace_configs:
                values = [s.get(key) for s in exp.snapshots if s.get(key) is not None]
                t_vals = [s["tick"] for s in exp.snapshots if s.get(key) is not None]
                if values:
                    has_any_trace = True
                    fig.add_trace(go.Scatter(
                        x=t_vals, y=values, mode="lines+markers",
                        name=label,
                        line={"color": color, "width": 2.5},
                        marker={"size": 5},
                    ))
            if has_any_trace:
                fig.update_layout(
                    title="Metrics over Simulation Ticks",
                    xaxis_title="Tick", yaxis_title="Score", yaxis={"range": [0, 1]},
                    height=300, margin={"l": 50, "r": 20, "t": 40, "b": 40},
                    template="plotly_white", font={"size": 12},
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                snapshot_section = [
                    html.H3("Snapshot Timeline", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                ]
            else:
                snapshot_section = [
                    html.H3("Snapshot Timeline", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
                    html.Div("No data", style={"color": "#adb5bd", "fontSize": "14px"}),
                ]
    else:
        # No snapshots at all (VAL-TABLE-017)
        snapshot_section = [
            html.H3("Snapshot Timeline", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div("No data", style={"color": "#adb5bd", "fontSize": "14px"}),
        ]

    # Back button label depends on source page (VAL-CROSS-008)
    page_labels = {
        "leaderboard": "← Back to Leaderboard",
        "timeline": "← Back to Timeline",
        "heatmap": "← Back to Heatmap",
        "retention": "← Back to Retention Curves",
        "params": "← Back to Parameter Sweep",
        "snapshots": "← Back to Snapshot Viewer",
        "analysis": "← Back to Analysis",
        "compare": "← Back to Phase Compare",
    }
    back_label = page_labels.get(source_page, "← Back to Leaderboard")

    # Assemble detail view
    content = [
        # Header
        html.Div(
            style={"padding": "16px 24px", "borderBottom": "1px solid #dee2e6", "display": "flex", "alignItems": "center", "gap": "16px", "position": "sticky", "top": 0, "backgroundColor": "white", "zIndex": 10},
            children=[
                html.Button(back_label, id="back-button", n_clicks=0,
                    style={"padding": "8px 16px", "border": "1px solid #dee2e6", "borderRadius": "6px", "backgroundColor": "#f8f9fa", "cursor": "pointer", "fontSize": "13px", "color": "#495057"}),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "12px"},
                    children=[
                        html.H2(exp.id, style={"margin": 0, "fontSize": "20px", "color": "#212529"}),
                        "⭐" if is_best else "",
                        _make_status_badge(display_status),
                    ],
                ),
                html.Span(f"{exp.era} · Phase {exp.phase}", style={"marginLeft": "auto", "fontSize": "13px", "color": "#6c757d"}),
            ],
        ),
    ] + error_section + metrics_html + later_html + hypothesis_section + params_section + cv_section + snapshot_section

    return content


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    [
        Output("leaderboard-grid", "rowData"),
        Output("experiment-count", "children"),
        Output("sort-indicator", "children"),
    ],
    [
        Input("era-dropdown", "value"),
        Input("status-filter", "value"),
        Input("search-input", "value"),
        Input("sort-state", "data"),
        Input("selected-phase", "data"),
    ],
)
def update_leaderboard(
    era: str,
    statuses: list[str] | None,
    search: str | None,
    sort_state: dict | None,
    selected_phase: int | None,
) -> tuple[list[dict], list, str]:
    """Update leaderboard table based on filters, search, sort, and phase.

    VAL-CROSS-001: era filter propagates atomically to leaderboard.
    VAL-CROSS-002: phase click filters leaderboard to that phase only.
    """
    # Apply filters
    filtered = _filter_dataframe(_df_all, era, statuses or [], search or "")

    # Apply phase filter (VAL-CROSS-002)
    if selected_phase is not None:
        filtered = filtered[filtered["phase"] == selected_phase]

    # Apply sort from state
    sort_col = "overall_score"
    ascending = False
    if sort_state:
        sort_col = sort_state.get("column", "overall_score")
        ascending = sort_state.get("ascending", False)

    filtered = _sort_dataframe(filtered, sort_col, ascending)

    # Build row data
    row_data = _build_row_data(filtered)

    # Count
    total_count = len(row_data)
    era_label = era if era != "All" else "both eras"
    if selected_phase is not None:
        era_label += f" · Phase {selected_phase}"
    count_children = [
        html.Span(f"Showing {total_count} experiments ({era_label})", style={"fontWeight": "600", "color": "#495057"}),
    ]
    if _best_exp_id and _best_exp_id in filtered["id"].values:
        count_children.append(html.Span(f" · Best: {_best_exp_id}", style={"color": "#d4a017"}))

    # Sort indicator
    arrow = "↑" if ascending else "↓"
    col_label = sort_col.replace("_", " ").title()
    sort_text = f"Sorted by {col_label} {arrow}"

    return row_data, count_children, sort_text


@callback(
    Output("selected-experiment", "data", allow_duplicate=True),
    Input("leaderboard-grid", "cellClicked"),
    State("active-page", "data"),
    prevent_initial_call=True,
)
def on_cell_click(cellClicked: dict | None, active_page: str) -> str | None:
    """Open detail view when a cell is clicked."""
    if not cellClicked or not cellClicked.get("data"):
        return dash.no_update
    # Track source page for round-trip navigation
    return cellClicked["data"].get("id")


@callback(
    Output("selected-experiment", "data", allow_duplicate=True),
    Input("leaderboard-grid", "selectedRows"),
    prevent_initial_call=True,
)
def on_row_click(selectedRows: list[dict] | None) -> str | None:
    """Open detail view when a row is selected via keyboard."""
    if not selectedRows:
        return dash.no_update
    return selectedRows[0].get("id")


@callback(
    [
        Output("selected-experiment", "data", allow_duplicate=True),
        Output("detail-source-page", "data", allow_duplicate=True),
    ],
    Input("back-button", "n_clicks"),
    State("detail-source-page", "data"),
    prevent_initial_call=True,
)
def on_back_button(n_clicks: int, source_page: str) -> tuple:
    """Close detail view on back button click."""
    # Keep source_page set so we can navigate back to it
    return None, source_page


@callback(
    [
        Output("detail-view", "style"),
        Output("detail-view", "children"),
        Output("leaderboard-view", "style", allow_duplicate=True),
        Output("active-page", "data", allow_duplicate=True),
    ],
    Input("selected-experiment", "data"),
    State("detail-source-page", "data"),
    prevent_initial_call="initial_duplicate",
)
def toggle_detail_view(exp_id: str | None, source_page: str) -> tuple:
    """Toggle between leaderboard and detail view.

    When closing (exp_id is None), navigate to the source page
    to support round-trip navigation (VAL-CROSS-008).
    """
    if exp_id is None:
        # Navigate back to source page
        return (
            {"display": "none", "position": "absolute", "top": 0, "left": 0, "right": 0, "bottom": 0, "backgroundColor": "white", "zIndex": 100, "overflow": "auto"},
            [],
            {"display": "none"},
            source_page or "leaderboard",
        )
    # Show detail view, hide current page
    return (
        {"display": "block", "position": "absolute", "top": 0, "left": 0, "right": 0, "bottom": 0, "backgroundColor": "white", "zIndex": 100, "overflow": "auto"},
        _build_detail_view(exp_id, source_page or "leaderboard"),
        {"display": "none"},
        dash.no_update,
    )


@callback(
    Output("url", "search", allow_duplicate=True),
    Input("selected-experiment", "data"),
    State("era-dropdown", "value"),
    prevent_initial_call=True,
)
def update_url_state(exp_id: str | None, era: str) -> str:
    """Update URL when selected experiment changes."""
    if exp_id:
        params = []
        if era and era != "All":
            params.append(f"era={era}")
        params.append(f"experiment={exp_id}")
        return "?" + "&".join(params)
    # No experiment selected — show era in URL if not "All"
    if era and era != "All":
        return f"?era={era}"
    return ""


@callback(
    Output("url", "search", allow_duplicate=True),
    Input("era-dropdown", "value"),
    State("selected-experiment", "data"),
    prevent_initial_call=True,
)
def update_url_era(era: str | None, exp_id: str | None) -> str:
    """Update URL when era changes (VAL-CROSS-001, VAL-TABLE-021)."""
    params = []
    if era and era != "All":
        params.append(f"era={era}")
    if exp_id:
        params.append(f"experiment={exp_id}")
    if params:
        return "?" + "&".join(params)
    return ""


@callback(
    [
        Output("selected-experiment", "data", allow_duplicate=True),
        Output("era-dropdown", "value", allow_duplicate=True),
    ],
    Input("url", "search"),
    prevent_initial_call="initial_duplicate",
)
def restore_from_url(search: str) -> tuple:
    """Restore experiment and era from URL on page load / refresh.

    dcc.Location(refresh=False) strips query params on load, but our
    clientside callback below re-writes them from window.location.
    This server callback restores state whenever the URL search changes.
    """
    if not search:
        return dash.no_update, dash.no_update

    try:
        params = dict(p.split("=", 1) for p in search.lstrip("?").split("&") if "=" in p)
        exp_id = params.get("experiment")
        era = params.get("era")
        restored_exp = dash.no_update
        restored_era = dash.no_update
        if exp_id and any(e.id == exp_id for e in _experiments):
            restored_exp = exp_id
        if era and era in ("memories_500", "LongMemEval"):
            restored_era = era
        return restored_exp, restored_era
    except (ValueError, AttributeError):
        return dash.no_update, dash.no_update


# Clientside callback: preserve URL params on page load
# dcc.Location strips query params; this re-writes them from window.location
app.clientside_callback(
    """
    function(n_clicks) {
        var loc = window.location;
        if (loc.search && !document.querySelector('[id="url"]')) {
            // Use Dash's internal location setter
        }
        // Re-set the search via history.replaceState to preserve it
        if (loc.search) {
            history.replaceState(null, '', loc.pathname + loc.search + loc.hash);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("url", "search", allow_duplicate=True),
    Input("url", "search"),
    prevent_initial_call="initial_duplicate",
)


# ---------------------------------------------------------------------------
# Era change: clear dependent state (VAL-CROSS-004)
# ---------------------------------------------------------------------------

@callback(
    [
        Output("retention-selected", "data", allow_duplicate=True),
        Output("retention-dropdown", "value", allow_duplicate=True),
        Output("param-exp-detail-selector", "value", allow_duplicate=True),
        Output("snapshot-selected-exp", "data", allow_duplicate=True),
        Output("snapshot-exp-selector", "value", allow_duplicate=True),
        Output("cv-exp-selector", "value", allow_duplicate=True),
        Output("selected-phase", "data", allow_duplicate=True),
    ],
    Input("era-dropdown", "value"),
    prevent_initial_call=True,
)
def clear_dependent_state_on_era_change(era: str) -> tuple:
    """Clear view-specific selections when era changes.

    This ensures clean state for:
    - Retention curve selection (experiments may not exist in new era)
    - Parameter sweep experiment selector
    - Snapshot viewer experiment selector
    - CV experiment selector
    - Phase filter
    (VAL-CROSS-004: no state leaks between era switches)
    """
    return (
        [],       # retention-selected
        [],       # retention-dropdown
        None,     # param-exp-detail-selector
        None,     # snapshot-selected-exp
        None,     # snapshot-exp-selector
        None,     # cv-exp-selector
        None,     # selected-phase
    )


# ---------------------------------------------------------------------------
# Navigation callbacks
# ---------------------------------------------------------------------------

# Tab button style configs
_TAB_ACTIVE = {
    "padding": "12px 20px",
    "border": "none",
    "borderBottom": "2px solid #1565C0",
    "backgroundColor": "transparent",
    "cursor": "pointer",
    "fontSize": "14px",
    "fontWeight": "600",
    "color": "#1565C0",
    "marginRight": "4px",
}
_TAB_INACTIVE = {
    "padding": "12px 20px",
    "border": "none",
    "borderBottom": "2px solid transparent",
    "backgroundColor": "transparent",
    "cursor": "pointer",
    "fontSize": "14px",
    "fontWeight": "500",
    "color": "#6c757d",
    "marginRight": "4px",
}


@callback(
    [
        Output("leaderboard-view", "style", allow_duplicate=True),
        Output("timeline-view", "style", allow_duplicate=True),
        Output("heatmap-view", "style", allow_duplicate=True),
        Output("retention-view", "style", allow_duplicate=True),
        Output("params-view", "style", allow_duplicate=True),
        Output("snapshots-view", "style", allow_duplicate=True),
        Output("analysis-view", "style", allow_duplicate=True),
        Output("compare-view", "style", allow_duplicate=True),
        Output("tab-leaderboard", "style"),
        Output("tab-timeline", "style"),
        Output("tab-heatmap", "style"),
        Output("tab-retention", "style"),
        Output("tab-params", "style"),
        Output("tab-snapshots", "style"),
        Output("tab-analysis", "style"),
        Output("tab-compare", "style"),
    ],
    Input("active-page", "data"),
    State("selected-experiment", "data"),
    prevent_initial_call=True,
)
def update_page_display(
    active_page: str,
    selected_exp: str | None,
) -> tuple:
    """Update page display when active-page changes (from tab clicks or detail view return).

    When detail view is open (selected_exp is not None), don't change page views.
    When detail view is closed, show the correct page.

    VAL-CROSS-008: supports round-trip navigation from parameter sweep → detail → back.
    """
    # If detail view is open, don't interfere with page views
    if selected_exp is not None:
        return tuple(dash.no_update for _ in range(15))

    page = active_page or "leaderboard"

    all_views = ["leaderboard", "timeline", "heatmap", "retention", "params", "snapshots", "analysis", "compare"]
    views = {v: {"display": "none"} for v in all_views}
    views[page] = {"display": "block", "padding": "0"}

    tab_to_page = {
        "tab-leaderboard": "leaderboard",
        "tab-timeline": "timeline",
        "tab-heatmap": "heatmap",
        "tab-retention": "retention",
        "tab-params": "params",
        "tab-snapshots": "snapshots",
        "tab-analysis": "analysis",
        "tab-compare": "compare",
    }
    page_to_tab = {v: k for k, v in tab_to_page.items()}

    all_tabs = ["tab-leaderboard", "tab-timeline", "tab-heatmap", "tab-retention", "tab-params", "tab-snapshots", "tab-analysis", "tab-compare"]
    tab_styles = {t: _TAB_INACTIVE.copy() for t in all_tabs}
    active_tab = page_to_tab.get(page)
    if active_tab:
        tab_styles[active_tab] = _TAB_ACTIVE.copy()

    return (
        views["leaderboard"],
        views["timeline"],
        views["heatmap"],
        views["retention"],
        views["params"],
        views["snapshots"],
        views["analysis"],
        views["compare"],
        tab_styles["tab-leaderboard"],
        tab_styles["tab-timeline"],
        tab_styles["tab-heatmap"],
        tab_styles["tab-retention"],
        tab_styles["tab-params"],
        tab_styles["tab-snapshots"],
        tab_styles["tab-analysis"],
        tab_styles["tab-compare"],
    )


@callback(
    [
        Output("active-page", "data", allow_duplicate=True),
    ],
    [
        Input("tab-leaderboard", "n_clicks"),
        Input("tab-timeline", "n_clicks"),
        Input("tab-heatmap", "n_clicks"),
        Input("tab-retention", "n_clicks"),
        Input("tab-params", "n_clicks"),
        Input("tab-snapshots", "n_clicks"),
        Input("tab-analysis", "n_clicks"),
        Input("tab-compare", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def on_tab_click(n_lb, n_tl, n_hm, n_rt, n_pm, n_sn, n_an, n_cp) -> tuple[str]:
    """Record which tab was clicked — active-page triggers update_page_display."""
    triggered = callback_ctx.triggered_id

    page_map = {
        "tab-leaderboard": "leaderboard",
        "tab-timeline": "timeline",
        "tab-heatmap": "heatmap",
        "tab-retention": "retention",
        "tab-params": "params",
        "tab-snapshots": "snapshots",
        "tab-analysis": "analysis",
        "tab-compare": "compare",
    }
    return (page_map.get(triggered, "leaderboard"),)


# ---------------------------------------------------------------------------
# Phase timeline callback
# ---------------------------------------------------------------------------

@callback(
    [
        Output("phase-timeline-graph", "figure"),
        Output("phase-clear-btn-container", "style"),
    ],
    [
        Input("era-dropdown", "value"),
        Input("selected-phase", "data"),
    ],
)
def update_phase_timeline(era: str, selected_phase: int | None) -> tuple:
    """Update phase timeline chart on era or phase change."""
    fig = charts.build_phase_timeline(_experiments, era, selected_phase)
    clear_style = {"display": "block"} if selected_phase is not None else {"display": "none"}
    return fig, clear_style


@callback(
    Output("selected-phase", "data", allow_duplicate=True),
    Input("phase-timeline-graph", "clickData"),
    State("selected-phase", "data"),
    prevent_initial_call=True,
)
def on_timeline_click(clickData, current_phase) -> int | None:
    """Handle phase timeline bar click — toggle phase filter.

    Clicking same phase again removes the filter.
    """
    if clickData is None:
        return dash.no_update

    try:
        # Extract phase number from customdata
        points = clickData.get("points", [])
        if not points:
            return dash.no_update
        customdata = points[0].get("customdata")
        if customdata is None:
            return dash.no_update

        clicked_phase = int(customdata)

        # Toggle: clicking same phase clears the filter
        if current_phase == clicked_phase:
            return None
        return clicked_phase
    except (KeyError, ValueError, TypeError):
        return dash.no_update


@callback(
    Output("selected-phase", "data", allow_duplicate=True),
    Input("phase-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_phase_filter(n_clicks: int) -> None:
    """Clear phase filter."""
    return None


# ---------------------------------------------------------------------------
# Metric progression callbacks
# ---------------------------------------------------------------------------

@callback(
    [
        Output("metric-overall-graph", "figure"),
        Output("metric-retrieval-graph", "figure"),
        Output("metric-plausibility-graph", "figure"),
    ],
    [
        Input("era-dropdown", "value"),
        Input("selected-phase", "data"),
        Input("active-page", "data"),
    ],
)
def update_metric_progression(era: str, selected_phase: int | None, active_page: str) -> tuple:
    """Update metric progression charts on era, phase, or page change."""
    # Only build charts when timeline page is active to save computation
    if active_page != "timeline":
        return dash.no_update, dash.no_update, dash.no_update

    figs = charts.build_metric_progression(_experiments, era, selected_phase)
    if len(figs) == 3:
        return figs[0], figs[1], figs[2]
    return dash.no_update, dash.no_update, dash.no_update


# ---------------------------------------------------------------------------
# Threshold heatmap callbacks
# ---------------------------------------------------------------------------

@callback(
    [
        Output("heatmap-recall-graph", "figure"),
        Output("heatmap-precision-graph", "figure"),
    ],
    [
        Input("era-dropdown", "value"),
        Input("selected-phase", "data"),
        Input("active-page", "data"),
    ],
)
def update_heatmaps(era: str, selected_phase: int | None, active_page: str) -> tuple:
    """Update threshold heatmaps on era, phase, or page change."""
    if active_page != "heatmap":
        return dash.no_update, dash.no_update

    figs = charts.build_threshold_heatmap(_experiments, era, selected_phase)
    if len(figs) == 2:
        return figs[0], figs[1]
    return dash.no_update, dash.no_update


# ---------------------------------------------------------------------------
# Retention curve callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("retention-dropdown", "options"),
    [
        Input("era-dropdown", "value"),
        Input("active-page", "data"),
    ],
)
def update_retention_options(era: str, active_page: str) -> list:
    """Update retention experiment dropdown options based on era."""
    if active_page != "retention":
        return dash.no_update

    available = charts.get_retention_available_experiments(_experiments, era)
    return [{"label": eid, "value": eid} for eid in available]


@callback(
    [
        Output("retention-selected", "data"),
        Output("retention-warning", "children"),
        Output("retention-count", "children"),
    ],
    [
        Input("retention-dropdown", "value"),
        Input("retention-clear-btn", "n_clicks"),
    ],
    State("retention-selected", "data"),
)
def update_retention_selection(
    dropdown_values: list[str] | None,
    clear_clicks: int,
    current_selection: list[str] | None,
) -> tuple:
    """Handle retention experiment selection with max 5 enforcement."""
    triggered = callback_ctx.triggered_id

    if triggered == "retention-clear-btn":
        return [], [], ["Select experiments to compare"]

    if dropdown_values is None:
        dropdown_values = []

    # Enforce max 5
    if len(dropdown_values) > 5:
        dropdown_values = dropdown_values[:5]

    # Check for warnings
    warnings = charts.check_retention_warnings(dropdown_values, _experiments)

    warning_children = []
    for w in warnings:
        warning_children.append(
            html.Div(
                f"⚠ {w}",
                style={"padding": "6px 12px", "backgroundColor": "#fff3cd", "borderRadius": "4px", "border": "1px solid #ffeaa7", "fontSize": "12px", "color": "#856404", "marginBottom": "4px"},
            )
        )

    count_text = f"{len(dropdown_values)} of 5 experiments selected"

    return dropdown_values, warning_children, count_text


@callback(
    Output("retention-graph", "figure"),
    [
        Input("retention-selected", "data"),
        Input("active-page", "data"),
    ],
)
def update_retention_chart(selected_ids: list[str] | None, active_page: str) -> go.Figure:
    """Update retention curve overlay chart."""
    if active_page != "retention":
        return dash.no_update

    fig = charts.build_retention_overlay(_experiments, selected_ids or [])
    return fig


# ---------------------------------------------------------------------------
# Leaderboard phase filter integration
# ---------------------------------------------------------------------------

@callback(
    [
        Output("status-filter", "value", allow_duplicate=True),
        Output("retention-selected", "data", allow_duplicate=True),
    ],
    Input("selected-phase", "data"),
    State("retention-selected", "data"),
    prevent_initial_call=True,
)
def phase_change_dependent_updates(selected_phase: int | None, current_retention: list[str] | None) -> tuple:
    """When phase is selected, clear retention selections for experiments not in that phase.

    Also keeps leaderboard phase filter working (status filter unchanged).
    """
    updated_retention = dash.no_update
    if selected_phase is not None and current_retention:
        filtered = []
        for exp_id in current_retention:
            exp = next((e for e in _experiments if e.id == exp_id), None)
            if exp and exp.phase == selected_phase:
                filtered.append(exp_id)
        if len(filtered) != len(current_retention):
            updated_retention = filtered

    return dash.no_update, updated_retention


# ---------------------------------------------------------------------------
# Parameter Sweep Callbacks
# ---------------------------------------------------------------------------

@callback(
    [
        Output("param-dimension-selector", "options"),
        Output("param-exp-detail-selector", "options"),
    ],
    [
        Input("era-dropdown", "value"),
        Input("active-page", "data"),
    ],
)
def update_param_sweep_options(era: str, active_page: str) -> tuple:
    """Update parameter sweep dimension options and experiment selector."""
    if active_page != "params":
        return dash.no_update, dash.no_update

    # Get available params for the toggle selector
    available_params = charts.get_param_sweep_available_params(_experiments, era)
    param_options = [{"label": p, "value": p} for p in available_params]

    # Get experiments with params and scores for detail selector
    filtered = _experiments if era == "All" else [e for e in _experiments if e.era == era]
    scored_with_params = [e for e in filtered if e.params and e.overall_score is not None]
    scored_with_params.sort(key=lambda e: e.overall_score, reverse=True)
    exp_options = [{"label": f"{e.id} (overall: {_format_score(e.overall_score)})", "value": e.id}
                   for e in scored_with_params[:100]]

    return param_options, exp_options


@callback(
    Output("param-sweep-graph", "figure"),
    [
        Input("era-dropdown", "value"),
        Input("param-dimension-selector", "value"),
        Input("active-page", "data"),
    ],
)
def update_param_sweep(era: str, enabled_params: list[str] | None, active_page: str) -> go.Figure:
    """Update parameter sweep chart."""
    if active_page != "params":
        return dash.no_update

    return charts.build_parameter_sweep(_experiments, era, enabled_params)


@callback(
    [
        Output("selected-experiment", "data", allow_duplicate=True),
        Output("detail-source-page", "data", allow_duplicate=True),
    ],
    Input("param-exp-detail-selector", "value"),
    State("active-page", "data"),
    prevent_initial_call=True,
)
def on_param_exp_detail_select(exp_id: str | None, active_page: str) -> tuple:
    """Navigate to experiment detail from parameter sweep (VAL-CROSS-008)."""
    if exp_id is None:
        return dash.no_update, dash.no_update
    # Track source page so detail view can show "Back to <Source>" button
    return exp_id, active_page or "leaderboard"


# ---------------------------------------------------------------------------
# Snapshot Viewer Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("snapshot-exp-selector", "options"),
    [
        Input("era-dropdown", "value"),
        Input("active-page", "data"),
    ],
)
def update_snapshot_exp_options(era: str, active_page: str) -> list:
    """Update snapshot experiment dropdown options."""
    if active_page != "snapshots":
        return dash.no_update

    filtered = _experiments if era == "All" else [e for e in _experiments if e.era == era]
    with_snapshots = [e for e in filtered if e.snapshots]
    return [{"label": f"{e.id} ({len(e.snapshots)} ticks)", "value": e.id} for e in with_snapshots]


@callback(
    Output("snapshot-current-tick", "data"),
    Input("snapshot-exp-selector", "value"),
    prevent_initial_call=True,
)
def on_snapshot_exp_change(exp_id: str | None) -> int:
    """Reset tick to 0 when experiment changes."""
    return 0


@callback(
    [
        Output("snapshot-viewer-graph", "figure"),
        Output("snapshot-tick-label", "children"),
        Output("snapshot-tick-cards", "children"),
    ],
    [
        Input("snapshot-selected-exp", "data"),
        Input("snapshot-current-tick", "data"),
        Input("active-page", "data"),
    ],
)
def update_snapshot_viewer(exp_id: str | None, current_tick: int, active_page: str) -> tuple:
    """Update snapshot viewer chart and tick info."""
    if active_page != "snapshots":
        return dash.no_update, dash.no_update, dash.no_update

    if exp_id is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Select an experiment to view snapshots",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig, "Tick: —", []

    fig = charts.build_snapshot_viewer(_experiments, exp_id, current_tick)
    tick_label = f"Tick: {current_tick}" if current_tick is not None else "Tick: —"

    # Build tick metric cards
    tick_data = charts.get_snapshot_tick_data(_experiments, exp_id, current_tick or 0)
    if tick_data:
        cards = []
        metric_labels = [
            ("overall_score", "Overall Score", "#1565C0"),
            ("recall_mean", "Recall Mean", "#2E7D32"),
            ("precision_mean", "Precision Mean", "#E65100"),
            ("plausibility_score", "Plausibility", "#6A1B9A"),
            ("mrr_mean", "MRR Mean", "#C62828"),
        ]
        for key, label, color in metric_labels:
            val = tick_data.get(key)
            cards.append(html.Div(
                style={"padding": "8px 12px", "backgroundColor": f"{color}10", "borderRadius": "6px", "border": f"1px solid {color}40", "minWidth": "140px"},
                children=[
                    html.Div(label, style={"fontSize": "10px", "color": color, "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                    html.Div(_format_score(val), style={"fontSize": "16px", "fontWeight": "700", "color": "#212529", "marginTop": "2px"}),
                ],
            ))
        tick_cards = html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=cards)
    else:
        tick_cards = html.Div("No data at this tick", style={"color": "#adb5bd", "fontSize": "13px"})

    return fig, tick_label, tick_cards


@callback(
    Output("snapshot-selected-exp", "data", allow_duplicate=True),
    Input("snapshot-exp-selector", "value"),
    prevent_initial_call=True,
)
def on_snapshot_exp_select(exp_id: str | None) -> str | None:
    """Set selected experiment for snapshot viewer."""
    return exp_id


@callback(
    Output("snapshot-current-tick", "data", allow_duplicate=True),
    [
        Input("snapshot-first", "n_clicks"),
        Input("snapshot-prev", "n_clicks"),
        Input("snapshot-next", "n_clicks"),
        Input("snapshot-last", "n_clicks"),
    ],
    State("snapshot-current-tick", "data"),
    State("snapshot-selected-exp", "data"),
    prevent_initial_call=True,
)
def on_snapshot_step(n_first, n_prev, n_next, n_last, current_tick: int, exp_id: str | None) -> int:
    """Step through ticks."""
    triggered = callback_ctx.triggered_id

    if exp_id is None:
        return dash.no_update

    # Get max tick for this experiment
    exp = next((e for e in _experiments if e.id == exp_id), None)
    max_tick = 200
    if exp and exp.snapshots:
        ticks = [s.get("tick", 0) for s in exp.snapshots]
        max_tick = max(ticks) if ticks else 200

    TICK_STEP = 20

    if triggered == "snapshot-first":
        return 0
    elif triggered == "snapshot-prev":
        return max(0, (current_tick or 0) - TICK_STEP)
    elif triggered == "snapshot-next":
        return min(max_tick, (current_tick or 0) + TICK_STEP)
    elif triggered == "snapshot-last":
        return max_tick

    return dash.no_update


@callback(
    Output("snapshot-interval", "disabled"),
    Input("snapshot-animate-btn", "n_clicks"),
    State("snapshot-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_snapshot_animation(n_clicks: int, currently_disabled: bool) -> bool:
    """Toggle animation play/pause."""
    return not currently_disabled


@callback(
    Output("snapshot-current-tick", "data", allow_duplicate=True),
    Input("snapshot-interval", "n_intervals"),
    State("snapshot-selected-exp", "data"),
    State("snapshot-current-tick", "data"),
    prevent_initial_call=True,
)
def animate_snapshot(n_intervals: int, exp_id: str | None, current_tick: int) -> int | None:
    """Advance tick during animation."""
    if exp_id is None:
        return dash.no_update

    exp = next((e for e in _experiments if e.id == exp_id), None)
    max_tick = 200
    if exp and exp.snapshots:
        ticks = [s.get("tick", 0) for s in exp.snapshots]
        max_tick = max(ticks) if ticks else 200

    next_tick = (current_tick or 0) + 20
    if next_tick > max_tick:
        return 0  # Loop back
    return next_tick


# ---------------------------------------------------------------------------
# Forgetting Depth & CV Analysis Callbacks
# ---------------------------------------------------------------------------

@callback(
    [
        Output("forgetting-depth-graph", "figure"),
        Output("strict-score-graph", "figure"),
    ],
    [
        Input("era-dropdown", "value"),
        Input("active-page", "data"),
    ],
)
def update_forgetting_depth_charts(era: str, active_page: str) -> tuple:
    """Update forgetting depth and strict score charts."""
    if active_page != "analysis":
        return dash.no_update, dash.no_update

    return (
        charts.build_forgetting_depth_chart(_experiments, era),
        charts.build_strict_score_chart(_experiments, era),
    )


@callback(
    Output("cv-exp-selector", "options"),
    [
        Input("era-dropdown", "value"),
        Input("active-page", "data"),
    ],
)
def update_cv_exp_options(era: str, active_page: str) -> list:
    """Update CV experiment selector options."""
    if active_page != "analysis":
        return dash.no_update

    available = charts.get_cv_available_experiments(_experiments, _cv_data, era)
    return [{"label": eid, "value": eid} for eid in available]


@callback(
    [
        Output("cv-fold-scores-graph", "figure"),
        Output("cv-fold-deltas-graph", "figure"),
    ],
    [
        Input("cv-exp-selector", "value"),
        Input("active-page", "data"),
    ],
)
def update_cv_charts(exp_id: str | None, active_page: str) -> tuple:
    """Update CV fold scores and deltas charts."""
    if active_page != "analysis":
        return dash.no_update, dash.no_update

    if exp_id is None:
        fig1 = go.Figure()
        fig1.add_annotation(text="Select an experiment with CV data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font={"size": 14, "color": "#9E9E9E"})
        fig1.update_layout(template="plotly_white")
        fig2 = go.Figure()
        fig2.update_layout(template="plotly_white")
        return fig1, fig2

    cv = _cv_data.get(exp_id)
    if cv is None:
        fig1 = go.Figure()
        fig1.add_annotation(text="No CV data for this experiment", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font={"size": 14, "color": "#9E9E9E"})
        fig1.update_layout(template="plotly_white")
        fig2 = go.Figure()
        fig2.update_layout(template="plotly_white")
        return fig1, fig2

    return (
        charts.build_cv_fold_scores(cv, exp_id),
        charts.build_cv_fold_deltas(cv, exp_id),
    )


# ---------------------------------------------------------------------------
# Phase Comparison Callbacks
# ---------------------------------------------------------------------------

@callback(
    [
        Output("compare-phase-a-dropdown", "options"),
        Output("compare-phase-b-dropdown", "options"),
    ],
    Input("active-page", "data"),
)
def update_compare_phase_options(active_page: str) -> tuple:
    """Update phase comparison dropdown options."""
    if active_page != "compare":
        return dash.no_update, dash.no_update

    available = charts.get_available_phases(_experiments)
    return available, available


@callback(
    Output("phase-comparison-results", "children"),
    [
        Input("compare-phase-a-dropdown", "value"),
        Input("compare-phase-b-dropdown", "value"),
    ],
)
def update_phase_comparison(phase_a: int | None, phase_b: int | None) -> list:
    """Update phase comparison results panel."""
    if phase_a is None or phase_b is None:
        return [html.Div("Select two phases to compare", style={"color": "#adb5bd", "fontSize": "14px"})]

    comparison = charts.build_phase_comparison(_experiments, phase_a, phase_b)

    if comparison["insufficient_data"]:
        return [
            html.Div(
                style={"padding": "24px", "backgroundColor": "#fff3cd", "borderRadius": "6px", "border": "1px solid #ffeaa7"},
                children=[
                    html.Div("⚠ Insufficient Data", style={"fontWeight": "700", "color": "#856404", "marginBottom": "8px"}),
                    html.Div("Both selected phases have no scored (non-validation_failed) experiments.", style={"color": "#856404", "fontSize": "14px"}),
                ],
            )
        ]

    # Phase headers
    phase_a_name = comparison["phase_a_name"]
    phase_b_name = comparison["phase_b_name"]
    count_a = comparison["count_a"]
    count_b = comparison["count_b"]
    scored_a = comparison["scored_count_a"]
    scored_b = comparison["scored_count_b"]

    children = [
        # Summary header
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "24px"},
            children=[
                html.Div(
                    style={"flex": "1", "padding": "16px", "backgroundColor": "#E3F2FD", "borderRadius": "6px", "border": "1px solid #90CAF9"},
                    children=[
                        html.Div(f"Phase {phase_a}: {phase_a_name}", style={"fontSize": "16px", "fontWeight": "700", "color": "#1565C0"}),
                        html.Div(f"{count_a} experiments ({scored_a} scored)", style={"fontSize": "13px", "color": "#6c757d", "marginTop": "4px"}),
                    ],
                ),
                html.Div(
                    style={"flex": "1", "padding": "16px", "backgroundColor": "#E8F5E9", "borderRadius": "6px", "border": "1px solid #A5D6A7"},
                    children=[
                        html.Div(f"Phase {phase_b}: {phase_b_name}", style={"fontSize": "16px", "fontWeight": "700", "color": "#2E7D32"}),
                        html.Div(f"{count_b} experiments ({scored_b} scored)", style={"fontSize": "13px", "color": "#6c757d", "marginTop": "4px"}),
                    ],
                ),
            ],
        ),
    ]

    # Build comparison table
    table_header = html.Tr(
        children=[
            html.Th("Metric", style={"textAlign": "left", "padding": "8px 12px", "borderBottom": "2px solid #dee2e6", "fontSize": "12px", "color": "#6c757d", "textTransform": "uppercase"}),
            html.Th(f"Phase {phase_a}", style={"textAlign": "right", "padding": "8px 12px", "borderBottom": "2px solid #dee2e6", "fontSize": "12px", "color": "#1565C0", "textTransform": "uppercase"}),
            html.Th(f"Phase {phase_b}", style={"textAlign": "right", "padding": "8px 12px", "borderBottom": "2px solid #dee2e6", "fontSize": "12px", "color": "#2E7D32", "textTransform": "uppercase"}),
            html.Th("Improvement", style={"textAlign": "right", "padding": "8px 12px", "borderBottom": "2px solid #dee2e6", "fontSize": "12px", "color": "#6c757d", "textTransform": "uppercase"}),
        ],
    )

    table_rows = []
    for metric in comparison["metrics"]:
        label = metric["label"]
        a = metric["phase_a"]
        b = metric["phase_b"]
        imp = metric["improvement_pct"]

        mean_a = _format_score(a["mean"])
        mean_b = _format_score(b["mean"])
        best_a = _format_score(a["best"])
        best_b = _format_score(b["best"])

        if scored_a == 0:
            mean_a_str = "insufficient data"
        else:
            mean_a_str = f"{mean_a} (best: {best_a})"

        if scored_b == 0:
            mean_b_str = "insufficient data"
        else:
            mean_b_str = f"{mean_b} (best: {best_b})"

        if imp is not None:
            imp_str = f"{imp:+.2f}%"
            imp_color = "#2E7D32" if imp > 0 else "#D32F2F" if imp < 0 else "#6c757d"
        else:
            imp_str = "N/A"
            imp_color = "#adb5bd"

        table_rows.append(html.Tr(
            children=[
                html.Td(label, style={"padding": "8px 12px", "borderBottom": "1px solid #e9ecef", "fontWeight": "500"}),
                html.Td(mean_a_str, style={"padding": "8px 12px", "borderBottom": "1px solid #e9ecef", "textAlign": "right", "fontFamily": "monospace", "fontSize": "13px"}),
                html.Td(mean_b_str, style={"padding": "8px 12px", "borderBottom": "1px solid #e9ecef", "textAlign": "right", "fontFamily": "monospace", "fontSize": "13px"}),
                html.Td(imp_str, style={"padding": "8px 12px", "borderBottom": "1px solid #e9ecef", "textAlign": "right", "fontWeight": "600", "color": imp_color}),
            ],
        ))

    children.append(html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": "white", "borderRadius": "6px", "overflow": "hidden"},
        children=[html.Tbody(children=[table_header] + table_rows)],
    ))

    return children


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
