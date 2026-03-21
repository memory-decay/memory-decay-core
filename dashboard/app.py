"""Memory Decay Experiment Dashboard.

Dash/Plotly application for visualizing 440+ memory decay experiments
across two research eras (memories_500 and LongMemEval).

Features:
- Sidebar: era selector, status multi-select filter, text search
- Leaderboard table: sortable columns, status badges, best experiment highlight
- Detail view: same-page overlay with metrics, params, hypothesis, CV, snapshots
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

from dashboard.data_loader import Experiment, load_all_experiments

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


def _build_detail_view(exp_id: str) -> list:
    """Build detail view content for an experiment."""
    exp = next((e for e in _experiments if e.id == exp_id), None)
    if exp is None:
        return [html.P(f"Experiment {exp_id} not found")]

    cv = _cv_data.get(exp_id)
    history_status_val = _history_status.get(exp_id)
    display_status = history_status_val or exp.status
    is_best = _best_exp_id == exp_id

    # Key metrics
    metrics = [
        ("Overall Score", exp.overall_score),
        ("Retrieval Score", exp.retrieval_score),
        ("Plausibility Score", exp.plausibility_score),
        ("Recall Mean", exp.recall_mean),
        ("Precision Mean", exp.precision_mean),
        ("MRR Mean", exp.mrr_mean),
        ("Correlation Score", exp.corr_score),
        ("Retention AUC", exp.retention_auc),
        ("Selectivity Score", exp.selectivity_score),
    ]

    metrics_html = [
        html.H3("Key Metrics", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))", "gap": "12px"},
            children=[
                html.Div(
                    style={"padding": "12px", "backgroundColor": "#f8f9fa", "borderRadius": "6px", "border": "1px solid #e9ecef"},
                    children=[
                        html.Div(name, style={"fontSize": "11px", "color": "#6c757d", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                        html.Div(_format_score(val), style={"fontSize": "20px", "fontWeight": "700", "color": "#212529", "marginTop": "4px"}),
                    ],
                )
                for name, val in metrics
            ],
        ),
    ]

    # Later-era metrics
    later_metrics = [
        ("Strict Score", exp.strict_score),
        ("Forgetting Depth", exp.forgetting_depth),
        ("Forgetting Score", exp.forgetting_score),
        ("Eval V2 Score", exp.eval_v2_score),
    ]
    has_later = any(v is not None for _, v in later_metrics)
    later_html = []
    if has_later:
        later_html = [
            html.H3("Strict Validation", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))", "gap": "12px"},
                children=[
                    html.Div(
                        style={"padding": "12px", "backgroundColor": "#fff8e1", "borderRadius": "6px", "border": "1px solid #ffe082"},
                        children=[
                            html.Div(name, style={"fontSize": "11px", "color": "#856404", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                            html.Div(_format_score(val), style={"fontSize": "20px", "fontWeight": "700", "color": "#212529", "marginTop": "4px"}),
                        ],
                    )
                    for name, val in later_metrics
                ],
            ),
        ]

    # Error section
    error_section = []
    if exp.error:
        error_section = [
            html.Div(
                style={"padding": "16px", "backgroundColor": "#f8d7da", "borderRadius": "6px", "border": "1px solid #f5c6cb", "marginBottom": "16px"},
                children=[
                    html.Div("⚠ Validation Error", style={"fontWeight": "700", "color": "#721c24", "marginBottom": "8px"}),
                    html.Div(exp.error, style={"color": "#721c24", "fontSize": "14px", "fontFamily": "monospace"}),
                ],
            ),
        ]

    # Hypothesis section
    hypothesis_section = [
        html.H3("Hypothesis", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
        html.Div(
            exp.hypothesis if exp.hypothesis else "Hypothesis not available",
            style={"padding": "16px", "backgroundColor": "#f8f9fa", "borderRadius": "6px", "border": "1px solid #e9ecef", "fontSize": "14px", "lineHeight": "1.6", "whiteSpace": "pre-wrap", "color": "#495057" if exp.hypothesis else "#adb5bd"},
        ),
    ]

    # Parameters section
    if exp.params:
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
                    for k, v in sorted(exp.params.items())
                ],
            ),
        ]
    else:
        params_section = [
            html.H3("Parameters", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div("Parameters not available", style={"color": "#adb5bd", "fontSize": "14px"}),
        ]

    # CV section
    cv_section = []
    if cv:
        cv_mean = cv.get("mean", {})
        cv_std = cv.get("std", {})
        cv_k = cv.get("k", "?")
        cv_worst = cv.get("worst_fold", {})

        cv_metrics = []
        if isinstance(cv_mean, dict):
            cv_metrics.extend([
                ("CV Mean (overall)", cv_mean.get("overall_score")),
                ("CV Mean (retrieval)", cv_mean.get("retrieval_score")),
                ("CV Mean (plausibility)", cv_mean.get("plausibility_score")),
            ])
        if isinstance(cv_std, dict):
            cv_metrics.extend([
                ("CV Std (overall)", cv_std.get("overall_score")),
                ("CV Std (retrieval)", cv_std.get("retrieval_score")),
            ])

        cv_section = [
            html.H3("Cross-Validation Results", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div(style={"marginBottom": "8px", "fontSize": "13px", "color": "#6c757d"}, children=[f"k = {cv_k} folds"]),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))", "gap": "12px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={"padding": "12px", "backgroundColor": "#e8f5e9", "borderRadius": "6px", "border": "1px solid #a5d6a7"},
                        children=[
                            html.Div(name, style={"fontSize": "11px", "color": "#2e7d32", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                            html.Div(_format_score(val), style={"fontSize": "18px", "fontWeight": "700", "color": "#212529", "marginTop": "4px"}),
                        ],
                    )
                    for name, val in cv_metrics
                ],
            ),
        ]

        if cv_worst and isinstance(cv_worst, dict):
            worst_overall = cv_worst.get("overall_score")
            cv_section.append(
                html.Div(
                    style={"padding": "12px", "backgroundColor": "#fff3e0", "borderRadius": "6px", "border": "1px solid #ffcc80", "marginBottom": "16px"},
                    children=[
                        html.Div("Worst Fold", style={"fontSize": "11px", "color": "#e65100", "textTransform": "uppercase"}),
                        html.Div(_format_score(worst_overall), style={"fontSize": "18px", "fontWeight": "700", "color": "#212529", "marginTop": "4px"}),
                    ],
                ),
            )

    # Snapshot mini-charts
    snapshot_section = []
    if exp.snapshots:
        ticks = [s["tick"] for s in exp.snapshots if "tick" in s]
        if ticks:
            fig = go.Figure()
            trace_configs = [
                ("overall_score", "Overall Score", "#2196F3"),
                ("retrieval_score", "Retrieval Score", "#4CAF50"),
                ("plausibility_score", "Plausibility Score", "#FF9800"),
            ]
            for key, label, color in trace_configs:
                values = [s.get(key) for s in exp.snapshots if s.get(key) is not None]
                t_vals = [s["tick"] for s in exp.snapshots if s.get(key) is not None]
                if values:
                    fig.add_trace(go.Scatter(x=t_vals, y=values, mode="lines+markers", name=label, line={"color": color, "width": 2}, marker={"size": 4}))
            fig.update_layout(
                title="Metrics over Simulation Ticks",
                xaxis_title="Tick", yaxis_title="Score", yaxis={"range": [0, 1]},
                height=280, margin={"l": 50, "r": 20, "t": 40, "b": 40},
                template="plotly_white", font={"size": 12},
            )
            snapshot_section = [
                html.H3("Snapshot Timeline", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ]
    elif exp.status not in ("validation_failed", "no_results"):
        snapshot_section = [
            html.H3("Snapshot Timeline", style={"marginTop": "24px", "marginBottom": "12px", "fontSize": "16px"}),
            html.Div("No snapshot data available", style={"color": "#adb5bd", "fontSize": "14px"}),
        ]

    # Assemble detail view
    content = [
        # Header
        html.Div(
            style={"padding": "16px 24px", "borderBottom": "1px solid #dee2e6", "display": "flex", "alignItems": "center", "gap": "16px", "position": "sticky", "top": 0, "backgroundColor": "white", "zIndex": 10},
            children=[
                html.Button("← Back to Leaderboard", id="back-button", n_clicks=0,
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
    ],
)
def update_leaderboard(
    era: str,
    statuses: list[str] | None,
    search: str | None,
    sort_state: dict | None,
) -> tuple[list[dict], list, str]:
    """Update leaderboard table based on filters, search, and sort."""
    # Apply filters
    filtered = _filter_dataframe(_df_all, era, statuses or [], search or "")

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
    prevent_initial_call=True,
)
def on_cell_click(cellClicked: dict | None) -> str | None:
    """Open detail view when a cell is clicked."""
    if not cellClicked or not cellClicked.get("data"):
        return dash.no_update
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
    Output("selected-experiment", "data", allow_duplicate=True),
    Input("back-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_back_button(n_clicks: int) -> None:
    """Close detail view on back button click."""
    return None


@callback(
    [
        Output("detail-view", "style"),
        Output("detail-view", "children"),
        Output("leaderboard-view", "style"),
    ],
    Input("selected-experiment", "data"),
)
def toggle_detail_view(exp_id: str | None) -> tuple[dict, list, dict]:
    """Toggle between leaderboard and detail view."""
    if exp_id is None:
        return (
            {"display": "none", "position": "absolute", "top": 0, "left": 0, "right": 0, "bottom": 0, "backgroundColor": "white", "zIndex": 100, "overflow": "auto"},
            [],
            {"display": "block"},
        )
    return (
        {"display": "block", "position": "absolute", "top": 0, "left": 0, "right": 0, "bottom": 0, "backgroundColor": "white", "zIndex": 100, "overflow": "auto"},
        _build_detail_view(exp_id),
        {"display": "none"},
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
    """Restore experiment and era from URL on page load / refresh."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
