"""Chart building functions for the Memory Decay Experiment Dashboard.

Provides Plotly figure builders for:
- Phase timeline (horizontal bar chart)
- Metric progression (line charts with phase shading)
- Threshold heatmap (recall/precision at thresholds 0.1-0.9)
- Retention curve overlay (multi-experiment comparison)
- Parameter sweep (parallel coordinates colored by overall_score)
- Snapshot viewer (expanded multi-metric tick view)
- Forgetting depth analysis (distribution with pass/fail)
- CV results (fold scores, fold deltas, worst-fold)
- Phase comparison (side-by-side statistics)

All charts support era filtering, phase highlighting, and follow the
conventions defined in the mission specification.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from dashboard.data_loader import Experiment, PHASE_RANGES

# ---------------------------------------------------------------------------
# Phase metadata
# ---------------------------------------------------------------------------

PHASE_NAMES: dict[int, str] = {
    0: "Project Setup",
    1: "Early Exploration",
    2: "Reinforcement Redesign",
    3: "Auto-Research First 25",
    4: "Protocol Fixes & Extended",
    5: "Scoring Overhaul",
    6: "Memory Chain",
    7: "LongMemEval Integration",
    8: "LongMemEval Auto-Research",
    9: "Batch Embedding & Strict",
}

# Phase colors for background shading (semi-transparent, distinct)
PHASE_COLORS: dict[int, str] = {
    1: "#1E88E5",  # Blue
    2: "#43A047",  # Green
    3: "#FB8C00",  # Orange
    4: "#8E24AA",  # Purple
    5: "#E53935",  # Red
    6: "#00ACC1",  # Cyan
    7: "#FFB300",  # Amber
    8: "#3949AB",  # Indigo
    9: "#6D4C41",  # Brown
}

# Phase colors with alpha for timeline bars
PHASE_BAR_COLORS: dict[int, str] = {
    1: "#90CAF9",
    2: "#A5D6A7",
    3: "#FFCC80",
    4: "#CE93D8",
    5: "#EF9A9A",
    6: "#80DEEA",
    7: "#FFE082",
    8: "#9FA8DA",
    9: "#BCAAA4",
}

# Retention curve palette (10 distinct colors)
RETENTION_COLORS = [
    "#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#C62828",
    "#00838F", "#F9A825", "#4E342E", "#283593", "#AD1457",
]

# Thresholds for heatmap
THRESHOLD_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
THRESHOLD_LABELS = [str(t) for t in THRESHOLD_VALUES]

# Retention curve ticks
RETENTION_TICKS = [40, 80, 120, 160, 200]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_date_range(phase: int) -> tuple[str, str]:
    """Return approximate (start_date, end_date) for a phase based on git history.

    These are approximate dates derived from first/last experiment commits.
    Phases 0 and 6 have no experiments.
    """
    # Approximate dates from git log analysis
    DATES = {
        0: ("—", "—"),
        1: ("Mar 18", "Mar 18"),
        2: ("Mar 18", "Mar 18"),
        3: ("Mar 18", "Mar 18"),
        4: ("Mar 18", "Mar 18"),
        5: ("Mar 18", "Mar 19"),
        6: ("—", "—"),
        7: ("Mar 20", "Mar 20"),
        8: ("Mar 20", "Mar 20"),
        9: ("Mar 21", "Mar 21"),
    }
    return DATES.get(phase, ("—", "—"))


def _get_sorted_experiments(
    experiments: list[Experiment],
    era: str = "All",
) -> list[Experiment]:
    """Get experiments sorted chronologically: old-era before LME-era, by phase then ID."""
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    # Sort: by phase first, then era (old before new within same phase is impossible
    # since they don't overlap), then by numeric ID within era
    def _sort_key(e: Experiment):
        phase = e.phase if e.phase is not None else 999
        era_order = 0 if e.era == "memories_500" else 1
        # Extract numeric ID
        if e.era == "memories_500":
            num_id = int(e.id.split("_")[1])
        else:
            num_id = int(e.id.split("_")[2])
        return (phase, era_order, num_id)

    return sorted(filtered, key=_sort_key)


def _phase_for_index(
    exp: Experiment,
    sorted_experiments: list[Experiment],
) -> tuple[int, int]:
    """Return (start_idx, end_idx) for the phase of this experiment in sorted list."""
    phase = exp.phase
    if phase is None:
        return (0, 0)

    indices = [i for i, e in enumerate(sorted_experiments) if e.phase == phase]
    if not indices:
        return (0, 0)
    return (min(indices), max(indices))


# ---------------------------------------------------------------------------
# Phase Timeline
# ---------------------------------------------------------------------------

def build_phase_timeline(
    experiments: list[Experiment],
    era: str = "All",
    selected_phase: Optional[int] = None,
) -> go.Figure:
    """Build the phase timeline horizontal bar chart.

    Shows all 9 phases with:
    - Phase name, date range, experiment count, best overall_score
    - Proportional width by experiment count
    - Clickable bars (clickData returns phase number)
    - Highlighted selected phase

    Args:
        experiments: All loaded experiments.
        era: Current era filter ("All", "memories_500", "LongMemEval").
        selected_phase: Currently selected phase (for highlighting).

    Returns:
        Plotly Figure with horizontal bar chart.
    """
    # Filter experiments by era
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    # Gather phase statistics
    phases_with_data: list[dict[str, Any]] = []
    for phase_num in range(10):
        phase_exps = [e for e in filtered if e.phase == phase_num]
        # Exclude validation_failed for best score
        scored_exps = [e for e in phase_exps
                       if e.status != "validation_failed" and e.overall_score is not None]

        best_score = max((e.overall_score for e in scored_exps), default=None)

        phases_with_data.append({
            "phase": phase_num,
            "name": PHASE_NAMES.get(phase_num, f"Phase {phase_num}"),
            "count": len(phase_exps),
            "best_score": best_score,
            "date_range": _phase_date_range(phase_num),
        })

    # Remove phases with 0 experiments (0, 6) for cleaner display,
    # but keep them as thin markers
    phases_display = [p for p in phases_with_data if p["count"] > 0]
    phases_empty = [p for p in phases_with_data if p["count"] == 0]

    # Build bar chart
    y_labels = []
    bar_values = []  # Experiment count (proportional width)
    hover_texts = []
    colors = []

    for p in phases_display:
        y_labels.append(f"Phase {p['phase']}: {p['name']}")
        bar_values.append(p["count"])
        best_str = f"{p['best_score']:.4f}" if p['best_score'] is not None else "N/A"
        hover_texts.append(
            f"Phase {p['phase']}: {p['name']}<br>"
            f"Date range: {p['date_range'][0]} — {p['date_range'][1]}<br>"
            f"Experiments: {p['count']}<br>"
            f"Best overall_score: {best_str}"
        )
        if selected_phase is not None and p["phase"] == selected_phase:
            colors.append(PHASE_COLORS.get(p["phase"], "#666"))
        else:
            colors.append(PHASE_BAR_COLORS.get(p["phase"], "#BDBDBD"))

    # Add empty phases as minimal markers
    for p in phases_empty:
        y_labels.append(f"Phase {p['phase']}: {p['name']}")
        bar_values.append(0.5)  # Minimal width
        hover_texts.append(
            f"Phase {p['phase']}: {p['name']}<br>"
            f"No experiments in this phase"
        )
        colors.append("#E0E0E0")

    fig = go.Figure()

    # Build text labels for bars
    bar_texts = []
    for p in phases_display:
        best_str = f"Best: {p['best_score']:.4f}" if p.get('best_score') is not None else ""
        bar_texts.append(f"{p['count']} exps<br>{best_str}")
    bar_texts += ["— " for _ in phases_empty]

    fig.add_trace(go.Bar(
        y=y_labels,
        x=bar_values,
        orientation="h",
        marker_color=colors,
        text=bar_texts,
        textposition="auto",
        hovertext=hover_texts,
        hoverinfo="text",
        customdata=[p["phase"] for p in phases_display] + [p["phase"] for p in phases_empty],
    ))

    fig.update_layout(
        title="Phase Timeline",
        xaxis_title="Number of Experiments",
        barmode="overlay",
        height=max(300, len(y_labels) * 40 + 80),
        margin={"l": 200, "r": 40, "t": 50, "b": 60},
        template="plotly_white",
        font={"size": 12},
        showlegend=False,
        clickmode="event+select",
    )

    # Reverse y-axis so Phase 1 is at top
    fig.update_yaxes(autorange="reversed")

    return fig


# ---------------------------------------------------------------------------
# Metric Progression Line Charts
# ---------------------------------------------------------------------------

def build_metric_progression(
    experiments: list[Experiment],
    era: str = "All",
    selected_phase: Optional[int] = None,
) -> list[go.Figure]:
    """Build 3 metric progression line charts (overall, retrieval, plausibility).

    Features:
    - Phase background shading (non-overlapping, colored bands with labels)
    - Chronological ordering by phase (old-era before LME-era)
    - Y-axis fixed [0,1]
    - Scoring discontinuity at Phase 4/5 boundary
    - validation_failed as red X markers, excluded from line interpolation
    - No line between non-contiguous experiment IDs

    Args:
        experiments: All loaded experiments.
        era: Current era filter.
        selected_phase: Phase to highlight (others de-emphasized).

    Returns:
        List of 3 Plotly Figures.
    """
    sorted_exps = _get_sorted_experiments(experiments, era)

    # Build x-positions (sequential index) and data arrays
    x_positions: list[int] = []
    phase_ids: list[Optional[int]] = []
    exp_ids: list[str] = []
    validation_failed_flags: list[bool] = []

    for i, exp in enumerate(sorted_exps):
        x_positions.append(i)
        phase_ids.append(exp.phase)
        exp_ids.append(exp.id)
        validation_failed_flags.append(exp.status == "validation_failed")

    # Phase boundaries for shading
    phase_boundaries: dict[int, tuple[int, int]] = {}
    for i, exp in enumerate(sorted_exps):
        p = exp.phase
        if p is not None:
            if p not in phase_boundaries:
                phase_boundaries[p] = (i, i)
            else:
                phase_boundaries[p] = (phase_boundaries[p][0], i)

    # Metric configs
    metric_configs = [
        ("overall_score", "Overall Score", "#1565C0"),
        ("retrieval_score", "Retrieval Score", "#2E7D32"),
        ("plausibility_score", "Plausibility Score", "#E65100"),
    ]

    figures: list[go.Figure] = []

    for metric_key, metric_name, color in metric_configs:
        fig = go.Figure()

        # Add phase background shading
        for phase_num, (start, end) in phase_boundaries.items():
            phase_color = PHASE_COLORS.get(phase_num, "#999")
            opacity = 0.25
            if selected_phase is not None and phase_num != selected_phase:
                opacity = 0.08  # De-emphasize non-selected phases

            fig.add_vrect(
                x0=start - 0.5, x1=end + 0.5,
                fillcolor=phase_color,
                opacity=opacity,
                layer="below",
                annotation_text=f"P{phase_num}" if (end - start) > 5 else "",
                annotation_position="top left",
                annotation_font={"size": 10, "color": phase_color},
                annotation_opacity=0.7,
            )

        # Separate data into contiguous groups (split by non-contiguous IDs)
        # Also separate validation_failed from line data
        line_x: list[float] = []
        line_y: list[float] = []
        vf_x: list[float] = []
        vf_y: list[float] = []
        hover_texts: list[str] = []

        current_era: Optional[str] = None
        prev_num_id: Optional[int] = None
        prev_exp_era: Optional[str] = None

        def _extract_num_id(exp: Experiment) -> int:
            if exp.era == "memories_500":
                return int(exp.id.split("_")[1])
            else:
                return int(exp.id.split("_")[2])

        # Build segments (groups of contiguous IDs within same era)
        segments: list[list[tuple[float, float, str]]] = []
        current_segment: list[tuple[float, float, str]] = []

        for i, exp in enumerate(sorted_exps):
            val = getattr(exp, metric_key, None)
            num_id = _extract_num_id(exp)

            is_vf = exp.status == "validation_failed"

            # Add validation_failed as red X markers
            if is_vf:
                vf_x.append(i)
                vf_y.append(0.5)  # Place in middle of chart
                hover_texts.append(
                    f"{exp.id}<br>{metric_name}: validation_failed<br>"
                    f"Phase {exp.phase} · {exp.era}"
                )
                # Don't add to line data, but also break the line
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                prev_num_id = num_id
                prev_exp_era = exp.era
                continue

            # Check contiguity (within same era)
            if prev_num_id is not None:
                if exp.era != prev_exp_era or num_id != prev_num_id + 1:
                    # Break the line - non-contiguous
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []

            if val is not None:
                current_segment.append((float(i), float(val), exp.id))

            prev_num_id = num_id
            prev_exp_era = exp.era

        if current_segment:
            segments.append(current_segment)

        # Draw line segments
        for seg in segments:
            if len(seg) < 2:
                continue
            seg_x = [s[0] for s in seg]
            seg_y = [s[1] for s in seg]
            seg_labels = [s[2] for s in seg]

            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y,
                mode="lines",
                line={"color": color, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            ))

            # Add markers for individual points (only when we have a few per segment)
            if len(seg) <= 100:
                fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y,
                    mode="markers",
                    marker={"color": color, "size": 4},
                    customdata=seg_labels,
                    hovertemplate="%{customdata}<br>" + metric_name + ": %{y:.4f}<extra></extra>",
                    showlegend=False,
                ))

        # Add single-point markers (segments with 1 point)
        for seg in segments:
            if len(seg) == 1:
                fig.add_trace(go.Scatter(
                    x=[seg[0][0]], y=[seg[0][1]],
                    mode="markers",
                    marker={"color": color, "size": 4},
                    customdata=[seg[0][2]],
                    hovertemplate="%{customdata}<br>" + metric_name + ": %{y:.4f}<extra></extra>",
                    showlegend=False,
                ))

        # Add validation_failed red X markers
        if vf_x:
            fig.add_trace(go.Scatter(
                x=vf_x, y=vf_y,
                mode="markers",
                marker={"color": "#D32F2F", "size": 10, "symbol": "x", "line": {"width": 2}},
                customdata=[exp_ids[i] for i in range(len(sorted_exps)) if validation_failed_flags[i]],
                hovertemplate="%{customdata}<br>" + metric_name + ": validation_failed<extra></extra>",
                showlegend=False,
                name="validation_failed",
            ))

        # Add Phase 4/5 scoring discontinuity indicator
        if 4 in phase_boundaries and 5 in phase_boundaries:
            p4_end = phase_boundaries[4][1]
            p5_start = phase_boundaries[5][0]
            boundary_x = (p4_end + p5_start) / 2.0

            fig.add_vline(
                x=boundary_x,
                line_dash="dash",
                line_color="#D32F2F",
                line_width=2,
                annotation_text="⚠ Scoring formula change<br>(scores not directly comparable)",
                annotation_position="top",
                annotation_font={"size": 10, "color": "#D32F2F"},
                annotation_bgcolor="white",
                annotation_bordercolor="#D32F2F",
                annotation_borderwidth=1,
            )

        fig.update_layout(
            title=metric_name,
            yaxis_title="Score",
            yaxis={"range": [0, 1]},
            xaxis_title="Experiment Sequence",
            height=280,
            margin={"l": 50, "r": 20, "t": 40, "b": 40},
            template="plotly_white",
            font={"size": 11},
            showlegend=False,
            hoverlabel={"namelength": -1},
        )

        # Add tick labels showing experiment IDs at sparse intervals
        tick_vals = []
        tick_texts = []
        step = max(1, len(sorted_exps) // 15)
        for i in range(0, len(sorted_exps), step):
            tick_vals.append(i)
            tick_texts.append(sorted_exps[i].id)
        if (len(sorted_exps) - 1) % step != 0:
            tick_vals.append(len(sorted_exps) - 1)
            tick_texts.append(sorted_exps[-1].id)

        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_texts, tickangle=-45)

        figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# Threshold Heatmap
# ---------------------------------------------------------------------------

def build_threshold_heatmap(
    experiments: list[Experiment],
    era: str = "All",
    selected_phase: Optional[int] = None,
) -> list[go.Figure]:
    """Build threshold metric heatmaps (recall and precision).

    Features:
    - Rows: thresholds 0.1-0.9
    - Columns: experiments
    - Linear color scale [0,1] with legend
    - Old-era blank cells for missing thresholds
    - Hover shows exact value
    - 2 panels: recall and precision

    Args:
        experiments: All loaded experiments.
        era: Current era filter.
        selected_phase: Currently selected phase filter.

    Returns:
        List of 2 Plotly Figures (recall heatmap, precision heatmap).
    """
    sorted_exps = _get_sorted_experiments(experiments, era)

    # If too many experiments, limit to those with threshold data
    exps_with_thresholds = [e for e in sorted_exps if e.threshold_metrics]

    # Limit columns for performance: show at most 100 experiments
    display_exps = exps_with_thresholds
    if len(display_exps) > 100:
        # Take evenly spaced samples
        step = len(display_exps) // 100
        display_exps = display_exps[::step]

    # Build data matrices
    n_thresholds = len(THRESHOLD_VALUES)
    n_exps = len(display_exps)

    # Recall matrix
    recall_z = []
    recall_text = []
    precision_z = []
    precision_text = []

    for threshold in THRESHOLD_VALUES:
        t_key = str(threshold)
        recall_row = []
        recall_text_row = []
        precision_row = []
        precision_text_row = []

        for exp in display_exps:
            t_data = exp.threshold_metrics.get(t_key, {})
            recall_val = t_data.get("recall_rate")
            precision_val = t_data.get("precision_rate")

            if recall_val is not None:
                recall_row.append(float(recall_val))
                recall_text_row.append(f"{float(recall_val):.4f}")
            else:
                recall_row.append(None)  # Blank cell
                recall_text_row.append("")

            if precision_val is not None:
                precision_row.append(float(precision_val))
                precision_text_row.append(f"{float(precision_val):.4f}")
            else:
                precision_row.append(None)
                precision_text_row.append("")

        recall_z.append(recall_row)
        recall_text.append(recall_text_row)
        precision_z.append(precision_row)
        precision_text.append(precision_text_row)

    # Column labels (experiment IDs)
    col_labels = [e.id for e in display_exps]

    figures: list[go.Figure] = []

    for z_data, text_data, title, colorscale_name in [
        (recall_z, recall_text, "Threshold Heatmap — Recall Rate", "Blues"),
        (precision_z, precision_text, "Threshold Heatmap — Precision Rate", "Reds"),
    ]:
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=z_data,
            x=col_labels,
            y=THRESHOLD_LABELS,
            text=text_data,
            texttemplate="%{text}",
            textfont={"size": 8},
            colorscale=colorscale_name,
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar={
                "title": "Score",
                "tickmode": "array",
                "tickvals": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "ticktext": ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
            },
            hovertemplate=(
                "Experiment: %{x}<br>"
                "Threshold: %{y}<br>"
                "Value: %{z:.4f}<extra></extra>"
            ),
            # None values will render as blank
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Experiment",
            yaxis_title="Threshold",
            height=350,
            margin={"l": 60, "r": 20, "t": 40, "b": 80},
            template="plotly_white",
            font={"size": 11},
        )

        # Rotate x-axis labels
        fig.update_xaxes(tickangle=-90, dtick=1)
        # Don't show every label if too many
        if len(col_labels) > 30:
            fig.update_xaxes(tickangle=-90)

        figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# Retention Curve Overlay
# ---------------------------------------------------------------------------

def build_retention_overlay(
    experiments: list[Experiment],
    selected_ids: list[str],
) -> go.Figure:
    """Build retention curve overlay chart.

    Features:
    - Multi-select 2-5 experiments
    - Curves at ticks 40/80/120/160/200
    - Unique colors + legend with experiment ID
    - Clear All control (handled in app callback)

    Args:
        experiments: All loaded experiments.
        selected_ids: List of experiment IDs to overlay.

    Returns:
        Plotly Figure with overlaid retention curves.
    """
    fig = go.Figure()

    for idx, exp_id in enumerate(selected_ids):
        exp = next((e for e in experiments if e.id == exp_id), None)
        if exp is None:
            continue

        # Get retention curve data
        retention = exp.retention_curve
        if retention is None:
            # Fallback: check final snapshot
            if exp.snapshots:
                for snap in reversed(exp.snapshots):
                    if snap.get("tick") == 200 and snap.get("retention_curve"):
                        retention = snap["retention_curve"]
                        break

        if retention is None:
            # No retention data - show message via annotation
            continue

        # Extract values at standard ticks
        x_vals = []
        y_vals = []
        for tick in RETENTION_TICKS:
            tick_key = str(tick)
            if tick_key in retention:
                x_vals.append(tick)
                y_vals.append(float(retention[tick_key]))

        if not x_vals:
            continue

        color = RETENTION_COLORS[idx % len(RETENTION_COLORS)]

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            name=exp_id,
            line={"color": color, "width": 2.5},
            marker={"size": 8},
            hovertemplate=(
                f"{exp_id}<br>"
                "Tick: %{x}<br>"
                "Retention: %{y:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Retention Curve Overlay",
        xaxis_title="Simulation Tick",
        yaxis_title="Retention Rate",
        xaxis={"tickvals": RETENTION_TICKS},
        yaxis={"range": [0, 1]},
        height=350,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        template="plotly_white",
        font={"size": 12},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    if not selected_ids:
        fig.add_annotation(
            text="Select 2-5 experiments to compare retention curves",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#9E9E9E"},
        )

    return fig


def get_retention_available_experiments(
    experiments: list[Experiment],
    era: str = "All",
) -> list[str]:
    """Get list of experiment IDs that have retention curve data.

    Fallback hierarchy: top-level retention_curve > final snapshot retention_curve.
    """
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    available: list[str] = []
    for exp in filtered:
        has_retention = False

        if exp.retention_curve:
            has_retention = True
        elif exp.snapshots:
            for snap in reversed(exp.snapshots):
                if snap.get("tick") == 200 and snap.get("retention_curve"):
                    has_retention = True
                    break

        if has_retention:
            available.append(exp.id)

    return available


def check_retention_warnings(
    selected_ids: list[str],
    experiments: list[Experiment],
) -> list[str]:
    """Check which selected experiments lack retention data and return warning messages."""
    warnings: list[str] = []
    for exp_id in selected_ids:
        exp = next((e for e in experiments if e.id == exp_id), None)
        if exp is None:
            warnings.append(f"{exp_id}: not found")
            continue

        has_retention = False
        if exp.retention_curve:
            has_retention = True
        elif exp.snapshots:
            for snap in reversed(exp.snapshots):
                if snap.get("tick") == 200 and snap.get("retention_curve"):
                    has_retention = True
                    break

        if not has_retention:
            warnings.append(f"{exp_id}: no retention data available")

    return warnings


# ---------------------------------------------------------------------------
# Parameter Sweep — Parallel Coordinates
# ---------------------------------------------------------------------------

# Parameters that appear frequently enough across eras to be useful dimensions
_COMMON_PARAMS: list[str] = [
    "lambda_fact", "lambda_episode", "alpha", "stability_weight",
    "stability_decay", "reinforcement_gain_direct", "reinforcement_gain_assoc",
    "stability_cap", "floor_max", "sigmoid_k", "sigmoid_mid",
    "jost_power", "activation_weight",
]

# Extra params specific to old era (beta_*)
_OLD_ERA_EXTRA: list[str] = [
    "beta_fact", "beta_episode", "floor_min", "floor_power",
    "consolidation_threshold", "sigmoid_steepness",
]


def get_param_sweep_dimensions(
    experiments: list[Experiment],
    era: str = "All",
    enabled_params: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Determine parallel coordinates dimensions based on era and param coverage.

    Returns:
        (dimensions, available_params) where dimensions is a list of
        Plotly Parcoords dimension dicts, and available_params is the
        full list of eligible param names for the toggle selector.
    """
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    # Only consider experiments with params
    with_params = [e for e in filtered if e.params]

    if not with_params:
        return [], []

    # Count param frequency
    param_count: Counter[str] = Counter()
    param_values: dict[str, list] = {}
    for e in with_params:
        for k, v in e.params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                param_count[k] += 1
                param_values.setdefault(k, []).append(v)

    total = len(with_params)

    # Select candidate params: those with >= 10% coverage
    candidates = [
        k for k, c in param_count.most_common()
        if c / total >= 0.10 and len(param_values[k]) > 1  # >1 unique values needed
    ]

    # If era is specific, use its param set; if "All", hide params with >50% null
    if era == "All":
        candidates = [k for k in candidates if param_count[k] / total > 0.50]

    # Build dimension dicts
    dimensions = []
    available = list(candidates)  # for toggle selector

    # If user selected specific params, filter
    if enabled_params is not None:
        candidates = [k for k in candidates if k in enabled_params]

    for param in candidates:
        vals = param_values.get(param, [])
        if not vals:
            continue
        # Filter out None/NaN
        clean_vals = [v for v in vals if v is not None]
        if not clean_vals:
            continue
        vmin = min(clean_vals)
        vmax = max(clean_vals)
        # Add 5% padding
        pad = (vmax - vmin) * 0.05 if vmax != vmin else 0.01
        dimensions.append({
            "label": param,
            "values": [e.params.get(param) for e in with_params],
            "range": [vmin - pad, vmax + pad],
        })

    return dimensions, available


def build_parameter_sweep(
    experiments: list[Experiment],
    era: str = "All",
    enabled_params: list[str] | None = None,
) -> go.Figure:
    """Build parallel coordinates plot colored by overall_score.

    Features:
    - Era-specific parameter dimensions
    - 'All' era hides params with >50% null
    - Parameter toggle selector (enabled_params)
    - Color scale maps to overall_score
    - Hover shows param values and score
    - Click navigates to experiment detail

    Args:
        experiments: All loaded experiments.
        era: Current era filter.
        enabled_params: List of param names to show as dimensions.

    Returns:
        Plotly Figure with parallel coordinates.
    """
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    # Only experiments with params AND overall_score
    scored = [e for e in filtered if e.params and e.overall_score is not None]

    if not scored:
        fig = go.Figure()
        fig.add_annotation(
            text="No experiments with parameter data and scores in current view",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    dimensions, available = get_param_sweep_dimensions(experiments, era, enabled_params)

    if not dimensions:
        fig = go.Figure()
        fig.add_annotation(
            text="No parameter dimensions available with sufficient coverage",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    # Color line by overall_score
    line_color = [e.overall_score for e in scored]

    # Custom data for hover: experiment IDs
    custom_data = [e.id for e in scored]

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=line_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Overall Score"),
            cmin=0,
            cmax=1,
        ),
        dimensions=dimensions,
        customdata=custom_data,
        labelangle=-30,
    ))

    # Note: go.Parcoords doesn't directly support clickData like scatter.
    # Navigation to detail will be handled via the experiment dropdown
    # in the parameter sweep view.

    era_label = era if era != "All" else "both eras"
    fig.update_layout(
        title=f"Parameter Sweep ({era_label})",
        height=500,
        margin={"l": 80, "r": 40, "t": 50, "b": 60},
        template="plotly_white",
        font={"size": 11},
    )

    return fig


def get_param_sweep_available_params(
    experiments: list[Experiment],
    era: str = "All",
) -> list[str]:
    """Get list of parameter names eligible for the toggle selector."""
    _, available = get_param_sweep_dimensions(experiments, era)
    return available


# ---------------------------------------------------------------------------
# Snapshot Viewer — Expanded Multi-Metric View
# ---------------------------------------------------------------------------

# Snapshot metric traces to display
_SNAPSHOT_METRICS: list[tuple[str, str, str]] = [
    ("overall_score", "Overall Score", "#1565C0"),
    ("recall_mean", "Recall Mean", "#2E7D32"),
    ("precision_mean", "Precision Mean", "#E65100"),
    ("plausibility_score", "Plausibility Score", "#6A1B9A"),
    ("mrr_mean", "MRR Mean", "#C62828"),
]


def build_snapshot_viewer(
    experiments: list[Experiment],
    exp_id: str,
    current_tick: int | None = None,
) -> go.Figure:
    """Build expanded snapshot viewer chart with multiple metric traces.

    Features:
    - Shows overall, recall, precision, plausibility, mrr over 11 ticks
    - Stepping controls (highlight current tick)
    - Animation mode (visual indicator)
    - Distinct from detail view mini-charts

    Args:
        experiments: All loaded experiments.
        exp_id: Selected experiment ID.
        current_tick: Currently highlighted tick (for stepping).

    Returns:
        Plotly Figure with multi-metric snapshot view.
    """
    exp = next((e for e in experiments if e.id == exp_id), None)

    fig = go.Figure()

    if exp is None or not exp.snapshots:
        fig.add_annotation(
            text="No snapshot data available for this experiment",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    ticks = [s.get("tick", i * 20) for i, s in enumerate(exp.snapshots)]
    has_any_trace = False

    for key, label, color in _SNAPSHOT_METRICS:
        values = []
        t_vals = []
        for s in exp.snapshots:
            val = s.get(key)
            if val is not None:
                t_vals.append(s.get("tick", 0))
                values.append(float(val))

        if values:
            has_any_trace = True
            fig.add_trace(go.Scatter(
                x=t_vals, y=values,
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 2.5},
                marker={"size": 6},
                hovertemplate=(
                    f"{exp_id}<br>"
                    f"{label}: %{{y:.4f}}<br>"
                    "Tick: %{x}<extra></extra>"
                ),
            ))

    if not has_any_trace:
        fig.add_annotation(
            text="No metric data in snapshots",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    # Add vertical line for current tick (stepping indicator)
    if current_tick is not None:
        fig.add_vline(
            x=current_tick,
            line_dash="dash", line_color="#D32F2F", line_width=2,
            annotation_text=f"Tick {current_tick}",
            annotation_position="top left",
            annotation_font={"size": 10, "color": "#D32F2F"},
        )

    fig.update_layout(
        title=f"Snapshot Viewer — {exp_id}",
        xaxis_title="Simulation Tick",
        yaxis_title="Score",
        yaxis={"range": [0, 1]},
        xaxis={"tickvals": list(range(0, 201, 20))},
        height=450,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        font={"size": 12},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def get_snapshot_tick_data(
    experiments: list[Experiment],
    exp_id: str,
    tick: int,
) -> dict[str, float | None]:
    """Get all metric values for a specific tick in an experiment's snapshots."""
    exp = next((e for e in experiments if e.id == exp_id), None)
    if exp is None or not exp.snapshots:
        return {}

    snapshot = next((s for s in exp.snapshots if s.get("tick") == tick), None)
    if snapshot is None:
        return {}

    result: dict[str, float | None] = {}
    for key, label, _ in _SNAPSHOT_METRICS:
        val = snapshot.get(key)
        result[key] = float(val) if val is not None else None
        result[f"{key}_label"] = label

    # Also include eval_v2, strict, selectivity if available
    for extra_key in ["eval_v2_score", "selectivity_score", "robustness_score"]:
        val = snapshot.get(extra_key)
        if val is not None:
            result[extra_key] = float(val)

    return result


# ---------------------------------------------------------------------------
# Forgetting Depth & Strict Validation Analysis
# ---------------------------------------------------------------------------

def build_forgetting_depth_chart(
    experiments: list[Experiment],
    era: str = "All",
) -> go.Figure:
    """Build forgetting depth distribution chart with pass/fail classification.

    Features:
    - Only experiments with forgetting_depth data shown
    - Pass/fail classification (strict_score thresholds or status)
    - Distribution bar chart

    Args:
        experiments: All loaded experiments.
        era: Current era filter.

    Returns:
        Plotly Figure with forgetting depth distribution.
    """
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    # Only experiments with forgetting_depth
    with_data = [e for e in filtered if e.forgetting_depth is not None]

    if not with_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No experiments with forgetting_depth data in current view",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    # Classify pass/fail based on strict_score
    passed: list[Experiment] = []
    failed: list[Experiment] = []
    no_strict: list[Experiment] = []

    for e in with_data:
        if e.strict_score is not None:
            if e.status == "validation_failed":
                failed.append(e)
            elif e.strict_score >= 0.4:
                passed.append(e)
            else:
                failed.append(e)
        else:
            no_strict.append(e)

    fig = go.Figure()

    # Passed experiments (green)
    if passed:
        fig.add_trace(go.Bar(
            x=[e.id for e in passed],
            y=[e.forgetting_depth for e in passed],
            name="Passed",
            marker_color="#4CAF50",
            text=[f"{e.forgetting_depth:.4f}" for e in passed],
            textposition="auto",
            hovertemplate=(
                "%{x}<br>"
                "Forgetting Depth: %{y:.4f}<br>"
                "Strict Score: %{customdata:.4f}<extra></extra>"
            ),
            customdata=[e.strict_score for e in passed],
        ))

    # Failed experiments (red)
    if failed:
        fig.add_trace(go.Bar(
            x=[e.id for e in failed],
            y=[e.forgetting_depth for e in failed],
            name="Failed",
            marker_color="#EF5350",
            text=[f"{e.forgetting_depth:.4f}" for e in failed],
            textposition="auto",
            hovertemplate=(
                "%{x}<br>"
                "Forgetting Depth: %{y:.4f}<br>"
                "Status: %{customdata}<extra></extra>"
            ),
            customdata=[e.status for e in failed],
        ))

    # No strict score (gray)
    if no_strict:
        fig.add_trace(go.Bar(
            x=[e.id for e in no_strict],
            y=[e.forgetting_depth for e in no_strict],
            name="No Strict Score",
            marker_color="#BDBDBD",
            text=[f"{e.forgetting_depth:.4f}" for e in no_strict],
            textposition="auto",
            hovertemplate="%{x}<br>Forgetting Depth: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title="Forgetting Depth Distribution (Pass/Fail Classification)",
        xaxis_title="Experiment",
        yaxis_title="Forgetting Depth",
        barmode="group",
        height=400,
        margin={"l": 60, "r": 20, "t": 50, "b": 100},
        template="plotly_white",
        font={"size": 11},
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis={"tickangle": -45},
    )

    return fig


def build_strict_score_chart(
    experiments: list[Experiment],
    era: str = "All",
) -> go.Figure:
    """Build strict_score distribution chart alongside forgetting_depth.

    Args:
        experiments: All loaded experiments.
        era: Current era filter.

    Returns:
        Plotly Figure with strict_score distribution.
    """
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    with_data = [e for e in filtered if e.strict_score is not None]

    if not with_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No experiments with strict_score data in current view",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    # Classify
    passed_ids = []
    passed_scores = []
    failed_ids = []
    failed_scores = []

    for e in with_data:
        if e.status == "validation_failed":
            failed_ids.append(e.id)
            failed_scores.append(e.strict_score)
        elif e.strict_score >= 0.4:
            passed_ids.append(e.id)
            passed_scores.append(e.strict_score)
        else:
            failed_ids.append(e.id)
            failed_scores.append(e.strict_score)

    fig = go.Figure()

    if passed_ids:
        fig.add_trace(go.Bar(
            x=passed_ids, y=passed_scores,
            name="Passed (≥0.4)", marker_color="#4CAF50",
            text=[f"{s:.4f}" for s in passed_scores], textposition="auto",
            hovertemplate="%{x}<br>Strict Score: %{y:.4f}<extra></extra>",
        ))

    if failed_ids:
        fig.add_trace(go.Bar(
            x=failed_ids, y=failed_scores,
            name="Failed (<0.4 / validation_failed)", marker_color="#EF5350",
            text=[f"{s:.4f}" for s in failed_scores], textposition="auto",
            hovertemplate="%{x}<br>Strict Score: %{y:.4f}<extra></extra>",
        ))

    # Add threshold line at 0.4
    fig.add_hline(
        y=0.4, line_dash="dash", line_color="#FF9800", line_width=2,
        annotation_text="Pass threshold (0.4)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Strict Score Distribution (Pass/Fail Classification)",
        xaxis_title="Experiment",
        yaxis_title="Strict Score",
        yaxis={"range": [0, 1]},
        barmode="group",
        height=400,
        margin={"l": 60, "r": 20, "t": 50, "b": 100},
        template="plotly_white",
        font={"size": 11},
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis={"tickangle": -45},
    )

    return fig


# ---------------------------------------------------------------------------
# CV Results Visualization
# ---------------------------------------------------------------------------

def build_cv_fold_scores(
    cv_data: dict,
    exp_id: str,
) -> go.Figure:
    """Build fold scores bar chart with mean±std and worst-fold highlight.

    Args:
        cv_data: Parsed cv_results.json dict.
        exp_id: Experiment ID (for title).

    Returns:
        Plotly Figure with fold scores bar chart.
    """
    fold_scores = cv_data.get("fold_scores", [])
    cv_mean = cv_data.get("mean", {})
    cv_std = cv_data.get("std", {})
    cv_worst = cv_data.get("worst_fold", {})

    if not fold_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No fold score data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    # Extract overall_score from each fold
    fold_labels = []
    fold_values = []
    for i, fs in enumerate(fold_scores):
        if isinstance(fs, dict):
            score = fs.get("overall_score")
            if score is not None:
                fold_labels.append(f"Fold {i + 1}")
                fold_values.append(float(score))

    if not fold_values:
        fig = go.Figure()
        fig.add_annotation(
            text="No fold overall_score data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    # Determine worst fold
    worst_score = None
    if isinstance(cv_worst, dict):
        worst_score = cv_worst.get("overall_score")

    mean_val = cv_mean.get("overall_score") if isinstance(cv_mean, dict) else None
    std_val = cv_std.get("overall_score") if isinstance(cv_std, dict) else None

    # Color bars: worst fold in red, others based on mean comparison
    colors = []
    for v in fold_values:
        if worst_score is not None and abs(v - worst_score) < 1e-6:
            colors.append("#EF5350")
        elif mean_val is not None and v >= mean_val:
            colors.append("#4CAF50")
        else:
            colors.append("#FF9800")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fold_labels, y=fold_values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in fold_values],
        textposition="auto",
        hovertemplate="Fold: %{x}<br>Score: %{y:.4f}<extra></extra>",
    ))

    # Mean line
    if mean_val is not None:
        mean_label = f"Mean: {mean_val:.4f}"
        if std_val is not None:
            mean_label += f" ± {std_val:.4f}"
        fig.add_hline(
            y=mean_val,
            line_dash="dash", line_color="#2196F3", line_width=2,
            annotation_text=mean_label,
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"Cross-Validation Fold Scores — {exp_id}",
        xaxis_title="Fold",
        yaxis_title="Overall Score",
        yaxis={"range": [0, 1]},
        height=350,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        font={"size": 12},
        showlegend=False,
    )

    return fig


def build_cv_fold_deltas(
    cv_data: dict,
    exp_id: str,
) -> go.Figure:
    """Build fold deltas chart showing deviation from mean.

    Args:
        cv_data: Parsed cv_results.json dict.
        exp_id: Experiment ID (for title).

    Returns:
        Plotly Figure with fold deltas bar chart.
    """
    fold_deltas = cv_data.get("fold_deltas", [])

    if not fold_deltas:
        fig = go.Figure()
        fig.add_annotation(
            text="No fold delta data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 14, "color": "#9E9E9E"},
        )
        fig.update_layout(template="plotly_white")
        return fig

    fold_labels = [f"Fold {i + 1}" for i in range(len(fold_deltas))]
    colors = ["#4CAF50" if d >= 0 else "#EF5350" for d in fold_deltas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fold_labels, y=fold_deltas,
        marker_color=colors,
        text=[f"{d:+.4f}" for d in fold_deltas],
        textposition="auto",
        hovertemplate="Fold: %{x}<br>Delta: %{y:+.4f}<extra></extra>",
    ))

    fig.add_hline(
        y=0, line_dash="solid", line_color="#424242", line_width=1,
    )

    fig.update_layout(
        title=f"Fold Deltas (deviation from mean) — {exp_id}",
        xaxis_title="Fold",
        yaxis_title="Score Delta",
        height=300,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        font={"size": 12},
        showlegend=False,
    )

    return fig


def get_cv_available_experiments(
    experiments: list[Experiment],
    cv_data: dict[str, dict],
    era: str = "All",
) -> list[str]:
    """Get list of experiment IDs that have cv_results.json."""
    filtered = experiments
    if era != "All":
        filtered = [e for e in experiments if e.era == era]

    return [e.id for e in filtered if e.id in cv_data]


# ---------------------------------------------------------------------------
# Phase Comparison Panel
# ---------------------------------------------------------------------------

_PHASE_COMPARISON_METRICS = [
    ("overall_score", "Overall Score"),
    ("retrieval_score", "Retrieval Score"),
    ("plausibility_score", "Plausibility Score"),
    ("recall_mean", "Recall Mean"),
    ("retention_auc", "Retention AUC"),
]


def build_phase_comparison(
    experiments: list[Experiment],
    phase_a: int,
    phase_b: int,
) -> dict[str, Any]:
    """Build phase comparison statistics.

    Args:
        experiments: All loaded experiments.
        phase_a: First phase number.
        phase_b: Second phase number.

    Returns:
        Dict with comparison data for rendering.
    """
    result: dict[str, Any] = {
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_a_name": PHASE_NAMES.get(phase_a, f"Phase {phase_a}"),
        "phase_b_name": PHASE_NAMES.get(phase_b, f"Phase {phase_b}"),
        "insufficient_data": False,
        "metrics": [],
    }

    # Gather experiments for each phase, excluding validation_failed
    exps_a = [e for e in experiments if e.phase == phase_a and e.status != "validation_failed"]
    exps_b = [e for e in experiments if e.phase == phase_b and e.status != "validation_failed"]

    for phase_key, exps in [("a", exps_a), ("b", exps_b)]:
        scored = [e for e in exps if e.overall_score is not None]
        result[f"count_{phase_key}"] = len(exps)
        result[f"scored_count_{phase_key}"] = len(scored)

    # Check insufficient data
    if result["scored_count_a"] == 0 and result["scored_count_b"] == 0:
        result["insufficient_data"] = True
        return result

    for metric_key, metric_label in _PHASE_COMPARISON_METRICS:
        # Calculate stats for each phase
        stats_a = _calc_phase_metric_stats(exps_a, metric_key)
        stats_b = _calc_phase_metric_stats(exps_b, metric_key)

        # Calculate improvement percentage
        improvement = None
        if stats_a["mean"] is not None and stats_b["mean"] is not None and stats_a["mean"] != 0:
            improvement = ((stats_b["mean"] - stats_a["mean"]) / abs(stats_a["mean"])) * 100

        result["metrics"].append({
            "key": metric_key,
            "label": metric_label,
            "phase_a": stats_a,
            "phase_b": stats_b,
            "improvement_pct": improvement,
        })

    return result


def _calc_phase_metric_stats(
    exps: list[Experiment],
    metric_key: str,
) -> dict[str, Any]:
    """Calculate mean, best, count for a metric in a set of experiments."""
    values = []
    for e in exps:
        val = getattr(e, metric_key, None)
        if val is not None:
            values.append(float(val))

    if not values:
        return {"mean": None, "best": None, "count": 0}

    return {
        "mean": sum(values) / len(values),
        "best": max(values),
        "count": len(values),
    }


def get_available_phases(
    experiments: list[Experiment],
) -> list[dict[str, Any]]:
    """Get list of phases that have experiments, for the comparison selector."""
    phases: dict[int, int] = {}
    for e in experiments:
        if e.phase is not None:
            phases[e.phase] = phases.get(e.phase, 0) + 1

    return [
        {"label": f"Phase {p}: {PHASE_NAMES.get(p, 'Unknown')} ({count} exps)",
         "value": p}
        for p, count in sorted(phases.items())
    ]
