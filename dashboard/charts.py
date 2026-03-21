"""Chart building functions for the Memory Decay Experiment Dashboard.

Provides Plotly figure builders for:
- Phase timeline (horizontal bar chart)
- Metric progression (line charts with phase shading)
- Threshold heatmap (recall/precision at thresholds 0.1-0.9)
- Retention curve overlay (multi-experiment comparison)

All charts support era filtering, phase highlighting, and follow the
conventions defined in the mission specification.
"""
from __future__ import annotations

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
