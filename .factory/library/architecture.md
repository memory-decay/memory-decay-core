# Architecture

Architectural decisions, patterns, and design choices for the experiment dashboard.

---

## Data Flow
1. `data_loader.py` reads raw JSON files → produces structured Experiment objects
2. Dash app stores loaded data in `dcc.Store` (server-side, shared across callbacks)
3. Callbacks read from Store, apply filters/sorts, produce filtered DataFrames
4. DataTable and Plotly figures consume filtered DataFrames

## State Management
- **Shared state**: `dcc.Store` for era, status filter, search query, selected experiment
- **URL state**: `dcc.Location` for bookmarkability (era, selected experiment)
- **Detail view**: Same-page conditional render (not separate route) — preserves sidebar state naturally

## Filter Propagation
- Era, status, and search filters stored in `dcc.Store`
- All views (table, charts, heatmap, sweep) read from same Store
- Callbacks triggered by Store changes update all dependent outputs
- Phase filter is view-specific (timeline click), also stored in Store

## Navigation Model
- Detail view: same-page overlay (conditional render based on selected_experiment in Store)
- Back button: clears selected_experiment from Store
- Browser back: handled via dcc.Location pop
- Parameter sweep → detail: sets selected_experiment in Store

## Chart Conventions
- All charts use Plotly (go.Figure)
- Y-axis fixed [0,1] for score metrics
- Phase shading: semi-transparent colored rectangles behind data
- Validation-failed markers: red X scatter points
- Color palettes: Plotly defaults (colorblind-safe)
- 4 decimal places for all numeric displays
