# Scissor Linkage Optimizer

Parametric solver for deployable scissor-linkage mechanisms. Sweeps offset and plate thickness, finds all feasible geometries, and produces an interactive HTML explorer with 3D viewer and CAD-ready CSV export.

---

## Quick Start

### 1. Install dependencies

```bash
pip install pyomo numpy plotly
```

You also need a nonlinear solver. **Bonmin** is preferred (handles integer variables for `n_cells`); **Ipopt** works too if `n` is fixed. Install via conda or your system package manager:

```bash
conda install -c conda-forge coinbonmin    # recommended
# or
conda install -c conda-forge ipopt
```

### 2. Run a sweep

```bash
python optimize.py \
  --stow-width 1.0 \
  --stow-depth 2.0 \
  --expanded-depth 5.0 \
  --offset-min 0.01 --offset-max 2.0 --offset-steps 50 \
  --thickness-list 0.002,0.005,0.01,0.02
```

This takes a few minutes depending on your machine. Progress prints to stdout.

### 3. Open the output

Two files are generated:

| File | What it is |
|------|------------|
| `scissor_results.json` | Raw solver output for every feasible point |
| `scissor_sweep.html` | **Interactive explorer** — open in any browser |

Open the HTML file and you're ready to go. Everything is self-contained (Plotly, Three.js, and all data are embedded in the single file).

---

## Using the Interactive Explorer

The HTML has three panels:

**Scatter Plot (top-left)** — Every dot is a feasible solution. Axes show extension ratios by default; use the toggle button to switch to actual dimensions (mm). Hover for a tooltip with key parameters. Color = plate thickness.

**3D Viewer (right)** — Click any point in the plot to load its geometry. Drag to orbit, right-drag to pan, scroll to zoom. Use the deployment slider (α) to animate from stowed to deployed. The cells slider lets you show a subset of cells for clarity.

**CSV Export (bottom-left)** — Shows the selected solution's dimensions rounded to 0.1 mm manufacturing resolution, alongside the raw solver values. A residuals table shows how much each kinematic constraint is violated by the rounding. Export downloads a CSV formatted for direct import into Fusion 360 / SolidWorks user parameters.

### Reading the residuals

After you click a point, the export panel shows constraint residuals in mm. These tell you how much the kinematic equalities are violated when you round to 0.1 mm:

| Color | Meaning |
|-------|---------|
| Green | < 0.001 mm — negligible, well within any tolerance |
| Grey | < 0.01 mm — fine for most applications |
| Gold | < 0.1 mm — worth noting, still likely acceptable |
| Red | ≥ 0.1 mm — review this constraint carefully |

The constraints checked are:

- **cell depth match** — long-arm and short-arm cell depths must be equal (stow & deploy)
- **initial state** — stowed geometry closure condition
- **final state** — deployed geometry closure condition
- **initial angle beta** — stow angle set by plate thickness
- **stow width / depth / expanded depth** — target envelope dimensions

In practice, rounding to 0.1 mm on members that are hundreds of mm long produces residuals in the sub-micron range. You'll almost always see all green.

---

## CLI Reference

### Required — Envelope constraints (meters)

At least one should be set; the rest can be left free (`None`).

| Flag | Description |
|------|-------------|
| `--stow-width` | Target stowed width (m) |
| `--stow-depth` | Target stowed depth / length (m) |
| `--expanded-depth` | Target deployed depth / length (m) |

### Sweep range

| Flag | Default | Description |
|------|---------|-------------|
| `--offset-min` | 0.001 | Smallest offset to try (m) |
| `--offset-max` | 2.0 | Largest offset to try (m) |
| `--offset-steps` | 50 | Number of offset values in the sweep |
| `--thickness-list` | — | Explicit comma-separated thicknesses (m), e.g. `0.002,0.005,0.01` |
| `--thickness-min` | 0.002 | Min thickness if not using `--thickness-list` |
| `--thickness-max` | 0.05 | Max thickness if not using `--thickness-list` |
| `--thickness-steps` | 8 | Number of thickness values if not using `--thickness-list` |

### Optional — Fix geometry variables

Any of these can be set to fix the variable instead of letting the solver choose it.

| Flag | Description |
|------|-------------|
| `--hinge-par` | Hinge parallel dimension (m) |
| `--n` | Number of scissor cells (integer) |
| `--short` | Short member hole-to-hole length (m) |
| `--long` | Long member hole-to-hole length (m) |

### Output & display

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output` | `scissor_results.json` | JSON output path |
| `--plot-html` | `scissor_sweep.html` | HTML explorer output path |
| `--plot-mode` | `ratio` | Initial plot axes: `ratio` or `actual` |
| `--actual-x` | `depth_actual` | X-axis in actual mode |
| `--actual-y` | `height_actual` | Y-axis in actual mode |
| `--tee` | off | Print solver output to stdout |

---

## CSV Export Format

The exported CSV matches Fusion 360's "User Parameters" import format:

```
Name,Unit,Expression,Value,Comments,Favorite
Hole_Diameter,mm,2.00 mm,2.00,,false
Plate_Thickness,mm,5.0 mm,5.0,,false
Member_Width,mm,Plate_Thickness,5.0,,false
Long_H2H,mm,379.8 mm,379.8,,false
Short_H2H,mm,189.9 mm,189.9,,false
Offset_H2H,mm,12.3 mm,12.3,,false
Hinge_Par,mm,25.1 mm,25.1,,false
Hinge_Perp,mm,2.00 mm,2.00,,false
```

All lengths are rounded to 0.1 mm. `Hole_Diameter` and `Hinge_Perp` are editable in the UI before export (they're not solver outputs — set them to match your fastener choice).

---

## How It Works

The solver uses Pyomo to build a nonlinear constraint-satisfaction model for a planar scissor linkage. The key geometry relationships are:

1. **Unit circle constraints** — `sin²(α/2) + cos²(α/2) = 1` for each state (stowed, deployed)
2. **Cell depth match** — long-arm and short-arm contributions to cell depth must be equal: `2(L−O)·sin(α/2) = 2S·sin(β/2)`
3. **Initial state (stowed)** — `L·cos(α/2) = S·cos(β/2) + H` (arms nest flush)
4. **Final state (deployed)** — `2L·cos(α/2) = S·cos(β/2)` (full extension condition)
5. **Beta from thickness** — `sin(β_stow/2) = T/(2S)` (minimum opening angle set by plate thickness)

The sweep iterates over a grid of `(thickness, offset)` pairs. For each, it fixes those two values and solves for the remaining geometry (long, short, hinge_par, n_cells, angles). Infeasible combinations are silently skipped.

After solving, each result is rounded to 0.1 mm and the constraint residuals are computed purely arithmetically (no re-solve) using the original solver angles. This tells you exactly how much manufacturing-resolution rounding perturbs each kinematic relationship.

---

## Typical Workflow

1. **Explore the design space** — run a broad sweep with loose offset range and several thicknesses. Open the HTML, look at the scatter plot to understand the tradeoffs (extension ratio vs. actual size vs. thickness).

2. **Narrow down** — identify a promising region, re-run with tighter offset range and more steps for finer resolution.

3. **Select a point** — click it in the plot. Use the 3D viewer to visually verify stowed and deployed configurations. Check that the achieved dimensions (shown in the export panel) meet your envelope constraints.

4. **Export** — set hole diameter and hinge_perp to match your hardware, verify the residuals are green, and hit Export CSV. Import into your CAD tool.

---

## File Structure

```
optimize.py          ← the whole tool (solver + HTML generator)
scissor_results.json ← generated: all feasible solutions as JSON
scissor_sweep.html   ← generated: interactive explorer (standalone)
scissor_parameters.csv ← generated: CAD export (from browser)
README.md            ← this file
```

No separate build step, no node_modules, no server. One Python file in, one HTML file out.