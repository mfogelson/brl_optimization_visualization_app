"""
Scissor-linkage optimizer — sweep OFFSET + THICKNESS (DISCRETE 0.001 grid),
collect feasible solutions, and make an interactive plot.

Interactive plot:
- Hover shows thickness + offset (and key vars)
- Toggle button switches between:
    1) Extension ratios (ER_length vs ER_height)
    2) Actual dimensions (choose axes via --actual-x/--actual-y)
- Click a point to show the 3D scissor-linkage viewer + full JSON

Usage:
  python optimize.py --stow-width 1.0 --stow-depth 2.0 --expanded-depth 5.0 \
    --offset-min 0.01 --offset-max 2.0 --offset-steps 50 \
    --thickness-min 0.002 --thickness-max 0.05 --thickness-steps 8

Or pass explicit thickness values:
  python optimize.py ... --thickness-list 0.002,0.005,0.01,0.02
"""

import json
import math
import argparse
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints

# Interactive plotting
try:
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    pd = None
    go = None
    pio = None


THICKNESS_QUANT = 0.001
LENGTH_QUANT = 0.0001  # 0.1 mm resolution for manufacturable lengths


def quantize_thickness(t: float, q: float = THICKNESS_QUANT) -> float:
    """Round to nearest 0.001 (or q)."""
    return float(f"{round(t / q) * q:.3f}")


def quantize_length(v: float, q: float = LENGTH_QUANT) -> float:
    """Round to nearest 0.0001 m (= 0.1 mm)."""
    return round(round(v / q) * q, 4)


def build_model(stow_width, stow_depth, expanded_depth, n_states=2,
                hinge_par=None, thickness=None, n=None,
                short=None, long=None, offset=None):
    m = pyo.ConcreteModel("ScissorLinkage")
    m.I = pyo.RangeSet(1, n_states)

    m.short     = pyo.Var(bounds=(0.01, 10.0), initialize=short if short is not None else 1.0)
    m.long      = pyo.Var(bounds=(0.01, 20.0), initialize=long if long is not None else 2.0)
    m.offset    = pyo.Var(bounds=(0.01, 10.0), initialize=offset if offset is not None else 0.2)
    m.thickness = pyo.Var(bounds=(0.001, 0.5), initialize=thickness if thickness is not None else 0.01)
    m.hinge_par = pyo.Var(bounds=(0.0, 10.0), initialize=hinge_par if hinge_par is not None else 0.05)

    m.n = pyo.Var(bounds=(1, 100), domain=pyo.Integers, initialize=n if n is not None else 10)

    if short is not None:     m.short.fix(short)
    if long is not None:      m.long.fix(long)
    if offset is not None:    m.offset.fix(offset)
    if thickness is not None: m.thickness.fix(thickness)
    if hinge_par is not None: m.hinge_par.fix(hinge_par)
    if n is not None:         m.n.fix(n)

    m.sa = pyo.Var(m.I, bounds=(0.0, 1.0), initialize=0.5)
    m.ca = pyo.Var(m.I, bounds=(0.0, 1.0), initialize=0.866)
    m.sb = pyo.Var(m.I, bounds=(0.0, 1.0), initialize=0.5)
    m.cb = pyo.Var(m.I, bounds=(0.0, 1.0), initialize=0.866)

    @m.Constraint(m.I)
    def unit_alpha(m, i):
        return m.sa[i] ** 2 + m.ca[i] ** 2 == 1.0

    @m.Constraint(m.I)
    def unit_beta(m, i):
        return m.sb[i] ** 2 + m.cb[i] ** 2 == 1.0

    @m.Expression(m.I)
    def a(m, i):
        return m.long * m.ca[i]

    @m.Expression(m.I)
    def b(m, i):
        return m.short * m.cb[i]

    @m.Expression(m.I)
    def c_long(m, i):
        return 2 * (m.long - m.offset) * m.sa[i]

    @m.Expression(m.I)
    def c_short(m, i):
        return 2 * m.short * m.sb[i]

    @m.Expression(m.I)
    def length(m, i):
        return m.c_long[i] * m.n

    @m.Expression(m.I)
    def cos_theta(m, i):
        num = (m.a[i] + m.hinge_par) ** 2
        den = (m.b[i] + 2 * m.hinge_par) ** 2
        return (1.0 - num / den)

    @m.Expression(m.I)
    def height(m, i):
        return (m.b[i] + 2 * m.hinge_par) * m.cos_theta[i]

    @m.Constraint(m.I)
    def cell_depth_match(m, i):
        return m.c_long[i] == m.c_short[i]

    if stow_width is not None:
        m.width_limit = pyo.Constraint(expr=2 * m.b[1] + 4 * m.hinge_par == stow_width)

    if stow_depth is not None:
        m.stow_depth_limit = pyo.Constraint(expr=m.length[1] == stow_depth)

    if expanded_depth is not None:
        m.expanded_depth_target = pyo.Constraint(expr=m.length[n_states] == expanded_depth)

    m.initial_state = pyo.Constraint(expr=m.a[1] == m.b[1] + m.hinge_par)
    m.final_state   = pyo.Constraint(expr=2.0 * m.a[n_states] == m.b[n_states])

    m.initial_angle_beta = pyo.Constraint(expr=m.sb[1] == (m.thickness / 2.0) / m.short)

    m.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    return m


def _pick_solver():
    for name in ("bonmin", "ipopt"):
        solver = SolverFactory(name)
        if solver.available():
            if name == "bonmin":
                solver.options["honor_original_bounds"] = "yes"
            return solver, name
    raise RuntimeError("No solver found (need bonmin or ipopt).")


def solve_one(stow_width, stow_depth, expanded_depth, n_states=2,
              hinge_par=None, thickness=None, n=None, short=None, long=None,
              offset=None, tee=False, debug_infeasible=False):
    model = build_model(stow_width, stow_depth, expanded_depth, n_states,
                        hinge_par, thickness, n, short, long, offset)

    if debug_infeasible:
        log_infeasible_constraints(model, log_expression=True, tol=1e-6)

    solver, solver_name = _pick_solver()
    res = solver.solve(model, tee=tee)
    term = res.solver.termination_condition

    ok_terms = {pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.locallyOptimal,
                pyo.TerminationCondition.feasible}

    if term not in ok_terms:
        raise RuntimeError(f"Solver failed: {solver_name} termination={term}")

    def angle(sa, ca):
        return 2.0 * math.atan2(value(sa), value(ca))

    def theta(i):
        a_val = value(model.a[i])
        b_val = value(model.b[i])
        h_val = value(model.hinge_par)
        num = (2 * a_val + 2 * h_val) ** 2
        den = 2 * (b_val + 2 * h_val) ** 2
        arg = 1.0 - num / den
        if arg < -1 or arg > 1:
            return None
        return float(np.arccos(arg))

    width_stow_actual = float(value(2 * model.b[1] + 4 * model.hinge_par))
    height_deploy_actual = float(value(model.height[n_states]))
    depth_deploy_actual = float(value(model.length[n_states]))

    out = {
        "inputs": {
            "stow_width": stow_width,
            "stow_depth": stow_depth,
            "expanded_depth": expanded_depth,
            "offset": float(offset) if offset is not None else None,
            "thickness": float(thickness) if thickness is not None else None,
        },
        "solution": {
            "long": float(value(model.long)),
            "short": float(value(model.short)),
            "offset": float(value(model.offset)),
            "hinge_par": float(value(model.hinge_par)),
            "n_cells": int(round(value(model.n))),
            "alpha_stow": float(angle(model.sa[1], model.ca[1])),
            "alpha_deploy": float(angle(model.sa[n_states], model.ca[n_states])),
            "beta_stow": float(angle(model.sb[1], model.cb[1])),
            "beta_deploy": float(angle(model.sb[n_states], model.cb[n_states])),
            "theta_deploy": theta(n_states),

            "a_stow": float(value(model.a[1])),
            "b_stow": float(value(model.b[1])),
            "c_stow": float(value(model.c_long[1])),
            "a_deploy": float(value(model.a[n_states])),
            "b_deploy": float(value(model.b[n_states])),
            "c_deploy": float(value(model.c_long[n_states])),

            "width_stow": float(value(2 * model.a[1] + 2 * model.hinge_par)),
            "depth_stow": float(value(model.length[1])),
            "depth_deploy": float(value(model.length[n_states])),
            "thickness": float(value(model.thickness)),

            "extension_ratio_length": float(value(model.length[n_states]) / value(model.thickness)),
            "extension_ratio_height": float(value(model.height[n_states]) / value(model.thickness * 8)),

            "height_stow": float(value(model.thickness * 8)),
            "height": float(value(model.height[n_states])),
            "length": float(value(model.length[n_states])),

            "width_actual": width_stow_actual,
            "height_actual": height_deploy_actual,
            "depth_actual": depth_deploy_actual,

            "objective": float(value(model.objective)),
        },
        "status": str(term),
    }
    return out


def parse_thicknesses(args):
    if args.thickness_list:
        vals = [float(x) for x in args.thickness_list.split(",") if x.strip()]
    else:
        vals = list(np.linspace(args.thickness_min, args.thickness_max, args.thickness_steps))

    vals = [quantize_thickness(v) for v in vals]
    vals = sorted(set(vals))
    return vals


def compute_rounding_residuals(raw_result):
    """
    Round long, short, offset, hinge_par to 0.0001 m (0.1 mm), then evaluate
    every kinematic constraint residual using the ORIGINAL solver angles.
    No re-solve — just arithmetic. Reports how much each constraint is
    violated by the rounding so you can judge if it's acceptable.

    Returns a dict with rounded values, per-constraint residuals (in meters),
    and recomputed dimensions.
    """
    sol = raw_result["solution"]
    inp = raw_result["inputs"]

    # Original values
    L = sol["long"]
    S = sol["short"]
    O = sol["offset"]
    H = sol["hinge_par"]
    T = sol["thickness"]
    n = sol["n_cells"]

    # Rounded values
    rL = quantize_length(L)
    rS = quantize_length(S)
    rO = quantize_length(O)
    rH = quantize_length(H)

    # Original angles (from solver, these are exact)
    alpha_stow   = sol["alpha_stow"]
    alpha_deploy = sol["alpha_deploy"]
    ha_s = alpha_stow / 2
    ha_d = alpha_deploy / 2
    sa1, ca1 = math.sin(ha_s), math.cos(ha_s)
    sa2, ca2 = math.sin(ha_d), math.cos(ha_d)

    beta_stow   = sol.get("beta_stow", 0)
    beta_deploy = sol.get("beta_deploy", 0)
    hb_s = beta_stow / 2
    hb_d = beta_deploy / 2
    sb1, cb1 = math.sin(hb_s), math.cos(hb_s)
    sb2, cb2 = math.sin(hb_d), math.cos(hb_d)

    # ── Recompute expressions with rounded values ──
    a1 = rL * ca1;  a2 = rL * ca2
    b1 = rS * cb1;  b2 = rS * cb2
    c_long1 = 2 * (rL - rO) * sa1;  c_long2 = 2 * (rL - rO) * sa2
    c_short1 = 2 * rS * sb1;        c_short2 = 2 * rS * sb2
    length1 = c_long1 * n;           length2 = c_long2 * n
    width1 = 2 * b1 + 4 * rH

    # cos_theta and height at deploy
    num2 = (2 * a2 + 2 * rH) ** 2
    den2 = 2 * (b2 + 2 * rH) ** 2
    cos_theta2 = 1.0 - num2 / den2 if den2 > 1e-14 else 1.0
    height2 = (b2 + 2 * rH) * cos_theta2

    # ── Constraint residuals (LHS - RHS, so 0 = perfect) ──
    residuals = {}

    # cell_depth_match: c_long == c_short (for each state)
    residuals["cell_depth_match_stow"]   = c_long1 - c_short1
    residuals["cell_depth_match_deploy"] = c_long2 - c_short2

    # initial_state: a[1] == b[1] + hinge_par
    residuals["initial_state"] = a1 - (b1 + rH)

    # final_state: 2*a[2] == b[2]
    residuals["final_state"] = 2 * a2 - b2

    # initial_angle_beta: sb[1] == thickness / (2 * short)
    residuals["initial_angle_beta"] = sb1 - T / (2 * rS)

    # stow_width: 2*b[1] + 4*hinge_par == stow_width
    stow_width_target = inp.get("stow_width")
    if stow_width_target is not None:
        residuals["stow_width"] = width1 - stow_width_target

    # stow_depth: length[1] == stow_depth
    stow_depth_target = inp.get("stow_depth")
    if stow_depth_target is not None:
        residuals["stow_depth"] = length1 - stow_depth_target

    # expanded_depth: length[2] == expanded_depth
    expanded_depth_target = inp.get("expanded_depth")
    if expanded_depth_target is not None:
        residuals["expanded_depth"] = length2 - expanded_depth_target

    # Max absolute residual
    max_err = max(abs(v) for v in residuals.values()) if residuals else 0.0

    return {
        "long": rL, "short": rS, "offset": rO,
        "hinge_par": rH, "thickness": T, "n_cells": n,
        "status": "ok" if max_err < 1e-3 else "check_residuals",
        "max_residual_m": max_err,
        "max_residual_mm": max_err * 1000,
        "residuals": {k: float(v) for k, v in residuals.items()},
        "residuals_mm": {k: float(v * 1000) for k, v in residuals.items()},
        # Recomputed dimensions with rounded values
        "a_stow": a1, "b_stow": b1, "c_stow": c_long1,
        "a_deploy": a2, "b_deploy": b2, "c_deploy": c_long2,
        "width_stow": width1,
        "depth_stow": length1,
        "depth_deploy": length2,
        "height": height2,
        "length": length2,
        "width_actual": width1,
        "height_actual": height2,
        "depth_actual": length2,
        # Keep original angles
        "alpha_stow": alpha_stow, "alpha_deploy": alpha_deploy,
        "beta_stow": beta_stow, "beta_deploy": beta_deploy,
    }


def sweep_offsets_and_thicknesses(args):
    offsets = list(np.linspace(args.offset_min, args.offset_max, args.offset_steps))
    thicknesses = parse_thicknesses(args)

    feasible = []
    for t in thicknesses:
        for off in offsets:
            try:
                r = solve_one(
                    args.stow_width, args.stow_depth, args.expanded_depth,
                    args.states, args.hinge_par, float(t), args.n,
                    args.short, args.long, float(off),
                    tee=args.tee, debug_infeasible=False
                )
                # Compute rounding residuals (no re-solve, just arithmetic)
                rounded = compute_rounding_residuals(r)
                r["rounded_solution"] = rounded
                feasible.append(r)
            except Exception:
                continue

    # Summary
    if feasible:
        max_errs = [f["rounded_solution"]["max_residual_mm"] for f in feasible]
        print(f"[sweep] Rounding residuals across {len(feasible)} solutions: "
              f"max={max(max_errs):.6f} mm, "
              f"median={sorted(max_errs)[len(max_errs)//2]:.6f} mm, "
              f"min={min(max_errs):.6f} mm")
    return feasible


# ════════════════════════════════════════════════════════════════════════════════
# THREE.JS 3D VIEWER (embedded as JS string)
# ════════════════════════════════════════════════════════════════════════════════

THREE_JS_VIEWER = r"""
// ─── 3D Scissor Linkage Viewer ───────────────────────────────────────────────
// Expects: window.viewerLoadSolution(solObj) where solObj = { inputs, solution }

(function() {
  const DEG = Math.PI / 180;
  const RAD = 180 / Math.PI;

  function compute2D(L, O, alpha) {
    const ha = alpha / 2;
    const ca = Math.cos(ha), sa = Math.sin(ha);
    const LmO = L - O;
    const P1x = L * ca, P1y = L * sa;
    const P2x = LmO * ca, P2y = -LmO * sa;
    const c = 2 * LmO * sa;
    const P3x = P1x, P3y = P1y - c;
    return { P1x, P1y, P2x, P2y, P3x, P3y, c, a: L * ca };
  }

  function addTube(grp, a, b, color, radius) {
    const dir = new THREE.Vector3().subVectors(b, a);
    const len = dir.length();
    if (len < 1e-5) return;
    const geo = new THREE.CylinderGeometry(radius, radius, len, 12);
    const mat = new THREE.MeshPhongMaterial({ color, shininess: 60 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.addVectors(a, b).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    grp.add(mesh);
  }

  function addBall(grp, pos, color, radius) {
    const geo = new THREE.SphereGeometry(radius, 12, 12);
    const mat = new THREE.MeshPhongMaterial({ color, shininess: 80 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    grp.add(mesh);
  }

  function shortApex(origin, pA, halfBeta, S, sign) {
    const cbh = Math.cos(halfBeta);
    const sbh = Math.sin(halfBeta);
    const tx = -Math.cos(pA);
    const tz = Math.sin(pA);
    return new THREE.Vector3(
      origin.x + S * cbh * tx,
      origin.y - sign * S * sbh,
      origin.z + S * cbh * tz
    );
  }

  function addShortDnHinge(grp, dnApex, pA, hingePar, hingePer, color, radius, Jrs) {
    const tx = -Math.cos(pA);
    const tz = Math.sin(pA);
    const nx = Math.sin(pA);
    const nz = Math.cos(pA);
    const hParEnd = new THREE.Vector3(
      dnApex.x + hingePar * tx, dnApex.y, dnApex.z + hingePar * tz
    );
    const hPerEnd = new THREE.Vector3(
      hParEnd.x + hingePer * nx, hParEnd.y, hParEnd.z + hingePer * nz
    );
    addTube(grp, dnApex, hParEnd, color, radius);
    addTube(grp, hParEnd, hPerEnd, color, radius);
    addBall(grp, hParEnd, color, Jrs);
    addBall(grp, hPerEnd, color, Jrs);
  }

  function addRefFrame(grp, pos, size, T) {
    const r = T * 0.4;
    addTube(grp, pos, new THREE.Vector3(pos.x + size, pos.y, pos.z), 0xff0000, r);
    addTube(grp, pos, new THREE.Vector3(pos.x, pos.y + size, pos.z), 0x00ff00, r);
    addTube(grp, pos, new THREE.Vector3(pos.x, pos.y, pos.z + size), 0x0000ff, r);
  }

  function addRotatedRefFrame(grp, pos, size, T, angle) {
    const r = T * 0.4;
    const ca = Math.cos(angle), sa = Math.sin(angle);
    addTube(grp, pos, new THREE.Vector3(pos.x + size * ca, pos.y, pos.z - size * sa), 0xff0000, r);
    addTube(grp, pos, new THREE.Vector3(pos.x, pos.y + size, pos.z), 0x00ff00, r);
    addTube(grp, pos, new THREE.Vector3(pos.x + size * sa, pos.y, pos.z + size * ca), 0x0000ff, r);
  }

  function buildScene(L, O, alpha, nCells, T, hingePar, hingePer, S, h, showCells) {
    const root = new THREE.Group();
    const { P1x, P1y, P2x, P2y, P3x, P3y, c, a } = compute2D(L, O, alpha);

    const sb = S > 1e-6 ? Math.min(1, Math.max(0, c / (2 * S))) : 0;
    const halfBeta = Math.asin(sb);
    const cb = Math.cos(halfBeta);
    const b = S * cb;
    const num = (2 * a + 2 * h) ** 2;
    const den = 2 * (b + 2 * h) ** 2;
    let cosTheta = den > 1e-14 ? 1 - num / den : 1;
    cosTheta = Math.max(-1, Math.min(1, cosTheta));
    const theta = Math.acos(cosTheta);

    const planeAngle = (Math.PI - theta) / 2;
    const leftPlaneAngle = (Math.PI + theta) / 2;

    const cellsToShow = Math.min(showCells, nCells);
    const Toff = T * 0.75;
    const Jr = T * 1.4;
    const Jrs = T * 1.1;
    const refSize = 0.08;

    const COL_LONG = 0x2255cc;
    const COL_OFF = 0xc4164a;
    const COL_HINGE = 0x999999;
    const COL_JOINT = 0x444444;
    const COL_SHORT_HINGE = 0x44bb44;
    const COL_SHORT = 0x1d8c36;
    const COL_LONGERON = 0xcc2255;

    for (let i = 0; i < cellsToShow; i++) {
      const g = new THREE.Group();
      g.position.set(0, -i * c, 0);

      // ── RIGHT HALF (+X) ──
      const rOup = new THREE.Vector3(0, 0, +T);
      const rOdn = new THREE.Vector3(0, 0, -T);
      const rP1 = new THREE.Vector3(P1x, P1y, +T);
      const rP2 = new THREE.Vector3(P2x, P2y, -T);

      addTube(g, rOup, rP1, COL_LONG, T);
      addTube(g, rOdn, rP2, COL_LONG, T);

      const rP1_up = new THREE.Vector3(P1x, P1y, +2*T);
      addBall(g, rP1_up, COL_HINGE, Jrs);
      const rH1 = new THREE.Vector3(P1x + hingePar, P1y, +2*T);
      const rH2 = new THREE.Vector3(P1x + hingePar, P1y, 2*T+hingePer);
      addTube(g, rP1_up, rH1, COL_HINGE, Toff);
      addTube(g, rH1, rH2, COL_HINGE, Toff);

      const rSH = new THREE.Vector3(
        rH2.x + hingePer * Math.sin(planeAngle), rH2.y,
        rH2.z + hingePer * Math.cos(planeAngle)
      );
      addTube(g, rH2, rSH, COL_SHORT_HINGE, Toff);

      const rSH_par = new THREE.Vector3(
        rSH.x - hingePar * Math.cos(planeAngle), rSH.y,
        rSH.z + hingePar * Math.sin(planeAngle)
      );
      addTube(g, rSH, rSH_par, COL_SHORT_HINGE, Toff);
      addBall(g, rSH, COL_SHORT_HINGE, Jrs);
      addBall(g, rSH_par, COL_SHORT_HINGE, Jrs);

      addRefFrame(g, rH2, refSize, T);
      addRotatedRefFrame(g, rSH_par, refSize, T, planeAngle);

      const rSH_par_up = new THREE.Vector3(rSH_par.x, rSH_par.y, rSH_par.z + T);
      const rShortDn = shortApex(rSH_par_up, planeAngle, halfBeta, S, -1);
      const rSH_par_up_up = new THREE.Vector3(rSH_par_up.x, rSH_par_up.y, rSH_par_up.z + T);
      const rShortUp = shortApex(rSH_par_up_up, planeAngle, halfBeta, S, +1);
      addTube(g, rSH_par_up_up, rShortUp, COL_SHORT, Toff);
      addBall(g, rShortUp, COL_SHORT, Jrs);
      addBall(g, rSH_par_up_up, COL_SHORT, Jrs);

      if (i > 0) {
        addTube(g, rSH_par_up, rShortDn, COL_SHORT, Toff);
        addBall(g, rSH_par_up, COL_SHORT, Jrs);
        addBall(g, rShortDn, COL_SHORT, Jrs);
        const rShortDn_dn = new THREE.Vector3(rShortDn.x, rShortDn.y, rShortDn.z - T);
        addShortDnHinge(g, rShortDn_dn, planeAngle, hingePar, -hingePer, COL_SHORT_HINGE, Toff, Jrs);
        addBall(g, rShortDn_dn, COL_SHORT_HINGE, Jrs);
      }

      if (i === cellsToShow - 1) {
        const rP3 = new THREE.Vector3(P3x, P3y, +T);
        const rP3_up = new THREE.Vector3(P3x, P3y, +2*T);
        const rP2_up = new THREE.Vector3(P2x, P2y, +T);
        addTube(g, rP2_up, rP3, COL_OFF, Toff);
        const rH3 = new THREE.Vector3(P3x + hingePar, P3y, 2*T);
        addTube(g, rP3_up, rH3, COL_HINGE, Toff);
        const rH3up = new THREE.Vector3(P3x + hingePar, P3y, 2*T + hingePer);
        addTube(g, rH3, rH3up, COL_HINGE, Toff);
        const rSH3 = new THREE.Vector3(
          rH3up.x + hingePer * Math.sin(planeAngle), rH3up.y,
          rH3up.z + hingePer * Math.cos(planeAngle)
        );
        addTube(g, rH3up, rSH3, COL_SHORT_HINGE, Toff);
        const rSH3_par = new THREE.Vector3(
          rSH3.x - hingePar * Math.cos(planeAngle), rSH3.y,
          rSH3.z + hingePar * Math.sin(planeAngle)
        );
        addTube(g, rSH3, rSH3_par, COL_SHORT_HINGE, Toff);
        addRefFrame(g, rH3up, refSize, T);
        addRotatedRefFrame(g, rSH3_par, refSize, T, planeAngle);
        const rSH3_par_up = new THREE.Vector3(rSH3_par.x, rSH3_par.y, rSH3_par.z + T);
        const rSh3Dn = shortApex(rSH3_par_up, planeAngle, halfBeta, S, -1);
        addTube(g, rSH3_par_up, rSh3Dn, COL_SHORT, Toff);
        addBall(g, rSH3_par_up, COL_SHORT, Jrs);
        addBall(g, rSh3Dn, COL_SHORT, Jrs);
        const rSh3Dn_up = new THREE.Vector3(rSh3Dn.x, rSh3Dn.y, rSh3Dn.z - T);
        addShortDnHinge(g, rSh3Dn_up, planeAngle, hingePar, -hingePer, COL_SHORT_HINGE, Toff, Jrs);
        addBall(g, rSh3Dn_up, COL_SHORT_HINGE, Jrs);
        addBall(g, rP2_up, COL_OFF, Jrs);
        addBall(g, rP3_up, COL_HINGE, Jrs);
        addBall(g, rP3, COL_OFF, Jrs);
        addBall(g, rH3, COL_HINGE, Jrs);
        addBall(g, rH3up, COL_HINGE, Jrs);
        addBall(g, rSH3, COL_SHORT_HINGE, Jrs);
        addBall(g, rSH3_par, COL_SHORT_HINGE, Jrs);
      }

      addBall(g, rOup, COL_JOINT, Jr);
      addBall(g, rOdn, COL_JOINT, Jr);
      addBall(g, rP1, COL_LONG, Jrs);
      addBall(g, rP2, COL_LONG, Jrs);
      addBall(g, rH1, COL_HINGE, Jrs);
      addBall(g, rH2, COL_HINGE, Jrs);

      // ── Longerons (right) ──
      const longeron_length = 0.0625;
      const longeron_halfAngle = Math.asin(c/(2*longeron_length));
      const rP1_offset = new THREE.Vector3(P1x, P1y, -2*T);
      const rP3r = new THREE.Vector3(P3x, P3y, -T);
      const rP3_offset = new THREE.Vector3(P3x, P3y, -3*T);
      const rP1_Longeron = new THREE.Vector3(
        rP1_offset.x - Math.cos(longeron_halfAngle)*longeron_length,
        rP1_offset.y - Math.sin(longeron_halfAngle)*longeron_length,
        rP1_offset.z - T
      );
      const rP3_Longeron = new THREE.Vector3(
        rP3_offset.x - Math.cos(longeron_halfAngle)*longeron_length,
        rP3_offset.y + Math.sin(longeron_halfAngle)*longeron_length,
        rP3_offset.z - T
      );
      addTube(g, rP1_offset, rP1_Longeron, COL_LONGERON, Toff);
      addTube(g, rP3_offset, rP3_Longeron, COL_LONGERON, Toff);
      addTube(g, rP1, rP1_offset, COL_HINGE, Jrs);
      addTube(g, rP3r, rP3_offset, COL_HINGE, Jrs);
      addBall(g, rP1_Longeron, COL_LONGERON, Jrs);
      addBall(g, rP3_Longeron, COL_LONGERON, Jrs);

      if (i > 0 && i % 2 === 0) {
        const rLongUp_ref = new THREE.Vector3(rShortUp.x, rShortUp.y, rShortUp.z +2*T);
        addTube(g, rShortUp, rLongUp_ref, COL_HINGE, Toff);
        const rLongUp = new THREE.Vector3(
          rShortUp.x + Math.cos(longeron_halfAngle)*longeron_length,
          rShortUp.y + Math.sin(longeron_halfAngle)*longeron_length,
          rShortUp.z +2*T
        );
        addTube(g, rLongUp_ref, rLongUp, COL_LONGERON, Toff);
        addBall(g, rLongUp_ref, COL_LONGERON, Jrs);
        addBall(g, rLongUp, COL_LONGERON, Jrs);

        const rLongDn_ref = new THREE.Vector3(rShortDn.x, rShortDn.y, rShortDn.z +2*T);
        const rLongDn = new THREE.Vector3(
          rShortDn.x + Math.cos(longeron_halfAngle)*longeron_length,
          rShortDn.y - Math.sin(longeron_halfAngle)*longeron_length,
          rShortDn.z +2*T
        );
        addTube(g, rLongDn_ref, rLongDn, COL_LONGERON, Toff);
        addBall(g, rLongDn_ref, COL_LONGERON, Jrs);
        addBall(g, rLongDn, COL_LONGERON, Jrs);
      }

      // ── LEFT HALF (-X) ──
      const lOup = new THREE.Vector3(0, 0, -T);
      const lOdn = new THREE.Vector3(0, 0, +T);
      const lP1 = new THREE.Vector3(-P1x, P1y, -T);
      const lP2 = new THREE.Vector3(-P2x, P2y, +T);

      addTube(g, lOup, lP1, COL_LONG, Toff);
      addTube(g, lOdn, lP2, COL_LONG, Toff);

      const spacer = new THREE.Vector3(-P1x, P1y, 2*T);
      addTube(g, lP1, spacer, COL_HINGE, Toff);
      const lH1 = new THREE.Vector3(spacer.x - hingePar, spacer.y, spacer.z);
      const lH2 = new THREE.Vector3(lH1.x, lH1.y, lH1.z + hingePer);
      addTube(g, spacer, lH1, COL_HINGE, Toff);
      addTube(g, lH1, lH2, COL_HINGE, Toff);
      addBall(g, spacer, COL_HINGE, Jrs);

      const lSH = new THREE.Vector3(
        lH2.x - hingePer * Math.sin(leftPlaneAngle), lH2.y,
        lH2.z - hingePer * Math.cos(leftPlaneAngle)
      );
      addTube(g, lH2, lSH, COL_SHORT_HINGE, Toff);
      const lSH_par = new THREE.Vector3(
        lSH.x - hingePar * Math.cos(leftPlaneAngle), lSH.y,
        lSH.z + hingePar * Math.sin(leftPlaneAngle)
      );
      addTube(g, lSH, lSH_par, COL_SHORT_HINGE, Toff);
      addBall(g, lSH, COL_SHORT_HINGE, Jrs);
      addBall(g, lSH_par, COL_SHORT_HINGE, Jrs);

      addRefFrame(g, lH2, refSize, T);
      addRotatedRefFrame(g, lSH_par, refSize, T, -leftPlaneAngle);

      const lSH_par_up = new THREE.Vector3(lSH_par.x, lSH_par.y, lSH_par.z + T);
      const lShortDn = shortApex(lSH_par_up, leftPlaneAngle, halfBeta, S, -1);
      const lSH_par_up_up = new THREE.Vector3(lSH_par_up.x, lSH_par_up.y, lSH_par_up.z + T);
      const lShortUp = shortApex(lSH_par_up_up, leftPlaneAngle, halfBeta, S, +1);
      addTube(g, lSH_par_up_up, lShortUp, COL_SHORT, Toff);
      addBall(g, lShortUp, COL_SHORT, Jrs);
      addBall(g, lSH_par_up_up, COL_SHORT, Jrs);

      if (i > 0) {
        addTube(g, lSH_par_up, lShortDn, COL_SHORT, Toff);
        addBall(g, lSH_par_up, COL_SHORT, Jrs);
        const lShortDn_dn = new THREE.Vector3(lShortDn.x, lShortDn.y, lShortDn.z - T);
        addShortDnHinge(g, lShortDn_dn, leftPlaneAngle, hingePar, hingePer, COL_SHORT_HINGE, Toff, Jrs);
        addBall(g, lShortDn_dn, COL_SHORT_HINGE, Jrs);
        addBall(g, lShortDn, COL_SHORT, Jrs);
      }

      if (i === cellsToShow - 1) {
        const lP3 = new THREE.Vector3(-P3x, P3y, -T);
        const lP2_dn = new THREE.Vector3(-P2x, P2y, -T);
        addBall(g, lP2_dn, COL_OFF, Jrs);
        addTube(g, lP2_dn, lP3, COL_OFF, Toff);
        const lH3 = new THREE.Vector3(-P3x - hingePar, P3y, +2*T);
        const lP3_up = new THREE.Vector3(-P3x, P3y, +2*T);
        addTube(g, lP3_up, lH3, COL_HINGE, Toff);
        addTube(g, lP3, lP3_up, COL_HINGE, Toff);
        const lH3up = new THREE.Vector3(-P3x - hingePar, P3y, 2*T + hingePer);
        addTube(g, lH3, lH3up, COL_HINGE, Toff);
        const lSH3 = new THREE.Vector3(
          lH3up.x - hingePer * Math.sin(leftPlaneAngle), lH3up.y,
          lH3up.z - hingePer * Math.cos(leftPlaneAngle)
        );
        addTube(g, lH3up, lSH3, COL_SHORT_HINGE, Toff);
        const lSH3_par = new THREE.Vector3(
          lSH3.x - hingePar * Math.cos(leftPlaneAngle), lSH3.y,
          lSH3.z + hingePar * Math.sin(leftPlaneAngle)
        );
        addTube(g, lSH3, lSH3_par, COL_SHORT_HINGE, Toff);
        addRefFrame(g, lH3up, refSize, T);
        addRotatedRefFrame(g, lSH3_par, refSize, T, -leftPlaneAngle);
        const lSH3_par_up = new THREE.Vector3(lSH3_par.x, lSH3_par.y, lSH3_par.z + T);
        const lSh3Dn = shortApex(lSH3_par_up, leftPlaneAngle, halfBeta, S, -1);
        addTube(g, lSH3_par_up, lSh3Dn, COL_SHORT, Toff);
        addBall(g, lSH3_par_up, COL_SHORT, Jrs);
        addBall(g, lSh3Dn, COL_SHORT, Jrs);
        const lSh3Dn_up = new THREE.Vector3(lSh3Dn.x, lSh3Dn.y, lSh3Dn.z - T);
        addShortDnHinge(g, lSh3Dn_up, leftPlaneAngle, hingePar, hingePer, COL_SHORT_HINGE, Toff, Jrs);
        addBall(g, lSh3Dn_up, COL_SHORT_HINGE, Jrs);
        addBall(g, lP3_up, COL_HINGE, Jrs);
        addBall(g, lP3, COL_OFF, Jrs);
        addBall(g, lH3, COL_HINGE, Jrs);
        addBall(g, lH3up, COL_HINGE, Jrs);
        addBall(g, lSH3, COL_SHORT_HINGE, Jrs);
        addBall(g, lSH3_par, COL_SHORT_HINGE, Jrs);
      }

      addBall(g, lP1, COL_LONG, Jrs);
      addBall(g, lP2, COL_LONG, Jrs);
      addBall(g, lH1, COL_HINGE, Jrs);
      addBall(g, lH2, COL_HINGE, Jrs);

      // Longerons (left)
      const longeron_length_l = 0.0625;
      const longeron_halfAngle_l = Math.asin(c/(2*longeron_length_l));
      const lP1_offset = new THREE.Vector3(-P1x, P1y, -2*T);
      const lP3l = new THREE.Vector3(-P3x, P3y, -T);
      const lP3_offset = new THREE.Vector3(-P3x, P3y, -3*T);
      const lP1_Longeron = new THREE.Vector3(
        lP1_offset.x + Math.cos(longeron_halfAngle_l)*longeron_length_l,
        lP1_offset.y - Math.sin(longeron_halfAngle_l)*longeron_length_l,
        lP1_offset.z - T
      );
      const lP3_Longeron = new THREE.Vector3(
        lP3_offset.x + Math.cos(longeron_halfAngle_l)*longeron_length_l,
        lP3_offset.y + Math.sin(longeron_halfAngle_l)*longeron_length_l,
        lP3_offset.z - T
      );
      addTube(g, lP1_offset, lP1_Longeron, COL_LONGERON, Toff);
      addTube(g, lP3_offset, lP3_Longeron, COL_LONGERON, Toff);
      addTube(g, lP1_offset, lP1, COL_HINGE, Jrs);
      addTube(g, lP3l, lP3_offset, COL_HINGE, Jrs);
      addBall(g, lP1_Longeron, COL_LONGERON, Jrs);
      addBall(g, lP3_Longeron, COL_LONGERON, Jrs);

      if (i > 0 && i % 2 === 1) {
        const lLongUp_ref = new THREE.Vector3(lShortUp.x, lShortUp.y, lShortUp.z +2*T);
        addTube(g, lShortUp, lLongUp_ref, COL_HINGE, Toff);
        const lLongUp = new THREE.Vector3(
          lShortUp.x - Math.cos(longeron_halfAngle_l)*longeron_length_l,
          lShortUp.y + Math.sin(longeron_halfAngle_l)*longeron_length_l,
          lShortUp.z +2*T
        );
        addTube(g, lLongUp_ref, lLongUp, COL_LONGERON, Toff);
        addBall(g, lLongUp_ref, COL_LONGERON, Jrs);
        addBall(g, lLongUp, COL_LONGERON, Jrs);

        const lLongDn_ref = new THREE.Vector3(lShortDn.x, lShortDn.y, lShortDn.z +2*T);
        const lLongDn = new THREE.Vector3(
          lShortDn.x - Math.cos(longeron_halfAngle_l)*longeron_length_l,
          lShortDn.y - Math.sin(longeron_halfAngle_l)*longeron_length_l,
          lShortDn.z +2*T
        );
        addTube(g, lLongDn_ref, lLongDn, COL_LONGERON, Toff);
        addBall(g, lLongDn_ref, COL_LONGERON, Jrs);
        addBall(g, lLongDn, COL_LONGERON, Jrs);
      }

      root.add(g);
    }

    // Ground plane
    const sz = Math.max(L * 4, c * cellsToShow * 1.2, 0.5);
    const plGeo = new THREE.PlaneGeometry(sz, sz);
    const plMat = new THREE.MeshBasicMaterial({ color: 0x4488ff, transparent: true, opacity: 0.015, side: THREE.DoubleSide });
    const pl = new THREE.Mesh(plGeo, plMat);
    pl.position.set(0, -c * Math.max(0, cellsToShow - 1) / 2, 0);
    root.add(pl);

    return { group: root, c, theta, halfBeta, cellsToShow };
  }

  // ── Viewer state ──
  let viewerState = null;

  function disposeGroup(group) {
    group.traverse(ch => {
      if (ch.geometry) ch.geometry.dispose();
      if (ch.material) ch.material.dispose();
    });
  }

  function initViewer(container) {
    const W = container.clientWidth, H = container.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x090b10, 1);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const d1 = new THREE.DirectionalLight(0xffffff, 0.7);
    d1.position.set(5, 8, 7); scene.add(d1);
    const d2 = new THREE.DirectionalLight(0x6688cc, 0.3);
    d2.position.set(-5, -4, 6); scene.add(d2);
    scene.add(new THREE.AxesHelper(0.3));

    const camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 200);
    camera.userData.tgt = new THREE.Vector3(0, 0, 0);

    const mouse = { down: false, btn: -1, lx: 0, ly: 0, rx: -0.3, ry: 0.85, d: 3, px: 0, py: 0 };

    const el = renderer.domElement;
    el.addEventListener("mousedown", e => { mouse.down = true; mouse.btn = e.button; mouse.lx = e.clientX; mouse.ly = e.clientY; });
    window.addEventListener("mouseup", () => { mouse.down = false; });
    el.addEventListener("mousemove", e => {
      if (!mouse.down) return;
      const dx = e.clientX - mouse.lx, dy = e.clientY - mouse.ly;
      if (mouse.btn === 0) { mouse.ry += dx * 0.005; mouse.rx = Math.max(-1.5, Math.min(1.5, mouse.rx + dy * 0.005)); }
      else if (mouse.btn === 2) { mouse.px -= dx * 0.005; mouse.py += dy * 0.005; }
      mouse.lx = e.clientX; mouse.ly = e.clientY;
    });
    el.addEventListener("wheel", e => { mouse.d = Math.max(0.05, Math.min(50, mouse.d + e.deltaY * 0.002)); });
    el.addEventListener("contextmenu", e => e.preventDefault());

    let frame = null;
    function animate() {
      frame = requestAnimationFrame(animate);
      const t = camera.userData.tgt || new THREE.Vector3();
      camera.position.set(
        t.x + mouse.px + mouse.d * Math.sin(mouse.ry) * Math.cos(mouse.rx),
        t.y + mouse.py + mouse.d * Math.sin(mouse.rx),
        t.z + mouse.d * Math.cos(mouse.ry) * Math.cos(mouse.rx)
      );
      camera.lookAt(t.x + mouse.px, t.y + mouse.py, t.z);
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener("resize", () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });

    viewerState = { scene, renderer, camera, mouse, linkage: null, frame, container };
  }

  function rebuildLinkage(sol, alphaFrac, showCells, thickness, hingePer) {
    if (!viewerState) return;
    const s = viewerState;
    if (s.linkage) { s.scene.remove(s.linkage); disposeGroup(s.linkage); }

    const alpha = (sol.alpha_stow || 0) + alphaFrac * ((sol.alpha_deploy || 1) - (sol.alpha_stow || 0));
    const { group, c } = buildScene(
      sol.long, sol.offset, alpha, sol.n_cells,
      thickness, sol.hinge_par, hingePer,
      sol.short, sol.hinge_par, showCells
    );
    s.linkage = group;
    s.scene.add(group);
    s.camera.userData.tgt = new THREE.Vector3(0, -c * Math.max(0, Math.min(showCells, sol.n_cells) - 1) / 2, 0);
  }

  // ── Public API ──
  window.viewerInit = initViewer;
  window.viewerRebuild = rebuildLinkage;
})();
"""


def make_interactive_plot(results, out_html="scissor_sweep.html",
                          actual_axes=("depth_actual", "height_actual"),
                          initial_mode="ratio"):
    """
    Interactive Plotly scatter with:
      - Toggle: Extension ratios vs Actual axes
      - Hover: thickness + offset + key vars
      - Click: loads the 3D Three.js viewer with the selected solution

    KEY FIX: customdata uses list-of-lists (not np.stack) to preserve
    mixed types (floats + JSON strings).
    """
    if pd is None or go is None or pio is None:
        raise RuntimeError(
            "plotly and pandas are required for interactive plots.\n"
            "Install with: pip install plotly pandas"
        )

    axis_label = {
        "er_len": "Extension ratio (length) [–]",
        "er_hgt": "Extension ratio (height) [–]",
        "depth_actual": "Deployed depth [m]",
        "height_actual": "Deployed height [m]",
        "width_actual": "Stowed width [m]",
    }

    rows = []
    payloads = []
    for r in results:
        s = r["solution"]
        inp = r["inputs"]
        rows.append({
            "er_len": s["extension_ratio_length"],
            "er_hgt": s["extension_ratio_height"],
            "depth_actual": s["depth_actual"],
            "height_actual": s["height_actual"],
            "width_actual": s["width_actual"],
            "thickness": quantize_thickness(
                inp["thickness"] if inp["thickness"] is not None else s["thickness"]
            ),
            "offset": float(inp["offset"] if inp["offset"] is not None else s["offset"]),
            "n_cells": s["n_cells"],
            "long": s["long"],
            "short": s["short"],
            "hinge_par": s["hinge_par"],
        })
        payloads.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows to plot.")

    x_actual_key, y_actual_key = actual_axes

    def _finite(series):
        return np.isfinite(series.to_numpy(dtype=float))

    finite_ratio = _finite(df["er_len"]) & _finite(df["er_hgt"])
    finite_actual = _finite(df[x_actual_key]) & _finite(df[y_actual_key])
    keep = finite_ratio | finite_actual
    df = df[keep].copy()
    # Also filter payloads to match
    kept_indices = np.where(keep)[0]
    payloads = [payloads[i] for i in kept_indices]
    df = df.reset_index(drop=True)

    if df.empty:
        raise RuntimeError("All points were non-finite.")

    print(f"[plot] Plotting {len(df)} points (after filtering non-finite).")

    thickness_levels = sorted(df["thickness"].unique())
    fig = go.Figure()

    # ── FIX: Build customdata as list-of-lists to avoid numpy dtype coercion ──
    # customdata columns: [thickness, offset, n_cells, long, short, hinge_par, payload_index]
    # payload_index references into the global payloads array embedded in JS.

    def hovertemplate():
        return (
            "x=%{x:.4f}<br>"
            "y=%{y:.4f}<br>"
            "thickness=%{customdata[0]:.3f}<br>"
            "offset=%{customdata[1]:.5f}<br>"
            "n_cells=%{customdata[2]}<br>"
            "long=%{customdata[3]:.4f}<br>"
            "short=%{customdata[4]:.4f}<br>"
            "hinge=%{customdata[5]:.4f}<extra></extra>"
        )

    # 2 traces per thickness: ratio & actual
    for t in thickness_levels:
        mask = df["thickness"] == t
        dft = df[mask]
        original_indices = dft.index.tolist()

        # Build customdata as a plain Python list-of-lists
        cd = []
        for idx in original_indices:
            cd.append([
                float(dft.loc[idx, "thickness"]),
                float(dft.loc[idx, "offset"]),
                int(dft.loc[idx, "n_cells"]),
                float(dft.loc[idx, "long"]),
                float(dft.loc[idx, "short"]),
                float(dft.loc[idx, "hinge_par"]),
                int(idx),  # payload index
            ])

        fig.add_trace(go.Scatter(
            x=dft["er_len"].tolist(),
            y=dft["er_hgt"].tolist(),
            mode="markers",
            name=f"t={t:.3f}",
            customdata=cd,
            hovertemplate=hovertemplate(),
            visible=True if initial_mode == "ratio" else False,
            marker=dict(size=8),
        ))

        fig.add_trace(go.Scatter(
            x=dft[x_actual_key].tolist(),
            y=dft[y_actual_key].tolist(),
            mode="markers",
            name=f"t={t:.3f}",
            customdata=cd,
            hovertemplate=hovertemplate(),
            visible=True if initial_mode == "actual" else False,
            marker=dict(size=8),
        ))

    n_traces = len(fig.data)
    vis_ratio = [False] * n_traces
    vis_actual = [False] * n_traces
    for i in range(len(thickness_levels)):
        vis_ratio[2 * i] = True
        vis_actual[2 * i + 1] = True

    ratio_x_title = axis_label["er_len"]
    ratio_y_title = axis_label["er_hgt"]
    actual_x_title = axis_label.get(x_actual_key, x_actual_key)
    actual_y_title = axis_label.get(y_actual_key, y_actual_key)

    x_title = actual_x_title if initial_mode == "actual" else ratio_x_title
    y_title = actual_y_title if initial_mode == "actual" else ratio_y_title

    fig.update_layout(
        title="Scissor sweep — click a point to visualize in 3D",
        xaxis_title=x_title,
        yaxis_title=y_title,
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.0,
            y=1.15,
            buttons=[
                dict(
                    label="Extension ratios",
                    method="update",
                    args=[
                        {"visible": vis_ratio},
                        {"xaxis": {"title": ratio_x_title},
                         "yaxis": {"title": ratio_y_title}},
                    ],
                ),
                dict(
                    label=f"Actual ({actual_x_title} vs {actual_y_title})",
                    method="update",
                    args=[
                        {"visible": vis_actual},
                        {"xaxis": {"title": actual_x_title},
                         "yaxis": {"title": actual_y_title}},
                    ],
                ),
            ],
        )],
        legend_title="Thickness [m]",
        margin=dict(t=90),
    )

    fig_json = pio.to_json(fig)
    payloads_json = json.dumps(payloads)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scissor Linkage — Sweep + 3D Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #090b10;
      color: #ccc;
    }}
    #app {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 60vh 40vh;
      width: 100vw;
      height: 100vh;
    }}
    #plotArea {{
      grid-column: 1;
      grid-row: 1;
      border-right: 1px solid #1f222b;
      border-bottom: 1px solid #1f222b;
    }}
    #plotDiv {{ width: 100%; height: 100%; }}
    #viewerArea {{
      grid-column: 2;
      grid-row: 1 / 3;
      position: relative;
      border-bottom: 1px solid #1f222b;
    }}
    #viewerMount {{ width: 100%; height: 100%; }}
    #viewerOverlay {{
      position: absolute;
      top: 12px; left: 12px;
      background: rgba(9,11,16,0.85);
      border: 1px solid #1f222b;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 11px;
      max-width: 260px;
      z-index: 10;
      pointer-events: none;
    }}
    #viewerOverlay .label {{ color: #666; text-transform: uppercase; letter-spacing: 2px; font-size: 8px; margin-bottom: 4px; }}
    #viewerOverlay .kv {{ display: flex; justify-content: space-between; line-height: 1.6; }}
    #viewerOverlay .kv .k {{ color: #888; }}
    #viewerOverlay .kv .v {{ color: #ddd; font-variant-numeric: tabular-nums; }}
    #controlsArea {{
      grid-column: 1;
      grid-row: 2;
      padding: 16px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    #controlsArea .section {{
      background: #11131a;
      border: 1px solid #1c1e28;
      border-radius: 6px;
      padding: 10px 14px;
    }}
    #controlsArea .section h3 {{
      font-size: 8px;
      color: #555;
      text-transform: uppercase;
      letter-spacing: 2px;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .slider-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }}
    .slider-row label {{ font-size: 11px; color: #888; }}
    .slider-row span {{ font-size: 11px; color: #bbb; font-variant-numeric: tabular-nums; }}
    input[type=range] {{ width: 100%; accent-color: #c8b88a; height: 3px; }}
    #jsonPanel {{
      background: #0d0f14;
      border: 1px solid #1c1e28;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 10px;
      overflow: auto;
      max-height: 200px;
      white-space: pre;
      color: #999;
    }}
    #placeholder3d {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #333;
      font-size: 14px;
      text-align: center;
      pointer-events: none;
    }}
    .legend-row {{ font-size: 9px; line-height: 1.9; color: #666; }}
    .legend-row span.c {{ display: inline-block; width: 16px; text-align: center; }}
    #exportBtn {{
      width: 100%;
      padding: 10px 0;
      background: linear-gradient(135deg, #c8b88a, #a89868);
      color: #090b10;
      border: none;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      cursor: pointer;
      font-family: inherit;
      transition: opacity 0.2s;
    }}
    #exportBtn:hover {{ opacity: 0.85; }}
    #exportBtn:disabled {{ opacity: 0.3; cursor: not-allowed; }}
    #exportStatus {{
      font-size: 9px;
      color: #44bb44;
      margin-top: 6px;
      min-height: 14px;
      transition: opacity 0.3s;
    }}
    .csv-field {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }}
    .csv-field label {{
      font-size: 10px;
      color: #888;
    }}
    .csv-field .val {{
      font-size: 11px;
      color: #c8b88a;
      font-variant-numeric: tabular-nums;
    }}
    .csv-field input {{
      width: 70px;
      background: #090b10;
      border: 1px solid #282b36;
      border-radius: 3px;
      color: #c8b88a;
      font-size: 11px;
      font-family: inherit;
      padding: 3px 6px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .csv-field input:focus {{
      outline: none;
      border-color: #c8b88a;
    }}
    #csvTable th {{
      border-bottom: 1px solid #1f222b;
      font-weight: 600;
      font-size: 8px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    #csvTable td {{
      border-bottom: 1px solid #0d0f14;
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="plotArea"><div id="plotDiv"></div></div>
    <div id="viewerArea">
      <div id="viewerMount"></div>
      <div id="placeholder3d">Click a point in the plot<br>to load the 3D viewer</div>
      <div id="viewerOverlay" style="display:none;">
        <div class="label">Selected Solution</div>
        <div id="overlayContent"></div>
      </div>
    </div>
    <div id="controlsArea">
      <div class="section">
        <h3>Deployment</h3>
        <div class="slider-row">
          <label>α fraction</label>
          <span id="alphaVal">70%</span>
        </div>
        <input type="range" id="alphaSlider" min="0.001" max="1" step="0.002" value="0.7" />
        <div style="display:flex;justify-content:space-between;font-size:8px;color:#555;margin-top:2px;">
          <span>STOWED</span><span>DEPLOYED</span>
        </div>
      </div>
      <div class="section">
        <h3>Visualization</h3>
        <div class="slider-row">
          <label>Member thickness</label>
          <span id="thicknessVal">0.006</span>
        </div>
        <input type="range" id="thicknessSlider" min="0.001" max="0.025" step="0.0005" value="0.006" />
        <div class="slider-row">
          <label>Hinge height</label>
          <span id="hingePerVal">0.08</span>
        </div>
        <input type="range" id="hingePerSlider" min="0.001" max="0.03" step="0.001" value="0.008" />
        <div class="slider-row">
          <label>Show cells</label>
          <span id="cellsVal">2</span>
        </div>
        <input type="range" id="cellsSlider" min="1" max="20" step="1" value="2" />
      </div>
      <div class="section">
        <h3>Legend</h3>
        <div class="legend-row"><span class="c" style="color:#2255cc">━━</span> long arms</div>
        <div class="legend-row"><span class="c" style="color:#1d8c36">━━</span> short members</div>
        <div class="legend-row"><span class="c" style="color:#c4164a">━━</span> offset (last cell)</div>
        <div class="legend-row"><span class="c" style="color:#999">━━</span> hinge (grey)</div>
        <div class="legend-row"><span class="c" style="color:#44bb44">━━</span> short hinges</div>
        <div class="legend-row"><span class="c" style="color:#cc2255">━━</span> longerons</div>
        <div style="font-size:8px;color:#444;margin-top:6px;">L-drag: orbit · R-drag: pan · Scroll: zoom</div>
      </div>
      <div class="section">
        <h3>Export to CAD CSV (rounded + re-solved)</h3>
        <div id="roundedStatus" style="font-size:9px;margin-bottom:6px;min-height:14px;"></div>
        <table id="csvTable" style="width:100%;border-collapse:collapse;font-size:10px;">
          <thead>
            <tr style="color:#555;text-align:left;">
              <th style="padding:2px 4px;">Parameter</th>
              <th style="padding:2px 4px;text-align:right;">Solver (m)</th>
              <th style="padding:2px 4px;text-align:right;">Rounded (mm)</th>
            </tr>
          </thead>
          <tbody id="csvTableBody">
            <tr><td colspan="3" style="color:#444;padding:6px;">Click a point first</td></tr>
          </tbody>
        </table>
        <div style="margin-top:8px;">
          <div class="csv-field">
            <label>Hole_Diameter</label>
            <div><input type="number" id="csvHoleDiam" value="2" step="0.1" min="0.1" /> <span style="font-size:9px;color:#555">mm</span></div>
          </div>
          <div class="csv-field">
            <label>Hinge_Perp</label>
            <div><input type="number" id="csvHingePerp" value="2" step="0.1" min="0.1" /> <span style="font-size:9px;color:#555">mm</span></div>
          </div>
        </div>
        <button id="exportBtn" disabled>Export CSV</button>
        <div id="exportStatus"></div>
      </div>
      <div class="section">
        <h3>Full JSON output</h3>
        <pre id="jsonPanel">Click a point to populate.</pre>
      </div>
    </div>
  </div>

  <script>
{THREE_JS_VIEWER}
  </script>

  <script>
    // ── Data ──
    const ALL_PAYLOADS = {payloads_json};
    const fig = {fig_json};

    // ── State ──
    let currentSol = null;
    let viewerReady = false;

    // ── Plot ──
    Plotly.newPlot("plotDiv", fig.data, fig.layout, {{ responsive: true }});

    // ── Click handler ──
    document.getElementById("plotDiv").on("plotly_click", function(evt) {{
      const pt = evt.points[0];
      const payloadIdx = pt.customdata[6];
      const payload = ALL_PAYLOADS[payloadIdx];
      if (!payload) return;

      currentSol = payload.solution;
      currentPayload = payload;

      // Show JSON
      document.getElementById("jsonPanel").textContent = JSON.stringify(payload, null, 2);

      // Update overlay
      const ov = document.getElementById("viewerOverlay");
      ov.style.display = "block";
      document.getElementById("overlayContent").innerHTML =
        '<div class="kv"><span class="k">long</span><span class="v">' + currentSol.long.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">short</span><span class="v">' + currentSol.short.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">offset</span><span class="v">' + currentSol.offset.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">hinge_par</span><span class="v">' + currentSol.hinge_par.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">n_cells</span><span class="v">' + currentSol.n_cells + '</span></div>' +
        '<div class="kv"><span class="k">α deploy</span><span class="v">' + (currentSol.alpha_deploy * 180/Math.PI).toFixed(1) + '°</span></div>' +
        '<div class="kv"><span class="k">thickness</span><span class="v">' + currentSol.thickness.toFixed(3) + '</span></div>';

      // Update cells slider max
      document.getElementById("cellsSlider").max = currentSol.n_cells;
      if (parseInt(document.getElementById("cellsSlider").value) > currentSol.n_cells) {{
        document.getElementById("cellsSlider").value = Math.min(2, currentSol.n_cells);
      }}

      // Init 3D viewer if not yet
      if (!viewerReady) {{
        document.getElementById("placeholder3d").style.display = "none";
        window.viewerInit(document.getElementById("viewerMount"));
        viewerReady = true;
      }}

      rebuildFromUI();
      updateCsvPreview();
    }});

    function rebuildFromUI() {{
      if (!currentSol) return;
      const alphaFrac = parseFloat(document.getElementById("alphaSlider").value);
      const thickness = parseFloat(document.getElementById("thicknessSlider").value);
      const hingePer = parseFloat(document.getElementById("hingePerSlider").value);
      const showCells = parseInt(document.getElementById("cellsSlider").value);
      window.viewerRebuild(currentSol, alphaFrac, showCells, thickness, hingePer);
    }}

    // ── Sliders ──
    document.getElementById("alphaSlider").addEventListener("input", function() {{
      document.getElementById("alphaVal").textContent = (this.value * 100).toFixed(0) + "%";
      rebuildFromUI();
    }});
    document.getElementById("thicknessSlider").addEventListener("input", function() {{
      document.getElementById("thicknessVal").textContent = parseFloat(this.value).toFixed(4);
      rebuildFromUI();
    }});
    document.getElementById("cellsSlider").addEventListener("input", function() {{
      document.getElementById("cellsVal").textContent = this.value;
      rebuildFromUI();
    }});

    // ── CSV Export ──
    let currentPayload = null;
    let currentRounded = null;

    function updateCsvPreview() {{
      if (!currentPayload) return;
      const sol = currentPayload.solution;
      const rd = currentPayload.rounded_solution;
      currentRounded = rd;

      const statusEl = document.getElementById("roundedStatus");
      const tbody = document.getElementById("csvTableBody");
      const exportBtn = document.getElementById("exportBtn");

      if (!rd) {{
        statusEl.innerHTML = '<span style="color:#f87171;">No rounding data</span>';
        exportBtn.disabled = true;
        return;
      }}

      const maxErr = rd.max_residual_mm || 0;
      const res = rd.residuals_mm || {{}};

      if (maxErr < 0.01) {{
        statusEl.innerHTML = '<span style="color:#44bb44;">✓ Max rounding error: ' + maxErr.toFixed(6) + ' mm — negligible</span>';
      }} else if (maxErr < 0.1) {{
        statusEl.innerHTML = '<span style="color:#c8b88a;">⚠ Max rounding error: ' + maxErr.toFixed(4) + ' mm — small</span>';
      }} else {{
        statusEl.innerHTML = '<span style="color:#f87171;">⚠ Max rounding error: ' + maxErr.toFixed(4) + ' mm — review constraints below</span>';
      }}
      exportBtn.disabled = false;

      // ── Rounded values table ──
      let html = '<tr style="color:#555;"><td colspan="3" style="padding:4px 4px 2px;font-size:8px;text-transform:uppercase;letter-spacing:1px;">Rounded Dimensions</td></tr>';
      const params = [
        ["Plate_Thickness", sol.thickness, rd.thickness],
        ["Long_H2H",        sol.long,      rd.long],
        ["Short_H2H",       sol.short,     rd.short],
        ["Offset_H2H",      sol.offset,    rd.offset],
        ["Hinge_Par",        sol.hinge_par, rd.hinge_par],
      ];
      for (const [name, raw, rounded] of params) {{
        const rawMM = (raw * 1000).toFixed(4);
        const rndMM = (rounded * 1000).toFixed(1);
        const changed = Math.abs(raw - rounded) > 1e-8;
        const color = changed ? "#c8b88a" : "#666";
        html += '<tr>' +
          '<td style="padding:2px 4px;color:#888;">' + name + '</td>' +
          '<td style="padding:2px 4px;text-align:right;color:#555;font-variant-numeric:tabular-nums;">' + rawMM + '</td>' +
          '<td style="padding:2px 4px;text-align:right;color:' + color + ';font-variant-numeric:tabular-nums;font-weight:' + (changed ? '600' : '400') + ';">' + rndMM + '</td>' +
          '</tr>';
      }}

      // ── Constraint residuals table ──
      html += '<tr><td colspan="3" style="border-top:1px solid #1f222b;padding:6px 4px 2px;font-size:8px;text-transform:uppercase;letter-spacing:1px;color:#555;">Constraint Residuals</td></tr>';
      const resKeys = Object.keys(res);
      for (const key of resKeys) {{
        const val = res[key];
        const absVal = Math.abs(val);
        let color = "#44bb44";
        if (absVal >= 0.1)       color = "#f87171";
        else if (absVal >= 0.01) color = "#c8b88a";
        else if (absVal >= 0.001) color = "#888";
        html += '<tr>' +
          '<td style="padding:1px 4px;color:#666;font-size:10px;" colspan="2">' + key.replace(/_/g, " ") + '</td>' +
          '<td style="padding:1px 4px;text-align:right;color:' + color + ';font-variant-numeric:tabular-nums;font-size:10px;">' + val.toFixed(6) + ' mm</td>' +
          '</tr>';
      }}

      // ── Achieved dimensions ──
      if (rd.depth_stow !== undefined) {{
        html += '<tr><td colspan="3" style="border-top:1px solid #1f222b;padding:6px 4px 2px;font-size:8px;text-transform:uppercase;letter-spacing:1px;color:#555;">Achieved Dimensions</td></tr>';
        const dimRows = [
          ["Stow depth",    rd.depth_stow],
          ["Deploy depth",  rd.depth_deploy],
          ["Stow width",    rd.width_stow],
          ["Deploy height", rd.height],
        ];
        for (const [label, val] of dimRows) {{
          if (val !== undefined) {{
            html += '<tr>' +
              '<td style="padding:1px 4px;color:#555;" colspan="2">' + label + '</td>' +
              '<td style="padding:1px 4px;text-align:right;color:#888;font-variant-numeric:tabular-nums;">' + (val * 1000).toFixed(2) + ' mm</td>' +
              '</tr>';
          }}
        }}
      }}

      tbody.innerHTML = html;
    }}

    function generateCsv() {{
      if (!currentRounded) return null;
      const rd = currentRounded;
      const isFeasible = rd && !String(rd.status || "").startsWith("infeasible");
      if (!isFeasible) return null;

      const toMM = (v) => (v * 1000).toFixed(1);
      const holeDiam = parseFloat(document.getElementById("csvHoleDiam").value) || 2.0;
      const hingePerp = parseFloat(document.getElementById("csvHingePerp").value) || 2.0;
      const plateThick = toMM(rd.thickness);

      const rows = [
        ["Name", "Unit", "Expression", "Value", "Comments", "Favorite"],
        ["Hole_Diameter",   "mm", holeDiam.toFixed(2) + " mm",  holeDiam.toFixed(2),  "", "false"],
        ["Plate_Thickness", "mm", plateThick + " mm",           plateThick,           "", "false"],
        ["Member_Width",    "mm", "Plate_Thickness",            plateThick,           "", "false"],
        ["Long_H2H",       "mm", toMM(rd.long) + " mm",        toMM(rd.long),        "", "false"],
        ["Short_H2H",      "mm", toMM(rd.short) + " mm",       toMM(rd.short),       "", "false"],
        ["Offset_H2H",     "mm", toMM(rd.offset) + " mm",      toMM(rd.offset),      "", "false"],
        ["Hinge_Par",       "mm", toMM(rd.hinge_par) + " mm",   toMM(rd.hinge_par),   "", "false"],
        ["Hinge_Perp",      "mm", hingePerp.toFixed(2) + " mm", hingePerp.toFixed(2), "", "false"],
      ];
      return rows.map(r => r.join(",")).join("\\n");
    }}

    document.getElementById("exportBtn").addEventListener("click", function() {{
      const csv = generateCsv();
      if (!csv) return;
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "scissor_parameters.csv";
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      const status = document.getElementById("exportStatus");
      status.textContent = "✓ Exported scissor_parameters.csv (rounded values)";
      status.style.opacity = 1;
      setTimeout(() => {{ status.style.opacity = 0; }}, 4000);
    }});

    // Hook hinge slider to also update CSV preview
    document.getElementById("hingePerSlider").addEventListener("input", function() {{
      document.getElementById("hingePerVal").textContent = parseFloat(this.value).toFixed(4);
      rebuildFromUI();
      updateCsvPreview();
    }});
  </script>
</body>
</html>"""

    with open(out_html, "w") as f:
        f.write(html)

    print(f"Interactive plot written to: {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stow-width", type=float, default=None)
    parser.add_argument("--stow-depth", type=float, default=None)
    parser.add_argument("--expanded-depth", type=float, default=None)
    parser.add_argument("--hinge-par", type=float, default=None)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--short", type=float, default=None)
    parser.add_argument("--long", type=float, default=None)

    parser.add_argument("--offset-min", type=float, default=0.001)
    parser.add_argument("--offset-max", type=float, default=2.0)
    parser.add_argument("--offset-steps", type=int, default=50)

    parser.add_argument("--thickness-list", type=str, default=None,
                        help="Comma-separated thicknesses, e.g. 0.002,0.005,0.01")
    parser.add_argument("--thickness-min", type=float, default=0.002)
    parser.add_argument("--thickness-max", type=float, default=0.05)
    parser.add_argument("--thickness-steps", type=int, default=8)

    parser.add_argument("--plot-mode", choices=["ratio", "actual"], default="ratio")
    parser.add_argument("--actual-x", choices=["depth_actual", "height_actual", "width_actual"],
                        default="depth_actual")
    parser.add_argument("--actual-y", choices=["height_actual", "depth_actual", "width_actual"],
                        default="height_actual")

    parser.add_argument("--tee", action="store_true")
    parser.add_argument("-o", "--output", default="scissor_results.json")
    parser.add_argument("--plot-html", default="scissor_sweep.html")
    args = parser.parse_args()

    results = sweep_offsets_and_thicknesses(args)

    if len(results) == 0:
        raise RuntimeError("No feasible solutions found in the (thickness, offset) sweep range.")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} feasible solutions to {args.output}")

    make_interactive_plot(
        results,
        out_html=args.plot_html,
        actual_axes=(args.actual_x, args.actual_y),
        initial_mode=args.plot_mode
    )