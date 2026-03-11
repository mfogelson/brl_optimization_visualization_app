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
            "n_cells": 10, #int(round(value(model.n))),
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
// Realistic materials: carbon fiber beams, brushed aluminum hinges

(function() {
  const DEG = Math.PI / 180;
  const RAD = 180 / Math.PI;

  // ── Procedural textures ──
  function makeCarbonFiberTexture(size) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, size, size);
    const cell = size / 16;
    for (let y = 0; y < size; y += cell) {
      for (let x = 0; x < size; x += cell) {
        const checker = ((x / cell + y / cell) % 2 === 0);
        ctx.fillStyle = checker ? '#222222' : '#181818';
        ctx.fillRect(x, y, cell, cell);
        ctx.strokeStyle = checker ? '#252525' : '#1c1c1c';
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + cell, y + cell); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + cell, y); ctx.lineTo(x, y + cell); ctx.stroke();
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(4, 1);
    return tex;
  }

  function makeCarbonNormalMap(size) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    const cell = size / 16;
    for (let y = 0; y < size; y += cell) {
      for (let x = 0; x < size; x += cell) {
        const checker = ((x / cell + y / cell) % 2 === 0);
        ctx.fillStyle = checker ? '#8888ff' : '#7878ff';
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(4, 1);
    return tex;
  }

  function makeBrushedMetalTexture(size) {
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#c8c8c8';
    ctx.fillRect(0, 0, size, size);
    for (let y = 0; y < size; y++) {
      const v = 180 + Math.random() * 30;
      ctx.strokeStyle = `rgb(${v},${v},${v})`;
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(size, y + 0.5); ctx.stroke();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1, 4);
    return tex;
  }

  // ── Material cache ──
  let matCache = null;
  function getMaterials() {
    if (matCache) return matCache;
    const cfTex = makeCarbonFiberTexture(256);
    const cfNorm = makeCarbonNormalMap(256);
    const alTex = makeBrushedMetalTexture(256);

    matCache = {
      carbonFiber: new THREE.MeshStandardMaterial({
        map: cfTex, normalMap: cfNorm, normalScale: new THREE.Vector2(0.3, 0.3),
        color: 0x2a2a2a, metalness: 0.15, roughness: 0.35,
      }),
      carbonFiberOffset: new THREE.MeshStandardMaterial({
        map: cfTex, normalMap: cfNorm, normalScale: new THREE.Vector2(0.3, 0.3),
        color: 0x1a1a22, metalness: 0.15, roughness: 0.35,
      }),
      aluminum: new THREE.MeshStandardMaterial({
        map: alTex, color: 0xd0d0d8, metalness: 0.85, roughness: 0.25,
      }),
      aluminumDark: new THREE.MeshStandardMaterial({
        map: alTex, color: 0x8888a0, metalness: 0.85, roughness: 0.3,
      }),
      aluminumGreen: new THREE.MeshStandardMaterial({
        map: alTex, color: 0x88aa88, metalness: 0.8, roughness: 0.28,
      }),
      carbonShort: new THREE.MeshStandardMaterial({
        map: cfTex, normalMap: cfNorm, normalScale: new THREE.Vector2(0.3, 0.3),
        color: 0x1a2a1a, metalness: 0.15, roughness: 0.35,
      }),
      longeron: new THREE.MeshStandardMaterial({
        map: cfTex, normalMap: cfNorm, normalScale: new THREE.Vector2(0.3, 0.3),
        color: 0x2a1a2a, metalness: 0.15, roughness: 0.35,
      }),
      joint: new THREE.MeshStandardMaterial({
        color: 0x444444, metalness: 0.9, roughness: 0.15,
      }),
    };
    return matCache;
  }

  const MAT_CARBON  = 'carbonFiber';
  const MAT_CARBON_OFF = 'carbonFiberOffset';
  const MAT_CARBON_SHORT = 'carbonShort';
  const MAT_ALU     = 'aluminum';
  const MAT_ALU_DARK = 'aluminumDark';
  const MAT_ALU_GREEN = 'aluminumGreen';
  const MAT_LONGERON = 'longeron';
  const MAT_JOINT   = 'joint';

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

  function addTube(grp, a, b, matKey, radius) {
    const dir = new THREE.Vector3().subVectors(b, a);
    const len = dir.length();
    if (len < 1e-5) return;
    const geo = new THREE.CylinderGeometry(radius, radius, len, 16);
    const mat = getMaterials()[matKey];
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.addVectors(a, b).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    grp.add(mesh);
  }

  function addBall(grp, pos, matKey, radius) {
    const geo = new THREE.SphereGeometry(radius, 16, 16);
    const mat = getMaterials()[matKey];
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

  function addShortDnHinge(grp, dnApex, pA, hingePar, hingePer, matKey, radius, Jrs) {
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
    addTube(grp, dnApex, hParEnd, matKey, radius);
    addTube(grp, hParEnd, hPerEnd, matKey, radius);
    addBall(grp, hParEnd, matKey, Jrs);
    addBall(grp, hPerEnd, matKey, Jrs);
  }

  function addRefFrame(grp, pos, size, T) {
    const r = T * 0.4;
    const matR = new THREE.MeshStandardMaterial({ color: 0xff0000, metalness: 0.5, roughness: 0.4 });
    const matG = new THREE.MeshStandardMaterial({ color: 0x00ff00, metalness: 0.5, roughness: 0.4 });
    const matB = new THREE.MeshStandardMaterial({ color: 0x0000ff, metalness: 0.5, roughness: 0.4 });
    function tube(a, b, mat) {
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length(); if (len < 1e-5) return;
      const geo = new THREE.CylinderGeometry(r, r, len, 8);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.addVectors(a, b).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
      grp.add(mesh);
    }
    tube(pos, new THREE.Vector3(pos.x + size, pos.y, pos.z), matR);
    tube(pos, new THREE.Vector3(pos.x, pos.y + size, pos.z), matG);
    tube(pos, new THREE.Vector3(pos.x, pos.y, pos.z + size), matB);
  }

  function addRotatedRefFrame(grp, pos, size, T, angle) {
    const r = T * 0.4;
    const ca = Math.cos(angle), sa = Math.sin(angle);
    const matR = new THREE.MeshStandardMaterial({ color: 0xff0000, metalness: 0.5, roughness: 0.4 });
    const matG = new THREE.MeshStandardMaterial({ color: 0x00ff00, metalness: 0.5, roughness: 0.4 });
    const matB = new THREE.MeshStandardMaterial({ color: 0x0000ff, metalness: 0.5, roughness: 0.4 });
    function tube(a, b, mat) {
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length(); if (len < 1e-5) return;
      const geo = new THREE.CylinderGeometry(r, r, len, 8);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.addVectors(a, b).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
      grp.add(mesh);
    }
    tube(pos, new THREE.Vector3(pos.x + size * ca, pos.y, pos.z - size * sa), matR);
    tube(pos, new THREE.Vector3(pos.x, pos.y + size, pos.z), matG);
    tube(pos, new THREE.Vector3(pos.x + size * sa, pos.y, pos.z + size * ca), matB);
  }

  function buildScene(L, O, alpha, nCells, T, hingePar, hingePer, S, h, showCells, longeronLen, showRefFrames, stowWidth) {
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

    for (let i = 0; i < cellsToShow; i++) {
      const g = new THREE.Group();
      g.position.set(0, -i * c, 0);

      // ── RIGHT HALF (+X) ──
      const rOup = new THREE.Vector3(0, 0, +T);
      const rOdn = new THREE.Vector3(0, 0, -T);
      const rP1 = new THREE.Vector3(P1x, P1y, +T);
      const rP2 = new THREE.Vector3(P2x, P2y, -T);

      addTube(g, rOup, rP1, MAT_CARBON, T);
      addTube(g, rOdn, rP2, MAT_CARBON, T);

      const rP1_up = new THREE.Vector3(P1x, P1y, +2*T);
      addBall(g, rP1_up, MAT_ALU, Jrs);
      const rH2 = new THREE.Vector3(P1x, P1y, 2*T+hingePer);
      addTube(g, rP1_up, rH2, MAT_ALU, Toff);

      const rSH = new THREE.Vector3(
        rH2.x + hingePer * Math.sin(planeAngle), rH2.y,
        rH2.z + hingePer * Math.cos(planeAngle)
      );
      addTube(g, rH2, rSH, MAT_ALU_GREEN, Toff);

      const rSH_par = new THREE.Vector3(
        rSH.x - hingePar * Math.cos(planeAngle), rSH.y,
        rSH.z + hingePar * Math.sin(planeAngle)
      );
      addTube(g, rSH, rSH_par, MAT_ALU_GREEN, Toff);
      addBall(g, rSH, MAT_ALU_GREEN, Jrs);
      addBall(g, rSH_par, MAT_ALU_GREEN, Jrs);

      if (showRefFrames) addRefFrame(g, rH2, refSize, T);
      if (showRefFrames) addRotatedRefFrame(g, rSH_par, refSize, T, planeAngle);

      const rSH_par_up = new THREE.Vector3(rSH_par.x, rSH_par.y, rSH_par.z + T);
      const rShortDn = shortApex(rSH_par_up, planeAngle, halfBeta, S, -1);
      const rSH_par_up_up = new THREE.Vector3(rSH_par_up.x, rSH_par_up.y, rSH_par_up.z + T);
      const rShortUp = shortApex(rSH_par_up_up, planeAngle, halfBeta, S, +1);
      addTube(g, rSH_par_up_up, rShortUp, MAT_CARBON_SHORT, Toff);
      addBall(g, rShortUp, MAT_CARBON_SHORT, Jrs);
      addBall(g, rSH_par_up_up, MAT_CARBON_SHORT, Jrs);

      if (i > 0) {
        addTube(g, rSH_par_up, rShortDn, MAT_CARBON_SHORT, Toff);
        addBall(g, rSH_par_up, MAT_CARBON_SHORT, Jrs);
        addBall(g, rShortDn, MAT_CARBON_SHORT, Jrs);
        const rShortDn_dn = new THREE.Vector3(rShortDn.x, rShortDn.y, rShortDn.z - T);
        addShortDnHinge(g, rShortDn_dn, planeAngle, hingePar, -hingePer, MAT_ALU_GREEN, Toff, Jrs);
        addBall(g, rShortDn_dn, MAT_ALU_GREEN, Jrs);
      }

      if (i === cellsToShow - 1) {
        const rP3 = new THREE.Vector3(P3x, P3y, +T);
        const rP3_up = new THREE.Vector3(P3x, P3y, +2*T);
        const rP2_up = new THREE.Vector3(P2x, P2y, +T);
        addTube(g, rP2_up, rP3, MAT_CARBON_OFF, Toff);
        const rH3up = new THREE.Vector3(P3x, P3y, 2*T + hingePer);
        addTube(g, rP3_up, rH3up, MAT_ALU, Toff);
        const rSH3 = new THREE.Vector3(
          rH3up.x + hingePer * Math.sin(planeAngle), rH3up.y,
          rH3up.z + hingePer * Math.cos(planeAngle)
        );
        addTube(g, rH3up, rSH3, MAT_ALU_GREEN, Toff);
        const rSH3_par = new THREE.Vector3(
          rSH3.x - hingePar * Math.cos(planeAngle), rSH3.y,
          rSH3.z + hingePar * Math.sin(planeAngle)
        );
        addTube(g, rSH3, rSH3_par, MAT_ALU_GREEN, Toff);
        if (showRefFrames) addRefFrame(g, rH3up, refSize, T);
        if (showRefFrames) addRotatedRefFrame(g, rSH3_par, refSize, T, planeAngle);
        const rSH3_par_up = new THREE.Vector3(rSH3_par.x, rSH3_par.y, rSH3_par.z + T);
        const rSh3Dn = shortApex(rSH3_par_up, planeAngle, halfBeta, S, -1);
        addTube(g, rSH3_par_up, rSh3Dn, MAT_CARBON_SHORT, Toff);
        addBall(g, rSH3_par_up, MAT_CARBON_SHORT, Jrs);
        addBall(g, rSh3Dn, MAT_CARBON_SHORT, Jrs);
        const rSh3Dn_up = new THREE.Vector3(rSh3Dn.x, rSh3Dn.y, rSh3Dn.z - T);
        addShortDnHinge(g, rSh3Dn_up, planeAngle, hingePar, -hingePer, MAT_ALU_GREEN, Toff, Jrs);
        addBall(g, rSh3Dn_up, MAT_ALU_GREEN, Jrs);
        addBall(g, rP2_up, MAT_CARBON_OFF, Jrs);
        addBall(g, rP3_up, MAT_ALU, Jrs);
        addBall(g, rP3, MAT_CARBON_OFF, Jrs);
        addBall(g, rH3up, MAT_ALU, Jrs);
        addBall(g, rSH3, MAT_ALU_GREEN, Jrs);
        addBall(g, rSH3_par, MAT_ALU_GREEN, Jrs);
      }

      addBall(g, rOup, MAT_JOINT, Jr);
      addBall(g, rOdn, MAT_JOINT, Jr);
      addBall(g, rP1, MAT_CARBON, Jrs);
      addBall(g, rP2, MAT_CARBON, Jrs);
      addBall(g, rH2, MAT_ALU, Jrs);

      // ── Longerons (right) ──
      const longeron_length = longeronLen;
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
      addTube(g, rP1_offset, rP1_Longeron, MAT_LONGERON, Toff);
      addTube(g, rP3_offset, rP3_Longeron, MAT_LONGERON, Toff);
      addTube(g, rP1, rP1_offset, MAT_ALU, Jrs);
      addTube(g, rP3r, rP3_offset, MAT_ALU, Jrs);
      addBall(g, rP1_Longeron, MAT_LONGERON, Jrs);
      addBall(g, rP3_Longeron, MAT_LONGERON, Jrs);

      if (i > 0 && i % 2 === 0) {
        const rLongUp_ref = new THREE.Vector3(rShortUp.x, rShortUp.y, rShortUp.z +2*T);
        addTube(g, rShortUp, rLongUp_ref, MAT_ALU, Toff);
        const rLongUp = new THREE.Vector3(
          rShortUp.x + Math.cos(longeron_halfAngle)*longeron_length,
          rShortUp.y + Math.sin(longeron_halfAngle)*longeron_length,
          rShortUp.z +2*T
        );
        addTube(g, rLongUp_ref, rLongUp, MAT_LONGERON, Toff);
        addBall(g, rLongUp_ref, MAT_LONGERON, Jrs);
        addBall(g, rLongUp, MAT_LONGERON, Jrs);

        const rLongDn_ref = new THREE.Vector3(rShortDn.x, rShortDn.y, rShortDn.z +2*T);
        const rLongDn = new THREE.Vector3(
          rShortDn.x + Math.cos(longeron_halfAngle)*longeron_length,
          rShortDn.y - Math.sin(longeron_halfAngle)*longeron_length,
          rShortDn.z +2*T
        );
        addTube(g, rLongDn_ref, rLongDn, MAT_LONGERON, Toff);
        addBall(g, rLongDn_ref, MAT_LONGERON, Jrs);
        addBall(g, rLongDn, MAT_LONGERON, Jrs);
      }

      // ── LEFT HALF (-X) ──
      const lOup = new THREE.Vector3(0, 0, -T);
      const lOdn = new THREE.Vector3(0, 0, +T);
      const lP1 = new THREE.Vector3(-P1x, P1y, -T);
      const lP2 = new THREE.Vector3(-P2x, P2y, +T);

      addTube(g, lOup, lP1, MAT_CARBON, Toff);
      addTube(g, lOdn, lP2, MAT_CARBON, Toff);

      const spacer = new THREE.Vector3(-P1x, P1y, 2*T);
      addTube(g, lP1, spacer, MAT_ALU, Toff);
      const lH2 = new THREE.Vector3(spacer.x, spacer.y, spacer.z + hingePer);
      addTube(g, spacer, lH2, MAT_ALU, Toff);
      addBall(g, spacer, MAT_ALU, Jrs);

      const lSH = new THREE.Vector3(
        lH2.x - hingePer * Math.sin(leftPlaneAngle), lH2.y,
        lH2.z - hingePer * Math.cos(leftPlaneAngle)
      );
      addTube(g, lH2, lSH, MAT_ALU_GREEN, Toff);
      const lSH_par = new THREE.Vector3(
        lSH.x - hingePar * Math.cos(leftPlaneAngle), lSH.y,
        lSH.z + hingePar * Math.sin(leftPlaneAngle)
      );
      addTube(g, lSH, lSH_par, MAT_ALU_GREEN, Toff);
      addBall(g, lSH, MAT_ALU_GREEN, Jrs);
      addBall(g, lSH_par, MAT_ALU_GREEN, Jrs);

      if (showRefFrames) addRefFrame(g, lH2, refSize, T);
      if (showRefFrames) addRotatedRefFrame(g, lSH_par, refSize, T, -leftPlaneAngle);

      const lSH_par_up = new THREE.Vector3(lSH_par.x, lSH_par.y, lSH_par.z + T);
      const lShortDn = shortApex(lSH_par_up, leftPlaneAngle, halfBeta, S, -1);
      const lSH_par_up_up = new THREE.Vector3(lSH_par_up.x, lSH_par_up.y, lSH_par_up.z + T);
      const lShortUp = shortApex(lSH_par_up_up, leftPlaneAngle, halfBeta, S, +1);
      addTube(g, lSH_par_up_up, lShortUp, MAT_CARBON_SHORT, Toff);
      addBall(g, lShortUp, MAT_CARBON_SHORT, Jrs);
      addBall(g, lSH_par_up_up, MAT_CARBON_SHORT, Jrs);

      if (i > 0) {
        addTube(g, lSH_par_up, lShortDn, MAT_CARBON_SHORT, Toff);
        addBall(g, lSH_par_up, MAT_CARBON_SHORT, Jrs);
        const lShortDn_dn = new THREE.Vector3(lShortDn.x, lShortDn.y, lShortDn.z - T);
        addShortDnHinge(g, lShortDn_dn, leftPlaneAngle, hingePar, hingePer, MAT_ALU_GREEN, Toff, Jrs);
        addBall(g, lShortDn_dn, MAT_ALU_GREEN, Jrs);
        addBall(g, lShortDn, MAT_CARBON_SHORT, Jrs);
      }

      if (i === cellsToShow - 1) {
        const lP3 = new THREE.Vector3(-P3x, P3y, -T);
        const lP2_dn = new THREE.Vector3(-P2x, P2y, -T);
        addBall(g, lP2_dn, MAT_CARBON_OFF, Jrs);
        addTube(g, lP2_dn, lP3, MAT_CARBON_OFF, Toff);
        const lP3_up = new THREE.Vector3(-P3x, P3y, +2*T);
        addTube(g, lP3_up, lP3, MAT_ALU, Toff);
        const lH3up = new THREE.Vector3(-P3x, P3y, 2*T + hingePer);
        addTube(g, lP3_up, lH3up, MAT_ALU, Toff);
        const lSH3 = new THREE.Vector3(
          lH3up.x - hingePer * Math.sin(leftPlaneAngle), lH3up.y,
          lH3up.z - hingePer * Math.cos(leftPlaneAngle)
        );
        addTube(g, lH3up, lSH3, MAT_ALU_GREEN, Toff);
        const lSH3_par = new THREE.Vector3(
          lSH3.x - hingePar * Math.cos(leftPlaneAngle), lSH3.y,
          lSH3.z + hingePar * Math.sin(leftPlaneAngle)
        );
        addTube(g, lSH3, lSH3_par, MAT_ALU_GREEN, Toff);
        if (showRefFrames) addRefFrame(g, lH3up, refSize, T);
        if (showRefFrames) addRotatedRefFrame(g, lSH3_par, refSize, T, -leftPlaneAngle);
        const lSH3_par_up = new THREE.Vector3(lSH3_par.x, lSH3_par.y, lSH3_par.z + T);
        const lSh3Dn = shortApex(lSH3_par_up, leftPlaneAngle, halfBeta, S, -1);
        addTube(g, lSH3_par_up, lSh3Dn, MAT_CARBON_SHORT, Toff);
        addBall(g, lSH3_par_up, MAT_CARBON_SHORT, Jrs);
        addBall(g, lSh3Dn, MAT_CARBON_SHORT, Jrs);
        const lSh3Dn_up = new THREE.Vector3(lSh3Dn.x, lSh3Dn.y, lSh3Dn.z - T);
        addShortDnHinge(g, lSh3Dn_up, leftPlaneAngle, hingePar, hingePer, MAT_ALU_GREEN, Toff, Jrs);
        addBall(g, lSh3Dn_up, MAT_ALU_GREEN, Jrs);
        addBall(g, lP3_up, MAT_ALU, Jrs);
        addBall(g, lP3, MAT_CARBON_OFF, Jrs);
        addBall(g, lH3up, MAT_ALU, Jrs);
        addBall(g, lSH3, MAT_ALU_GREEN, Jrs);
        addBall(g, lSH3_par, MAT_ALU_GREEN, Jrs);
      }

      addBall(g, lP1, MAT_CARBON, Jrs);
      addBall(g, lP2, MAT_CARBON, Jrs);
      addBall(g, lH2, MAT_ALU, Jrs);

      // Longerons (left)
      const longeron_length_l = longeronLen;
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
      addTube(g, lP1_offset, lP1_Longeron, MAT_LONGERON, Toff);
      addTube(g, lP3_offset, lP3_Longeron, MAT_LONGERON, Toff);
      addTube(g, lP1_offset, lP1, MAT_ALU, Jrs);
      addTube(g, lP3l, lP3_offset, MAT_ALU, Jrs);
      addBall(g, lP1_Longeron, MAT_LONGERON, Jrs);
      addBall(g, lP3_Longeron, MAT_LONGERON, Jrs);

      if (i > 0 && i % 2 === 1) {
        const lLongUp_ref = new THREE.Vector3(lShortUp.x, lShortUp.y, lShortUp.z +2*T);
        addTube(g, lShortUp, lLongUp_ref, MAT_ALU, Toff);
        const lLongUp = new THREE.Vector3(
          lShortUp.x - Math.cos(longeron_halfAngle_l)*longeron_length_l,
          lShortUp.y + Math.sin(longeron_halfAngle_l)*longeron_length_l,
          lShortUp.z +2*T
        );
        addTube(g, lLongUp_ref, lLongUp, MAT_LONGERON, Toff);
        addBall(g, lLongUp_ref, MAT_LONGERON, Jrs);
        addBall(g, lLongUp, MAT_LONGERON, Jrs);

        const lLongDn_ref = new THREE.Vector3(lShortDn.x, lShortDn.y, lShortDn.z +2*T);
        const lLongDn = new THREE.Vector3(
          lShortDn.x - Math.cos(longeron_halfAngle_l)*longeron_length_l,
          lShortDn.y - Math.sin(longeron_halfAngle_l)*longeron_length_l,
          lShortDn.z +2*T
        );
        addTube(g, lLongDn_ref, lLongDn, MAT_LONGERON, Toff);
        addBall(g, lLongDn_ref, MAT_LONGERON, Jrs);
        addBall(g, lLongDn, MAT_LONGERON, Jrs);
      }

      root.add(g);
    }

    // ── Solar panel surface (translucent blue, on top, stow width x deployed depth) ──
    const panelDepth = c * cellsToShow;
    const panelWidth = stowWidth || (2 * S + 4 * hingePar);
    if (panelDepth > 1e-4 && panelWidth > 1e-4) {
      const panelGeo = new THREE.PlaneGeometry(panelWidth, panelDepth);
      const panelMat = new THREE.MeshStandardMaterial({
        color: 0x1155cc,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
        metalness: 0.3,
        roughness: 0.6,
      });
      const panel = new THREE.Mesh(panelGeo, panelMat);
      // Position at top of structure (P1y-gy level), centered over deployed cells
      panel.position.set(0, P1y-gy, -0.06);
      // Rotate to lie flat in XY plane (default PlaneGeometry is in XY, we want XY)
      root.add(panel);

      // Grid lines on panel for solar cell look
      const gridMat = new THREE.LineBasicMaterial({ color: 0x2266dd, transparent: true, opacity: 0.25 });
      const nGridX = Math.max(2, Math.round(panelWidth / (T * 20)));
      const nGridY = Math.max(2, Math.round(panelDepth / (T * 20)));
      for (let gx = 0; gx <= nGridX; gx++) {
        const x = -panelWidth/2 + gx * panelWidth / nGridX;
        const pts = [new THREE.Vector3(x, P1y-gy, -0.061), new THREE.Vector3(x, P1y - panelDepth, 0.001)];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
        root.add(new THREE.Line(lineGeo, gridMat));
      }
      for (let gy = 0; gy <= nGridY; gy++) {
        const y = P1y - gy * panelDepth / nGridY;
        const pts = [new THREE.Vector3(-panelWidth/2, y, 0.001), new THREE.Vector3(panelWidth/2, y, 0.001)];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
        root.add(new THREE.Line(lineGeo, gridMat));
      }
    }

    // ── Satellite bus (top of structure) ──
    const busWidth = panelWidth;
    const busHeight = busWidth * 0.6;
    const busDepth = busWidth * 0.8;
    const busY = P1y + busHeight / 2 + T * 4;

    const busMat = new THREE.MeshStandardMaterial({
      color: 0xcccc88,
      metalness: 0.7,
      roughness: 0.3,
    });
    const busGeo = new THREE.BoxGeometry(busWidth, busHeight, busDepth);
    const bus = new THREE.Mesh(busGeo, busMat);
    bus.position.set(0, busY, 0);
    root.add(bus);

    // Solar panel wings on satellite (small fixed panels)
    const wingW = busWidth * 0.15;
    const wingH = busHeight * 0.8;
    const wingD = busDepth * 1.8;
    const wingMat = new THREE.MeshStandardMaterial({
      color: 0x1144aa,
      metalness: 0.4,
      roughness: 0.5,
      side: THREE.DoubleSide,
    });
    const wingGeoL = new THREE.BoxGeometry(wingW, wingH * 0.05, wingD);
    const wingL = new THREE.Mesh(wingGeoL, wingMat);
    wingL.position.set(-busWidth/2 - wingW/2, busY, 0);
    root.add(wingL);
    const wingR = new THREE.Mesh(wingGeoL, wingMat);
    wingR.position.set(busWidth/2 + wingW/2, busY, 0);
    root.add(wingR);

    // Antenna dish
    const dishR = busWidth * 0.2;
    const dishGeo = new THREE.SphereGeometry(dishR, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
    const dishMat = new THREE.MeshStandardMaterial({
      color: 0xeeeeee,
      metalness: 0.8,
      roughness: 0.2,
      side: THREE.DoubleSide,
    });
    const dish = new THREE.Mesh(dishGeo, dishMat);
    dish.position.set(busWidth * 0.25, busY + busHeight/2, busDepth * 0.15);
    dish.rotation.x = -Math.PI / 2;
    root.add(dish);

    // Antenna mast
    const mastH = busHeight * 0.3;
    const mastGeo = new THREE.CylinderGeometry(T * 0.3, T * 0.3, mastH, 8);
    const mastMat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, metalness: 0.8, roughness: 0.3 });
    const mast = new THREE.Mesh(mastGeo, mastMat);
    mast.position.set(busWidth * 0.25, busY + busHeight/2 + mastH/2, busDepth * 0.15);
    root.add(mast);

    // Connecting strut from satellite to structure top
    const strutMat = getMaterials()[MAT_ALU];
    const strutTop = new THREE.Vector3(0, busY - busHeight/2, 0);
    const strutBot = new THREE.Vector3(0, P1y, 0);
    addTube(root, strutTop, strutBot, MAT_ALU, T * 1.5);

    return { group: root, c, theta, halfBeta, cellsToShow };
  }

  // ── Viewer state ──
  let viewerState = null;

  function disposeGroup(group) {
    group.traverse(ch => {
      if (ch.geometry) ch.geometry.dispose();
    });
  }

  function initViewer(container) {
    const W = container.clientWidth, H = container.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x12141a, 1);
    renderer.physicallyCorrectLights = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xc8d8f0, 0x303840, 0.6));
    const d1 = new THREE.DirectionalLight(0xfff0e0, 1.2);
    d1.position.set(5, 8, 7); scene.add(d1);
    const d2 = new THREE.DirectionalLight(0xc0d0f0, 0.4);
    d2.position.set(-5, -4, 6); scene.add(d2);
    const d3 = new THREE.DirectionalLight(0xe0e8ff, 0.3);
    d3.position.set(0, -2, -8); scene.add(d3);

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

  function rebuildLinkage(sol, alphaFrac, showCells, thickness, hingePer, showRefFrames) {
    if (!viewerState) return;
    const s = viewerState;
    if (s.linkage) { s.scene.remove(s.linkage); disposeGroup(s.linkage); }

    const alpha = (sol.alpha_stow || 0) + alphaFrac * ((sol.alpha_deploy || 1) - (sol.alpha_stow || 0));
    const deployC = compute2D(sol.long, sol.offset, sol.alpha_deploy || 1).c;
    const longeronLen = 0.5 * deployC;
    const { group, c } = buildScene(
      sol.long, sol.offset, alpha, sol.n_cells,
      thickness, sol.hinge_par, hingePer,
      sol.short, sol.hinge_par, showCells, longeronLen, showRefFrames,
      sol.width_actual || 0
    );
    s.linkage = group;
    s.scene.add(group);
    s.camera.userData.tgt = new THREE.Vector3(0, -c * Math.max(0, Math.min(showCells, sol.n_cells) - 1) / 2, 0);
  }

  // ── GIF recording ──
  function captureFrame() {
    if (!viewerState) return null;
    const s = viewerState;
    const t = s.camera.userData.tgt || new THREE.Vector3();
    s.camera.position.set(
      t.x + s.mouse.px + s.mouse.d * Math.sin(s.mouse.ry) * Math.cos(s.mouse.rx),
      t.y + s.mouse.py + s.mouse.d * Math.sin(s.mouse.rx),
      t.z + s.mouse.d * Math.cos(s.mouse.ry) * Math.cos(s.mouse.rx)
    );
    s.camera.lookAt(t.x + s.mouse.px, t.y + s.mouse.py, t.z);
    s.renderer.setClearColor(0x000000, 0);
    s.renderer.render(s.scene, s.camera);
    const dataUrl = s.renderer.domElement.toDataURL("image/png");
    s.renderer.setClearColor(0x12141a, 1);
    return dataUrl;
  }

  window.viewerInit = initViewer;
  window.viewerRebuild = rebuildLinkage;
  window.viewerCaptureFrame = captureFrame;
  window.getViewerState = function() { return viewerState; };
})();
"""


def make_interactive_plot(results, out_html="scissor_sweep.html",
                          actual_axes=("depth_actual", "height_actual"),
                          initial_mode="ratio"):
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
    kept_indices = np.where(keep)[0]
    payloads = [payloads[i] for i in kept_indices]
    df = df.reset_index(drop=True)

    if df.empty:
        raise RuntimeError("All points were non-finite.")

    print(f"[plot] Plotting {len(df)} points (after filtering non-finite).")

    thickness_levels = sorted(df["thickness"].unique())
    fig = go.Figure()

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

    for t in thickness_levels:
        mask = df["thickness"] == t
        dft = df[mask]
        original_indices = dft.index.tolist()

        cd = []
        for idx in original_indices:
            cd.append([
                float(dft.loc[idx, "thickness"]),
                float(dft.loc[idx, "offset"]),
                int(dft.loc[idx, "n_cells"]),
                float(dft.loc[idx, "long"]),
                float(dft.loc[idx, "short"]),
                float(dft.loc[idx, "hinge_par"]),
                int(idx),
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
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/exporters/GLTFExporter.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
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
    #exportBtn, #gifBtn {{
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
    #exportBtn:hover, #gifBtn:hover, #glbBtn:hover, #glbAnimBtn:hover {{ opacity: 0.85; }}
    #exportBtn:disabled, #gifBtn:disabled, #glbBtn:disabled, #glbAnimBtn:disabled {{ opacity: 0.3; cursor: not-allowed; }}
    #gifBtn {{
      background: linear-gradient(135deg, #88aacc, #6688aa);
      margin-top: 6px;
    }}
    #exportStatus, #gifStatus {{
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
    .gif-settings {{
      display: flex;
      gap: 8px;
      margin-bottom: 6px;
      align-items: center;
    }}
    .gif-settings label {{
      font-size: 9px;
      color: #666;
    }}
    .gif-settings input {{
      width: 50px;
      background: #090b10;
      border: 1px solid #282b36;
      border-radius: 3px;
      color: #88aacc;
      font-size: 10px;
      font-family: inherit;
      padding: 2px 4px;
      text-align: right;
    }}
    .gif-settings input:focus {{
      outline: none;
      border-color: #88aacc;
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
          <label>&alpha; fraction</label>
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
        <input type="range" id="cellsSlider" min="1" max="999" step="1" value="2" />
      </div>
      <div style="margin:6px 0;">
        <label style="font-size:10px;color:#888;cursor:pointer;">
          <input type="checkbox" id="refFrameToggle" style="vertical-align:middle;" />
          Show reference frames
        </label>
      </div>
      <div class="section">
        <h3>Animation &amp; GIF Export</h3>
        <div class="gif-settings">
          <label>Frames</label>
          <input type="number" id="gifFrames" value="60" min="10" max="300" />
          <label>Delay (ms)</label>
          <input type="number" id="gifDelay" value="33" min="10" max="200" />
        </div>
        <button id="gifBtn" disabled>Record Deployment GIF</button>
        <div id="gifStatus"></div>
      </div>
      <div class="section">
        <h3>3D Export</h3>
        <button id="glbBtn" disabled style="width:100%;padding:10px 0;background:linear-gradient(135deg,#6a9f5b,#4a7f3b);color:#fff;border:none;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;font-family:inherit;transition:opacity 0.2s;">Export .GLB (current frame)</button>
        <div id="glbStatus" style="font-size:9px;color:#44bb44;margin-top:6px;min-height:14px;"></div>
        <div style="margin-top:8px;">
          <div class="gif-settings">
            <label>Frames</label>
            <input type="number" id="glbAnimFrames" value="30" min="2" max="120" style="width:50px;background:#090b10;border:1px solid #282b36;border-radius:3px;color:#6a9f5b;font-size:10px;font-family:inherit;padding:2px 4px;text-align:right;" />
          </div>
          <button id="glbAnimBtn" disabled style="width:100%;padding:10px 0;background:linear-gradient(135deg,#5b8f9f,#3b6f7f);color:#fff;border:none;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;font-family:inherit;transition:opacity 0.2s;margin-top:6px;">Export Animation .ZIP</button>
          <div id="glbAnimStatus" style="font-size:9px;color:#44bb44;margin-top:6px;min-height:14px;"></div>
        </div>
      </div>
      <div class="section">
        <h3>Legend</h3>
        <div class="legend-row"><span class="c" style="color:#2255cc">&mdash;</span> long arms</div>
        <div class="legend-row"><span class="c" style="color:#1d8c36">&mdash;</span> short members</div>
        <div class="legend-row"><span class="c" style="color:#c4164a">&mdash;</span> offset (last cell)</div>
        <div class="legend-row"><span class="c" style="color:#999">&mdash;</span> hinge (grey)</div>
        <div class="legend-row"><span class="c" style="color:#44bb44">&mdash;</span> short hinges</div>
        <div class="legend-row"><span class="c" style="color:#cc2255">&mdash;</span> longerons</div>
        <div style="font-size:8px;color:#444;margin-top:6px;">L-drag: orbit &middot; R-drag: pan &middot; Scroll: zoom</div>
      </div>
      <div class="section">
        <h3>Export to CAD CSV (rounded)</h3>
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

    // ── Dark mode for plot ──
    Object.assign(fig.layout, {{
      paper_bgcolor: "#090b10",
      plot_bgcolor: "#0d0f14",
      font: {{ color: "#ccc", family: "'JetBrains Mono', monospace" }},
    }});
    if (fig.layout.xaxis) Object.assign(fig.layout.xaxis, {{ gridcolor: "#1f222b", zerolinecolor: "#1f222b", color: "#888" }});
    if (fig.layout.yaxis) Object.assign(fig.layout.yaxis, {{ gridcolor: "#1f222b", zerolinecolor: "#1f222b", color: "#888" }});
    if (fig.layout.legend) Object.assign(fig.layout.legend, {{ bgcolor: "rgba(0,0,0,0)", font: {{ color: "#999" }} }});

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

      document.getElementById("jsonPanel").textContent = JSON.stringify(payload, null, 2);

      const ov = document.getElementById("viewerOverlay");
      ov.style.display = "block";
      document.getElementById("overlayContent").innerHTML =
        '<div class="kv"><span class="k">long</span><span class="v">' + currentSol.long.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">short</span><span class="v">' + currentSol.short.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">offset</span><span class="v">' + currentSol.offset.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">hinge_par</span><span class="v">' + currentSol.hinge_par.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">n_cells</span><span class="v">' + currentSol.n_cells + '</span></div>' +
        '<div class="kv"><span class="k">&alpha; deploy</span><span class="v">' + (currentSol.alpha_deploy * 180/Math.PI).toFixed(1) + '&deg;</span></div>' +
        '<div class="kv"><span class="k">thickness</span><span class="v">' + currentSol.thickness.toFixed(3) + '</span></div>';

      document.getElementById("cellsSlider").max = currentSol.n_cells;
      if (parseInt(document.getElementById("cellsSlider").value) > currentSol.n_cells) {{
        document.getElementById("cellsSlider").value = Math.min(2, currentSol.n_cells);
      }}

      if (!viewerReady) {{
        document.getElementById("placeholder3d").style.display = "none";
        window.viewerInit(document.getElementById("viewerMount"));
        viewerReady = true;
      }}

      // Enable GIF + GLB buttons
      document.getElementById("gifBtn").disabled = false;
      document.getElementById("glbBtn").disabled = false;
      document.getElementById("glbAnimBtn").disabled = false;

      rebuildFromUI();
      updateCsvPreview();
    }});

    function rebuildFromUI() {{
      if (!currentSol) return;
      const alphaFrac = parseFloat(document.getElementById("alphaSlider").value);
      const thickness = parseFloat(document.getElementById("thicknessSlider").value);
      const hingePer = parseFloat(document.getElementById("hingePerSlider").value);
      const showCells = parseInt(document.getElementById("cellsSlider").value);
      const showRefFrames = document.getElementById("refFrameToggle").checked;
      window.viewerRebuild(currentSol, alphaFrac, showCells, thickness, hingePer, showRefFrames);
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
    document.getElementById("refFrameToggle").addEventListener("change", function() {{
      rebuildFromUI();
    }});

    // ── GLB Export ──
    document.getElementById("glbBtn").addEventListener("click", function() {{
      var viewerState = window.getViewerState();
      if (!viewerState || !viewerState.linkage) return;
      const statusEl = document.getElementById("glbStatus");
      statusEl.textContent = "Exporting...";
      statusEl.style.color = "#44bb44";
      const exporter = new THREE.GLTFExporter();
      exporter.parse(viewerState.linkage, function(result) {{
        const blob = new Blob([result], {{ type: "application/octet-stream" }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "scissor_linkage.glb";
        a.click();
        URL.revokeObjectURL(url);
        statusEl.textContent = "Downloaded scissor_linkage.glb";
        setTimeout(function() {{ statusEl.textContent = ""; }}, 3000);
      }}, {{ binary: true }});
    }});

    // ── Animated GLB ZIP Export ──
    document.getElementById("glbAnimBtn").addEventListener("click", async function() {{
      if (!currentSol || !viewerReady) return;
      const btn = this;
      const statusEl = document.getElementById("glbAnimStatus");
      btn.disabled = true;
      btn.textContent = "Exporting...";
      statusEl.textContent = "";
      statusEl.style.color = "#44bb44";

      const nFrames = parseInt(document.getElementById("glbAnimFrames").value) || 30;
      const thickness = parseFloat(document.getElementById("thicknessSlider").value);
      const hingePer = parseFloat(document.getElementById("hingePerSlider").value);
      const showCells = parseInt(document.getElementById("cellsSlider").value);
      const showRefFrames = document.getElementById("refFrameToggle").checked;
      const exporter = new THREE.GLTFExporter();
      const zip = new JSZip();

      function exportFrame(frac) {{
        return new Promise(function(resolve) {{
          window.viewerRebuild(currentSol, frac, showCells, thickness, hingePer, showRefFrames);
          var vs = window.getViewerState();
          exporter.parse(vs.linkage, function(result) {{
            resolve(result);
          }}, {{ binary: true }});
        }});
      }}

      try {{
        for (let i = 0; i < nFrames; i++) {{
          const frac = i / (nFrames - 1);
          statusEl.textContent = "Frame " + (i + 1) + " / " + nFrames;
          const glb = await exportFrame(frac);
          const padded = String(i).padStart(4, "0");
          zip.file("frame_" + padded + ".glb", glb);
        }}

        // Add Blender import script
        const blenderScript = `import bpy
import os
import glob

# ── Scissor Linkage Animation Importer ──
# Run this script in Blender's Scripting workspace.
# Set FOLDER to the directory where you extracted the .glb frames.

FOLDER = r"/path/to/extracted/frames"  # <-- EDIT THIS
FPS = 24

# Find all frame GLB files
files = sorted(glob.glob(os.path.join(FOLDER, "frame_*.glb")))
if not files:
    raise FileNotFoundError(f"No frame_*.glb files found in {{FOLDER}}")

n_frames = len(files)
print(f"Importing {{n_frames}} frames...")

bpy.context.scene.render.fps = FPS
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = n_frames

# Import each frame, hide on non-active frames
collections = []
for i, filepath in enumerate(files):
    frame_num = i + 1
    col_name = f"Frame_{{frame_num:04d}}"
    col = bpy.data.collections.new(col_name)
    bpy.context.scene.collection.children.link(col)

    # Set active collection for import
    layer_collection = bpy.context.view_layer.layer_collection.children[col_name]
    bpy.context.view_layer.active_layer_collection = layer_collection

    bpy.ops.import_scene.gltf(filepath=filepath)

    # Move imported objects to our collection
    for obj in bpy.context.selected_objects:
        for old_col in obj.users_collection:
            old_col.objects.unlink(obj)
        col.objects.link(obj)

    collections.append(col)

# Keyframe visibility: each collection visible only on its frame
for i, col in enumerate(collections):
    frame_num = i + 1
    for obj in col.all_objects:
        # Hide on all frames by default
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=1)
        obj.keyframe_insert("hide_render", frame=1)

        # Show on this frame
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert("hide_viewport", frame=frame_num)
        obj.keyframe_insert("hide_render", frame=frame_num)

        # Hide again on next frame
        if frame_num < n_frames:
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert("hide_viewport", frame=frame_num + 1)
            obj.keyframe_insert("hide_render", frame=frame_num + 1)

        # Set interpolation to constant (step) for hide properties
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'

bpy.context.scene.frame_set(1)
print(f"Done! {{n_frames}} frames imported. Press Space to play.")
`;
        zip.file("import_animation.py", blenderScript);

        statusEl.textContent = "Compressing ZIP...";
        const blob = await zip.generateAsync({{ type: "blob" }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "scissor_animation.zip";
        a.click();
        URL.revokeObjectURL(url);
        statusEl.textContent = "Downloaded scissor_animation.zip (" + nFrames + " frames)";
      }} catch (err) {{
        statusEl.textContent = "Error: " + err.message;
        statusEl.style.color = "#f87171";
      }}

      // Restore current slider position
      rebuildFromUI();
      btn.disabled = false;
      btn.textContent = "Export Animation .ZIP";
    }});

    // ── GIF Recording ──
    let gifRecording = false;
    let gifWorkerBlobUrl = null;

    // Pre-fetch the gif.js worker source and create a blob URL to avoid cross-origin Worker restriction
    async function ensureGifWorker() {{
      if (gifWorkerBlobUrl) return gifWorkerBlobUrl;
      const resp = await fetch("https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js");
      const text = await resp.text();
      const blob = new Blob([text], {{ type: "application/javascript" }});
      gifWorkerBlobUrl = URL.createObjectURL(blob);
      return gifWorkerBlobUrl;
    }}

    // Also need the gif.js library itself
    async function loadGifJs() {{
      if (window.GIF) return;
      return new Promise((resolve, reject) => {{
        const s = document.createElement("script");
        s.src = "https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js";
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
      }});
    }}

    document.getElementById("gifBtn").addEventListener("click", async function() {{
      if (!currentSol || !viewerReady || gifRecording) return;
      gifRecording = true;
      const btn = this;
      const statusEl = document.getElementById("gifStatus");
      btn.disabled = true;
      btn.textContent = "Recording...";
      statusEl.textContent = "Loading encoder...";
      statusEl.style.color = "#88aacc";
      statusEl.style.opacity = 1;

      try {{
        await loadGifJs();
        const workerUrl = await ensureGifWorker();

        const nFrames = parseInt(document.getElementById("gifFrames").value) || 60;
        const delay = parseInt(document.getElementById("gifDelay").value) || 33;
        const thickness = parseFloat(document.getElementById("thicknessSlider").value);
        const hingePer = parseFloat(document.getElementById("hingePerSlider").value);
        const showCells = parseInt(document.getElementById("cellsSlider").value);
        const showRefFrames = document.getElementById("refFrameToggle").checked;

        // Collect PNG frames
        const frames = [];
        for (let f = 0; f <= nFrames; f++) {{
          const frac = f / nFrames;
          window.viewerRebuild(currentSol, frac, showCells, thickness, hingePer, showRefFrames);
          const dataUrl = window.viewerCaptureFrame();
          if (dataUrl) frames.push(dataUrl);
          statusEl.textContent = "Capturing frame " + (f + 1) + " / " + (nFrames + 1);
          await new Promise(r => setTimeout(r, 10));
        }}

        statusEl.textContent = "Encoding GIF (" + frames.length + " frames)...";

        const firstImg = new Image();
        await new Promise((resolve) => {{
          firstImg.onload = resolve;
          firstImg.src = frames[0];
        }});
        const W = firstImg.naturalWidth;
        const H = firstImg.naturalHeight;

        const gif = new GIF({{
          workers: 2,
          workerScript: workerUrl,
          width: W,
          height: H,
          transparent: 0x000000,
          quality: 10,
        }});

        for (let i = 0; i < frames.length; i++) {{
          const img = new Image();
          await new Promise((resolve) => {{
            img.onload = resolve;
            img.src = frames[i];
          }});
          const cvs = document.createElement("canvas");
          cvs.width = W; cvs.height = H;
          const ctx = cvs.getContext("2d");
          ctx.drawImage(img, 0, 0);
          gif.addFrame(ctx, {{ copy: true, delay: delay }});
        }}

        gif.on("finished", function(blob) {{
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "scissor_deployment.gif";
          a.style.display = "none";
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

          statusEl.textContent = "\\u2713 Exported scissor_deployment.gif (" + frames.length + " frames, transparent background)";
          statusEl.style.color = "#44bb44";
          btn.textContent = "Record Deployment GIF";
          btn.disabled = false;
          gifRecording = false;

          rebuildFromUI();
        }});

        gif.on("progress", function(p) {{
          statusEl.textContent = "Encoding GIF... " + Math.round(p * 100) + "%";
        }});

        gif.render();
      }} catch (err) {{
        statusEl.textContent = "\\u2716 GIF export failed: " + err.message;
        statusEl.style.color = "#f87171";
        btn.textContent = "Record Deployment GIF";
        btn.disabled = false;
        gifRecording = false;
      }}
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
        statusEl.innerHTML = '<span style="color:#44bb44;">\\u2713 Max rounding error: ' + maxErr.toFixed(6) + ' mm \\u2014 negligible</span>';
      }} else if (maxErr < 0.1) {{
        statusEl.innerHTML = '<span style="color:#c8b88a;">\\u26A0 Max rounding error: ' + maxErr.toFixed(4) + ' mm \\u2014 small</span>';
      }} else {{
        statusEl.innerHTML = '<span style="color:#f87171;">\\u26A0 Max rounding error: ' + maxErr.toFixed(4) + ' mm \\u2014 review constraints below</span>';
      }}
      exportBtn.disabled = false;

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
      status.textContent = "\\u2713 Exported scissor_parameters.csv (rounded values)";
      status.style.opacity = 1;
      setTimeout(() => {{ status.style.opacity = 0; }}, 4000);
    }});

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