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
        m.width_limit = pyo.Constraint(expr=2 * m.b[1] + 4 * m.hinge_par <= stow_width)

    if stow_depth is not None:
        m.stow_depth_limit = pyo.Constraint(expr=m.length[1] == stow_depth)

    if expanded_depth is not None:
        m.expanded_depth_target = pyo.Constraint(expr=m.length[n_states] == expanded_depth)

    m.initial_state = pyo.Constraint(expr=m.a[1] == m.b[1] + m.hinge_par)
    m.final_state   = pyo.Constraint(expr=2.0 * m.a[n_states] == m.b[n_states])

    m.initial_angle_beta = pyo.Constraint(expr=m.sb[1] == (m.thickness / 2.0) / m.short)

    m.objective = pyo.Objective(expr=m.length[1], sense=pyo.minimize)
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
    # import pdb; pdb.set_trace()
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

            "extension_ratio_length": float(value(model.length[n_states]) / (value(model.thickness)*value(model.n))),
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


def sweep_offsets_and_thicknesses(args, progress_callback=None):
    offsets = list(np.linspace(args.offset_min, args.offset_max, args.offset_steps))
    thicknesses = parse_thicknesses(args)
    total = len(thicknesses) * len(offsets)
    done = 0
    #print args 
    print(args)
    feasible = []
    for t in thicknesses:
      for off in offsets:
          try:
        
              r = solve_one(args.stow_width, args.stow_depth, args.expanded_depth, args.states, args.hinge_par, float(t), args.n, args.short, args.long, float(off), tee=args.tee, debug_infeasible=False)
              # import pdb; pdb.set_trace()
              # Compute rounding residuals (no re-solve, just arithmetic)
              rounded = compute_rounding_residuals(r)
              r["rounded_solution"] = rounded
              feasible.append(r)
          except Exception:
              # import pdb; pdb.set_trace()
              pass
          done += 1
          if progress_callback:
              progress_callback(done, total, len(feasible))

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

  // Collected member endpoints from buildScene (for FEA)
  let _collectedMembers = [];
  let _collectEnabled = false;
  let _collectOffset = [0, 0, 0];  // group position offset for current cell

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

    if (_collectEnabled) {
      const ox = _collectOffset[0], oy = _collectOffset[1], oz = _collectOffset[2];
      _collectedMembers.push({
        a: [a.x + ox, a.y + oy, a.z + oz],
        b: [b.x + ox, b.y + oy, b.z + oz],
        mat: matKey,
        radius: radius
      });
    }
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
    _collectedMembers = [];
    _collectEnabled = true;
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
      _collectOffset = [0, -i * c, 0];

      // ── RIGHT HALF (+X) ──
      const rOup = new THREE.Vector3(0, 0, +T);
      const rOdn = new THREE.Vector3(0, 0, -T);
      const rP1 = new THREE.Vector3(P1x, P1y, +T);
      const rP2 = new THREE.Vector3(P2x, P2y, -T);

      addTube(g, rOup, rOdn, MAT_ALU, Toff);  // rigid link: connect crossing point
      addTube(g, rOup, rP1, MAT_CARBON, T);
      addTube(g, rOdn, rP2, MAT_CARBON, T);

      const rP1_up = new THREE.Vector3(P1x, P1y, +2*T);
      addTube(g, rP1, rP1_up, MAT_ALU, Toff);  // explicit rigid link
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
      addTube(g, rSH_par, rSH_par_up, MAT_ALU, Toff);  // rigid link
      const rShortDn = shortApex(rSH_par_up, planeAngle, halfBeta, S, -1);
      const rSH_par_up_up = new THREE.Vector3(rSH_par_up.x, rSH_par_up.y, rSH_par_up.z + T);
      addTube(g, rSH_par_up, rSH_par_up_up, MAT_ALU, Toff);  // rigid link
      const rShortUp = shortApex(rSH_par_up_up, planeAngle, halfBeta, S, +1);
      addTube(g, rSH_par_up_up, rShortUp, MAT_CARBON_SHORT, Toff);
      addBall(g, rShortUp, MAT_CARBON_SHORT, Jrs);
      addBall(g, rSH_par_up_up, MAT_CARBON_SHORT, Jrs);

      if (i > 0) {
        addTube(g, rSH_par_up, rShortDn, MAT_CARBON_SHORT, Toff);
        addBall(g, rSH_par_up, MAT_CARBON_SHORT, Jrs);
        addBall(g, rShortDn, MAT_CARBON_SHORT, Jrs);
        const rShortDn_dn = new THREE.Vector3(rShortDn.x, rShortDn.y, rShortDn.z - T);
        addTube(g, rShortDn, rShortDn_dn, MAT_ALU, Toff);  // rigid link
        addShortDnHinge(g, rShortDn_dn, planeAngle, hingePar, -hingePer, MAT_ALU_GREEN, Toff, Jrs);
        addBall(g, rShortDn_dn, MAT_ALU_GREEN, Jrs);
      }

      if (i === cellsToShow - 1) {
        const rP3 = new THREE.Vector3(P3x, P3y, +T);
        const rP3_up = new THREE.Vector3(P3x, P3y, +2*T);
        const rP2_up = new THREE.Vector3(P2x, P2y, +T);
        addTube(g, rP2, rP2_up, MAT_ALU, Toff);  // rigid link P2→P2_up
        addTube(g, rP2_up, rP3, MAT_CARBON_OFF, Toff);
        addTube(g, rP3, rP3_up, MAT_ALU, Toff);  // rigid link P3→P3_up
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
        addTube(g, rSH3_par, rSH3_par_up, MAT_ALU, Toff);  // rigid link
        const rSh3Dn = shortApex(rSH3_par_up, planeAngle, halfBeta, S, -1);
        addTube(g, rSH3_par_up, rSh3Dn, MAT_CARBON_SHORT, Toff);
        addBall(g, rSH3_par_up, MAT_CARBON_SHORT, Jrs);
        addBall(g, rSh3Dn, MAT_CARBON_SHORT, Jrs);
        const rSh3Dn_up = new THREE.Vector3(rSh3Dn.x, rSh3Dn.y, rSh3Dn.z - T);
        addTube(g, rSh3Dn, rSh3Dn_up, MAT_ALU, Toff);  // rigid link
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
        addTube(g, rShortDn, rLongDn_ref, MAT_ALU, Toff);  // rigid link
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
      addTube(g, lSH_par, lSH_par_up, MAT_ALU, Toff);  // rigid link
      const lShortDn = shortApex(lSH_par_up, leftPlaneAngle, halfBeta, S, -1);
      const lSH_par_up_up = new THREE.Vector3(lSH_par_up.x, lSH_par_up.y, lSH_par_up.z + T);
      addTube(g, lSH_par_up, lSH_par_up_up, MAT_ALU, Toff);  // rigid link
      const lShortUp = shortApex(lSH_par_up_up, leftPlaneAngle, halfBeta, S, +1);
      addTube(g, lSH_par_up_up, lShortUp, MAT_CARBON_SHORT, Toff);
      addBall(g, lShortUp, MAT_CARBON_SHORT, Jrs);
      addBall(g, lSH_par_up_up, MAT_CARBON_SHORT, Jrs);

      if (i > 0) {
        addTube(g, lSH_par_up, lShortDn, MAT_CARBON_SHORT, Toff);
        addBall(g, lSH_par_up, MAT_CARBON_SHORT, Jrs);
        const lShortDn_dn = new THREE.Vector3(lShortDn.x, lShortDn.y, lShortDn.z - T);
        addTube(g, lShortDn, lShortDn_dn, MAT_ALU, Toff);  // rigid link
        addShortDnHinge(g, lShortDn_dn, leftPlaneAngle, hingePar, hingePer, MAT_ALU_GREEN, Toff, Jrs);
        addBall(g, lShortDn_dn, MAT_ALU_GREEN, Jrs);
        addBall(g, lShortDn, MAT_CARBON_SHORT, Jrs);
      }

      if (i === cellsToShow - 1) {
        const lP3 = new THREE.Vector3(-P3x, P3y, -T);
        const lP2_dn = new THREE.Vector3(-P2x, P2y, -T);
        addTube(g, lP2, lP2_dn, MAT_ALU, Toff);  // rigid link
        addBall(g, lP2_dn, MAT_CARBON_OFF, Jrs);
        addTube(g, lP2_dn, lP3, MAT_CARBON_OFF, Toff);
        const lP3_up = new THREE.Vector3(-P3x, P3y, +2*T);
        addTube(g, lP3, lP3_up, MAT_ALU, Toff);  // rigid link
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
        addTube(g, lSH3_par, lSH3_par_up, MAT_ALU, Toff);  // rigid link
        const lSh3Dn = shortApex(lSH3_par_up, leftPlaneAngle, halfBeta, S, -1);
        addTube(g, lSH3_par_up, lSh3Dn, MAT_CARBON_SHORT, Toff);
        addBall(g, lSH3_par_up, MAT_CARBON_SHORT, Jrs);
        addBall(g, lSh3Dn, MAT_CARBON_SHORT, Jrs);
        const lSh3Dn_up = new THREE.Vector3(lSh3Dn.x, lSh3Dn.y, lSh3Dn.z - T);
        addTube(g, lSh3Dn, lSh3Dn_up, MAT_ALU, Toff);  // rigid link
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
        addTube(g, lShortDn, lLongDn_ref, MAT_ALU, Toff);  // rigid link
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
    _collectOffset = [0, 0, 0];  // reset for non-cell members (struts, etc.)

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
      panel.position.set(0, P1y - panelDepth / 2, -0.06);
      // Rotate to lie flat in XY plane (default PlaneGeometry is in XY, we want XY)
      root.add(panel);

      // Grid lines on panel for solar cell look
      const gridMat = new THREE.LineBasicMaterial({ color: 0x2266dd, transparent: true, opacity: 0.25 });
      const nGridX = Math.max(2, Math.round(panelWidth / (T * 20)));
      const nGridY = Math.max(2, Math.round(panelDepth / (T * 20)));
      for (let gx = 0; gx <= nGridX; gx++) {
        const x = -panelWidth/2 + gx * panelWidth / nGridX;
        const pts = [new THREE.Vector3(x, P1y, -0.061), new THREE.Vector3(x, P1y - panelDepth, 0.001)];
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

    _collectEnabled = false;
    return { group: root, c, theta, halfBeta, cellsToShow, members: _collectedMembers.slice() };
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
    const bld = buildScene(
      sol.long, sol.offset, alpha, sol.n_cells,
      thickness, sol.hinge_par, hingePer,
      sol.short, sol.hinge_par, showCells, longeronLen, showRefFrames,
      sol.width_actual || 0
    );
    const { group, c } = bld;
    s.linkage = group;
    s.sceneMembers = bld.members;  // store for FEA
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

  // ── Keypoint overlay ──
  let kpGroup = null;
  const KP_COLORS = {
    1:  0x2ca02c,  // green (short / layer 1)
    '-1': 0xd62728, // red (layer -1)
    2:  0x1f77b4,  // blue (long)
    3:  0xd62728,  // red (offset)
    0:  0x888888,
  };

  function buildKeypointGroup(kps, tubeRadius) {
    const grp = new THREE.Group();
    for (const kp of kps) {
      const p1 = new THREE.Vector3(kp[0], -kp[1], kp[2]);
      const p2 = new THREE.Vector3(kp[3], -kp[4], kp[5]);
      const layer = kp[8] !== undefined ? kp[8] : 0;
      const color = KP_COLORS[layer] || KP_COLORS[String(layer)] || 0x888888;
      const dir = new THREE.Vector3().subVectors(p2, p1);
      const len = dir.length();
      if (len < 1e-6) continue;
      const geo = new THREE.CylinderGeometry(tubeRadius, tubeRadius, len, 10);
      const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.4, roughness: 0.5, transparent: true, opacity: 0.85 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.addVectors(p1, p2).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
      grp.add(mesh);
      // Joint spheres
      const ballGeo = new THREE.SphereGeometry(tubeRadius * 1.3, 8, 8);
      const ballMat = new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 0.3, roughness: 0.6 });
      const b1 = new THREE.Mesh(ballGeo, ballMat); b1.position.copy(p1); grp.add(b1);
      const b2 = new THREE.Mesh(ballGeo.clone(), ballMat.clone()); b2.position.copy(p2); grp.add(b2);
    }
    return grp;
  }

  function showKeypoints(kps, tubeRadius) {
    if (!viewerState) return;
    clearKeypoints();
    kpGroup = buildKeypointGroup(kps, tubeRadius || 0.003);
    viewerState.scene.add(kpGroup);
  }

  function clearKeypoints() {
    if (!viewerState || !kpGroup) return;
    viewerState.scene.remove(kpGroup);
    disposeGroup(kpGroup);
    kpGroup = null;
  }

  window.viewerInit = initViewer;
  window.viewerRebuild = rebuildLinkage;
  window.viewerCaptureFrame = captureFrame;
  window.getViewerState = function() { return viewerState; };
  window.viewerShowKeypoints = showKeypoints;
  window.viewerClearKeypoints = clearKeypoints;
})();
"""


def make_interactive_plot(results, out_html="scissor_sweep.html",
                          actual_axes=("depth_actual", "height_actual"),
                          initial_mode="ratio"):
    if pd is None:
        raise RuntimeError(
            "pandas is required for interactive plots.\n"
            "Install with: pip install pandas"
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

    payloads_json = json.dumps(payloads)
    rows_json = df.to_json(orient='records')

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
    #plotArea {{ display: flex; flex-direction: column; }}
    #plotDiv {{ width: 100%; flex: 1; min-height: 0; }}
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
    .fea-field {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }}
    .fea-field label {{
      font-size: 10px;
      color: #888;
    }}
    .fea-input {{
      width: 110px;
      background: #090b10;
      border: 1px solid #282b36;
      border-radius: 3px;
      color: #e74c3c;
      font-size: 10px;
      font-family: inherit;
      padding: 3px 6px;
      text-align: right;
    }}
    .fea-input:focus {{ outline: none; border-color: #e74c3c; }}
    select.fea-input {{ text-align: left; cursor: pointer; }}
    .fea-result-header {{
      font-size: 8px;
      color: #e74c3c;
      text-transform: uppercase;
      letter-spacing: 2px;
      font-weight: 600;
      margin-bottom: 6px;
      padding-bottom: 4px;
      border-bottom: 1px solid #2a1515;
    }}
    #feaResultsTable td {{
      padding: 2px 4px;
      border-bottom: 1px solid #1a1a1a;
    }}
    #feaResultsTable td:first-child {{ color: #888; }}
    #feaResultsTable td:last-child {{ color: #ddd; text-align: right; font-variant-numeric: tabular-nums; }}
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
    .plot-controls {{
      display: flex;
      gap: 12px;
      padding: 10px 14px;
      background: #0d0f14;
      border-bottom: 1px solid #1f222b;
      align-items: center;
      flex-wrap: wrap;
    }}
    .plot-controls .ctrl-group {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .plot-controls label {{
      font-size: 10px;
      color: #666;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      white-space: nowrap;
    }}
    .plot-controls select {{
      background: #12141c;
      border: 1px solid #1f222b;
      border-radius: 4px;
      color: #e0e0e0;
      padding: 5px 8px;
      font-size: 12px;
      font-family: inherit;
      cursor: pointer;
      min-width: 140px;
    }}
    .plot-controls select:focus {{
      outline: none;
      border-color: #3a7bd5;
    }}
    .plot-controls select:hover {{
      border-color: #2a3a5a;
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="plotArea">
      <div class="plot-controls">
        <div class="ctrl-group">
          <label for="xSelect">X Axis</label>
          <select id="xSelect">
            <option value="er_len">Extension ratio (length)</option>
            <option value="er_hgt">Extension ratio (height)</option>
            <option value="depth_actual">Deployed depth (m)</option>
            <option value="height_actual">Deployed height (m)</option>
            <option value="width_actual">Stowed width (m)</option>
            <option value="thickness">Thickness (m)</option>
            <option value="offset">Offset (m)</option>
            <option value="n_cells">Number of cells</option>
            <option value="long">Long link (m)</option>
            <option value="short">Short link (m)</option>
            <option value="hinge_par">Hinge parameter (m)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label for="ySelect">Y Axis</label>
          <select id="ySelect">
            <option value="er_hgt">Extension ratio (height)</option>
            <option value="er_len">Extension ratio (length)</option>
            <option value="depth_actual">Deployed depth (m)</option>
            <option value="height_actual">Deployed height (m)</option>
            <option value="width_actual">Stowed width (m)</option>
            <option value="thickness">Thickness (m)</option>
            <option value="offset">Offset (m)</option>
            <option value="n_cells">Number of cells</option>
            <option value="long">Long link (m)</option>
            <option value="short">Short link (m)</option>
            <option value="hinge_par">Hinge parameter (m)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label for="seriesSelect">Color By</label>
          <select id="seriesSelect">
            <option value="thickness">Thickness</option>
            <option value="offset">Offset</option>
            <option value="n_cells">Number of cells</option>
            <option value="long">Long link</option>
            <option value="short">Short link</option>
            <option value="hinge_par">Hinge parameter</option>
            <option value="er_len">ER (length)</option>
            <option value="er_hgt">ER (height)</option>
            <option value="depth_actual">Deployed depth</option>
            <option value="height_actual">Deployed height</option>
            <option value="width_actual">Stowed width</option>
            <option value="none">None (single color)</option>
          </select>
        </div>
      </div>
      <div id="plotDiv"></div>
    </div>
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
          <span id="thicknessVal">0.05</span>
        </div>
        <input type="range" id="thicknessSlider" min="0.001" max="0.100" step="0.0005" value="0.006" />
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
        <h3>Export Keypoints CSV</h3>
        <div class="csv-field">
          <label>Cross Width</label>
          <div><input type="number" id="kpCrossWidth" value="0.1" step="0.01" min="0.001" /> <span style="font-size:9px;color:#555">m</span></div>
        </div>
        <div class="csv-field">
          <label>Cross Height</label>
          <div><input type="number" id="kpCrossHeight" value="0.1" step="0.01" min="0.001" /> <span style="font-size:9px;color:#555">m</span></div>
        </div>
        <div style="margin:8px 0 6px 0;">
          <label style="font-size:10px;color:#888;cursor:pointer;">
            <input type="checkbox" id="showPetKp" style="vertical-align:middle;" />
            Show PET keypoints in 3D
          </label>
        </div>
        <div style="margin-bottom:8px;">
          <label style="font-size:10px;color:#888;cursor:pointer;">
            <input type="checkbox" id="showDartKp" style="vertical-align:middle;" />
            Show DART keypoints in 3D
          </label>
        </div>
        <button id="exportPetBtn" disabled style="width:100%;padding:10px 0;background:linear-gradient(135deg,#9b59b6,#8e44ad);color:#fff;border:none;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;font-family:inherit;transition:opacity 0.2s;">Export PET Keypoints CSV</button>
        <div style="margin-top:6px;">
          <button id="exportDartBtn" disabled style="width:100%;padding:10px 0;background:linear-gradient(135deg,#e67e22,#d35400);color:#fff;border:none;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;font-family:inherit;transition:opacity 0.2s;">Export DART Keypoints CSV</button>
        </div>
        <div id="kpExportStatus" style="font-size:9px;color:#44bb44;margin-top:6px;min-height:14px;transition:opacity 0.3s;"></div>
      </div>
      <div class="section" id="feaSection">
        <h3>Beam FEA</h3>
        <div id="feaConfigPanel">
          <div class="fea-field">
            <label>Material</label>
            <select id="feaMaterial" class="fea-input">
              <option value="aluminum_6061">Al 6061-T6</option>
              <option value="stainless_304">SS 304</option>
              <option value="carbon_fiber">CFRP</option>
            </select>
          </div>
          <div class="fea-field">
            <label>Profile</label>
            <select id="feaProfile" class="fea-input">
              <option value="square">Square</option>
              <option value="circle">Circle</option>
              <option value="rectangle">Rectangle</option>
            </select>
          </div>
          <div class="fea-field">
            <label>Hollow</label>
            <input type="checkbox" id="feaHollow" checked style="accent-color:#e74c3c;" />
          </div>
          <div class="fea-field">
            <label>Outer dim (mm)</label>
            <input type="number" id="feaDim" class="fea-input" value="10" step="0.5" min="1" />
          </div>
          <div class="fea-field">
            <label>Wall (mm)</label>
            <input type="number" id="feaWall" class="fea-input" value="1" step="0.1" min="0.1" />
          </div>
          <div class="fea-field">
            <label>Tip load (N)</label>
            <input type="number" id="feaLoad" class="fea-input" value="100" step="10" min="0.1" />
          </div>
          <div class="fea-field">
            <label>Load dir</label>
            <select id="feaLoadDir" class="fea-input">
              <option value="X">X (lateral)</option>
              <option value="Y" selected>Y (along deploy)</option>
              <option value="Z">Z (out of plane)</option>
            </select>
          </div>
          <div style="margin-top:6px;padding-top:6px;border-top:1px solid #1c1e28;">
            <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Boundary Conditions</div>
            <div class="fea-field">
              <label>Fix</label>
              <select id="feaFixMode" class="fea-input">
                <option value="base">Base (cell 0)</option>
                <option value="tip">Tip (last cell)</option>
                <option value="both">Both ends</option>
                <option value="base_wide">Base (2 cells)</option>
                <option value="custom">Custom Y range</option>
              </select>
            </div>
            <div id="feaFixCustomRow" style="display:none;">
              <div class="fea-field">
                <label>Fix Y from</label>
                <input type="number" id="feaFixYMin" class="fea-input" value="" step="0.1" placeholder="min" />
              </div>
              <div class="fea-field">
                <label>Fix Y to</label>
                <input type="number" id="feaFixYMax" class="fea-input" value="" step="0.1" placeholder="max" />
              </div>
            </div>
            <div class="fea-field">
              <label>Load at</label>
              <select id="feaLoadAt" class="fea-input">
                <option value="auto">Opposite end</option>
                <option value="base">Base (cell 0)</option>
                <option value="tip">Tip (last cell)</option>
                <option value="mid">Mid-span</option>
              </select>
            </div>
            <div class="fea-field">
              <label>Fix DOFs</label>
              <select id="feaFixDofs" class="fea-input">
                <option value="all">All 6 (rigid)</option>
                <option value="pin">XYZ only (pin)</option>
              </select>
            </div>
          </div>
        </div>
        <button id="feaRunBtn" disabled style="width:100%;margin-top:8px;padding:10px 0;background:linear-gradient(135deg,#e74c3c,#c0392b);color:#fff;border:none;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;font-family:inherit;transition:opacity 0.2s;">Run FEA</button>
        <div id="feaStatus" style="font-size:9px;color:#aaa;margin-top:6px;min-height:14px;"></div>
        <div id="feaResults" style="display:none;margin-top:10px;">
          <div class="fea-result-header">Results</div>
          <table id="feaResultsTable" style="width:100%;border-collapse:collapse;font-size:10px;">
            <tbody id="feaResultsBody"></tbody>
          </table>
          <div id="feaFreqChart" style="margin-top:8px;height:120px;"></div>
          <div style="margin-top:6px;">
            <div class="fea-field">
              <label>Show mode</label>
              <select id="feaModeSelect" class="fea-input" style="width:80px;">
                <option value="-1">None</option>
              </select>
            </div>
            <div class="fea-field">
              <label>Scale</label>
              <input type="range" id="feaModeScale" min="0.1" max="5" step="0.1" value="1" style="width:80px;accent-color:#e74c3c;" />
              <span id="feaModeScaleVal" style="font-size:9px;color:#888;margin-left:4px;">1.0x</span>
            </div>
          </div>
        </div>
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
    const ALL_ROWS = {rows_json};

    const FIELD_LABELS = {{
      er_len: "Extension ratio (length)",
      er_hgt: "Extension ratio (height)",
      depth_actual: "Deployed depth [m]",
      height_actual: "Deployed height [m]",
      width_actual: "Stowed width [m]",
      thickness: "Thickness [m]",
      offset: "Offset [m]",
      n_cells: "Number of cells",
      long: "Long link [m]",
      short: "Short link [m]",
      hinge_par: "Hinge parameter [m]",
    }};

    // ── State ──
    let currentSol = null;
    let viewerReady = false;

    // ── Dynamic plot rebuild ──
    function getSeriesGroups(rows, seriesKey) {{
      if (seriesKey === "none") {{
        const indices = rows.map((_, i) => i);
        return {{ "all": indices }};
      }}
      const vals = rows.map(r => r[seriesKey]);
      const uniqueVals = [...new Set(vals)].sort((a, b) => a - b);

      if (uniqueVals.length <= 20) {{
        const groups = {{}};
        rows.forEach((r, i) => {{
          const v = r[seriesKey];
          const key = typeof v === "number" ? Number(v.toPrecision(4)) : String(v);
          if (!groups[key]) groups[key] = [];
          groups[key].push(i);
        }});
        return groups;
      }}

      // Bucket into ~10 groups for continuous variables
      const nBuckets = 10;
      const min = uniqueVals[0], max = uniqueVals[uniqueVals.length - 1];
      const range = max - min;
      if (range === 0) {{
        const indices = rows.map((_, i) => i);
        return {{ [String(min)]: indices }};
      }}
      const step = range / nBuckets;
      const groups = {{}};
      rows.forEach((r, i) => {{
        const v = r[seriesKey];
        const bucket = Math.min(nBuckets - 1, Math.floor((v - min) / step));
        const lo = parseFloat((min + bucket * step).toPrecision(3));
        const hi = parseFloat((min + (bucket + 1) * step).toPrecision(3));
        const key = lo + "-" + hi;
        if (!groups[key]) groups[key] = [];
        groups[key].push(i);
      }});
      return groups;
    }}

    function rebuildPlot() {{
      const xKey = document.getElementById("xSelect").value;
      const yKey = document.getElementById("ySelect").value;
      const seriesKey = document.getElementById("seriesSelect").value;

      const groups = getSeriesGroups(ALL_ROWS, seriesKey);
      const sortedKeys = Object.keys(groups).sort((a, b) => {{
        const na = parseFloat(a), nb = parseFloat(b);
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return a.localeCompare(b);
      }});

      const traces = [];
      for (const key of sortedKeys) {{
        const indices = groups[key];
        const xs = indices.map(i => ALL_ROWS[i][xKey]);
        const ys = indices.map(i => ALL_ROWS[i][yKey]);
        const cd = indices.map(i => [
          ALL_ROWS[i].thickness,
          ALL_ROWS[i].offset,
          ALL_ROWS[i].n_cells,
          ALL_ROWS[i].long,
          ALL_ROWS[i].short,
          ALL_ROWS[i].hinge_par,
          i,  // payload index
        ]);

        const label = seriesKey === "none" ? "all" : key;
        traces.push({{
          x: xs,
          y: ys,
          mode: "markers",
          name: seriesKey === "none" ? "All points" : (seriesKey + "=" + label),
          customdata: cd,
          hovertemplate:
            FIELD_LABELS[xKey] + ": %{{x:.4f}}<br>" +
            FIELD_LABELS[yKey] + ": %{{y:.4f}}<br>" +
            "thickness=%{{customdata[0]:.3f}}<br>" +
            "offset=%{{customdata[1]:.5f}}<br>" +
            "n_cells=%{{customdata[2]}}<br>" +
            "long=%{{customdata[3]:.4f}}<br>" +
            "short=%{{customdata[4]:.4f}}<br>" +
            "hinge=%{{customdata[5]:.4f}}<extra></extra>",
          marker: {{ size: 8 }},
        }});
      }}

      const legendTitle = seriesKey === "none" ? "" : FIELD_LABELS[seriesKey] || seriesKey;

      const layout = {{
        title: "Scissor sweep — click a point to visualize in 3D",
        xaxis: {{
          title: FIELD_LABELS[xKey] || xKey,
          gridcolor: "#1f222b",
          zerolinecolor: "#1f222b",
          color: "#888",
        }},
        yaxis: {{
          title: FIELD_LABELS[yKey] || yKey,
          gridcolor: "#1f222b",
          zerolinecolor: "#1f222b",
          color: "#888",
        }},
        paper_bgcolor: "#090b10",
        plot_bgcolor: "#0d0f14",
        font: {{ color: "#ccc", family: "'JetBrains Mono', monospace" }},
        legend: {{
          title: {{ text: legendTitle }},
          bgcolor: "rgba(0,0,0,0)",
          font: {{ color: "#999" }},
        }},
        margin: {{ t: 50 }},
      }};

      Plotly.react("plotDiv", traces, layout, {{ responsive: true }});
    }}

    // ── Initial plot ──
    rebuildPlot();

    // ── Dropdown listeners ──
    document.getElementById("xSelect").addEventListener("change", rebuildPlot);
    document.getElementById("ySelect").addEventListener("change", rebuildPlot);
    document.getElementById("seriesSelect").addEventListener("change", rebuildPlot);

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
      const s = currentSol;
      const fmt = (v) => v !== undefined && v !== null ? v.toFixed(4) : "—";
      document.getElementById("overlayContent").innerHTML =
        '<div class="label" style="margin-top:4px;">Deployed</div>' +
        '<div class="kv"><span class="k">depth</span><span class="v">' + fmt(s.depth_actual) + ' m</span></div>' +
        '<div class="kv"><span class="k">height</span><span class="v">' + fmt(s.height_actual) + ' m</span></div>' +
        '<div class="kv"><span class="k">width</span><span class="v">' + fmt(s.width_actual) + ' m</span></div>' +
        '<div class="label" style="margin-top:6px;">Stowed</div>' +
        '<div class="kv"><span class="k">depth</span><span class="v">' + fmt(s.depth_stow) + ' m</span></div>' +
        '<div class="kv"><span class="k">width</span><span class="v">' + fmt(s.width_stow) + ' m</span></div>' +
        '<div class="kv"><span class="k">height</span><span class="v">' + fmt(s.height_stow) + ' m</span></div>' +
        '<div class="label" style="margin-top:6px;">Parameters</div>' +
        '<div class="kv"><span class="k">n_cells</span><span class="v">' + s.n_cells + '</span></div>' +
        '<div class="kv"><span class="k">thickness</span><span class="v">' + s.thickness.toFixed(3) + ' m</span></div>' +
        '<div class="kv"><span class="k">long</span><span class="v">' + s.long.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">short</span><span class="v">' + s.short.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">offset</span><span class="v">' + s.offset.toFixed(4) + '</span></div>' +
        '<div class="kv"><span class="k">hinge_par</span><span class="v">' + s.hinge_par.toFixed(4) + '</span></div>';

      document.getElementById("cellsSlider").max = currentSol.n_cells;
      if (parseInt(document.getElementById("cellsSlider").value) > currentSol.n_cells) {{
        document.getElementById("cellsSlider").value = Math.min(2, currentSol.n_cells);
      }}

      if (!viewerReady) {{
        document.getElementById("placeholder3d").style.display = "none";
        window.viewerInit(document.getElementById("viewerMount"));
        viewerReady = true;
      }}

      // Enable GIF + GLB + keypoint buttons
      document.getElementById("gifBtn").disabled = false;
      document.getElementById("glbBtn").disabled = false;
      document.getElementById("glbAnimBtn").disabled = false;
      document.getElementById("exportPetBtn").disabled = false;
      document.getElementById("exportDartBtn").disabled = false;
      document.getElementById("feaRunBtn").disabled = false;

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
      rebuildKeypointOverlay();
    }}

    function rebuildKeypointOverlay() {{
      if (!currentSol) {{ window.viewerClearKeypoints(); return; }}
      const showPet = document.getElementById("showPetKp").checked;
      const showDart = document.getElementById("showDartKp").checked;
      if (!showPet && !showDart) {{ window.viewerClearKeypoints(); return; }}
      const abt = getCurrentAlphaBetaTheta();
      if (!abt) return;
      const cw = parseFloat(document.getElementById("kpCrossWidth").value) || 0.1;
      const ch = parseFloat(document.getElementById("kpCrossHeight").value) || 0.1;
      const nCells = currentSol.n_cells;
      const tubeR = parseFloat(document.getElementById("thicknessSlider").value) * 0.5;
      const alpha_kp = Math.PI - abt.alpha;
      const beta_kp = Math.PI - abt.beta;
      let kps;
      if (showPet) {{
        const l1 = currentSol.short / 2;
        const l2 = currentSol.offset;
        const l3 = currentSol.long - currentSol.offset;
        kps = getPetKeypoints(l1, l2, l3, alpha_kp, beta_kp, abt.theta, nCells, cw, ch, false);
      }} else {{
        kps = getDartKeypoints(currentSol.short, currentSol.long, currentSol.offset, currentSol.hinge_par, alpha_kp, beta_kp, abt.theta, nCells, cw, ch, false);
      }}
      window.viewerShowKeypoints(kps, tubeR);
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

    // ── Keypoint overlay checkboxes (mutually exclusive) ──
    document.getElementById("showPetKp").addEventListener("change", function() {{
      if (this.checked) document.getElementById("showDartKp").checked = false;
      rebuildKeypointOverlay();
    }});
    document.getElementById("showDartKp").addEventListener("change", function() {{
      if (this.checked) document.getElementById("showPetKp").checked = false;
      rebuildKeypointOverlay();
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

    // ── Keypoint generation (ported from Python) ──
    function getCurrentAlphaBetaTheta() {{
      if (!currentSol) return null;
      const alphaFrac = parseFloat(document.getElementById("alphaSlider").value);
      const alpha = (currentSol.alpha_stow || 0) + alphaFrac * ((currentSol.alpha_deploy || 1) - (currentSol.alpha_stow || 0));
      const L = currentSol.long;
      const O = currentSol.offset;
      const S = currentSol.short;
      const h = currentSol.hinge_par;

      // compute beta and theta same as buildScene
      const ha = alpha / 2;
      const LmO = L - O;
      const c = 2 * LmO * Math.sin(ha);
      const sb = S > 1e-6 ? Math.min(1, Math.max(0, c / (2 * S))) : 0;
      const halfBeta = Math.asin(sb);
      const beta = 2 * halfBeta;
      const cb = Math.cos(halfBeta);
      const b = S * cb;
      const a = L * Math.cos(ha);
      const num = (2 * a + 2 * h) ** 2;
      const den = 2 * (b + 2 * h) ** 2;
      let cosTheta = den > 1e-14 ? 1 - num / den : 1;
      cosTheta = Math.max(-1, Math.min(1, cosTheta));
      const theta = Math.acos(cosTheta);
      return {{ alpha, beta, theta }};
    }}

    function matMul3(A, v) {{
      return [
        A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
        A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
        A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2]
      ];
    }}

    function rotZ3(angle) {{
      const c = Math.cos(angle), s = Math.sin(angle);
      return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
    }}

    function rotY3(angle) {{
      const c = Math.cos(angle), s = Math.sin(angle);
      return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
    }}

    function getScissorKeypoints(length, alpha, nCells, cw, ch, addEnds, addLongerons) {{
      const rotZ = rotZ3(alpha / 2);
      const rotZ_ = rotZ3(-alpha / 2);
      const kps = [];
      const front1Verts = [];
      const front2Verts = [];
      for (let n = 0; n < nCells; n++) {{
        const cy = length * Math.cos(alpha / 2) * n;
        const center = [0, cy, 0];
        const back1 = matMul3(rotZ, [0, -length/2, 0]).map((v, i) => v + center[i]);
        const front1 = matMul3(rotZ, [0, length/2, 0]).map((v, i) => v + center[i]);
        const back2 = matMul3(rotZ_, [0, -length/2, 0]).map((v, i) => v + center[i]);
        const front2 = matMul3(rotZ_, [0, length/2, 0]).map((v, i) => v + center[i]);
        front1Verts.push(front1);
        front2Verts.push(front2);
        if (addEnds && n === 0) {{
          const backCenter = [0, -length * Math.cos(alpha / 2), 0];
          kps.push([...backCenter, ...back1, cw, ch, 1]);
          kps.push([...backCenter, ...back2, cw, ch, -1]);
        }}
        kps.push([...back1, ...center, cw, ch, 1]);
        kps.push([...center, ...front1, cw, ch, 1]);
        kps.push([...back2, ...center, cw, ch, -1]);
        kps.push([...center, ...front2, cw, ch, -1]);
        if (addEnds && n === nCells - 1) {{
          const frontCenter = [0, length * Math.cos(alpha / 2) * nCells, 0];
          kps.push([...frontCenter, ...front1, cw, ch, 1]);
          kps.push([...frontCenter, ...front2, cw, ch, -1]);
        }}
      }}
      if (addLongerons) {{
        for (let i = 0; i < front1Verts.length - 1; i++) {{
          kps.push([...front1Verts[i], ...front1Verts[i+1], cw, ch, 1]);
          kps.push([...front2Verts[i], ...front2Verts[i+1], cw, ch, -1]);
        }}
      }}
      return kps;
    }}

    function getOffsetKeypoints(l2, l3, alpha, nCells, cw, ch) {{
      const kps = [];
      for (let i = 0; i < nCells; i++) {{
        const yOff = 2 * l3 * Math.cos(alpha / 2) * i;
        kps.push([
          Math.sin(alpha/2)*l3, Math.cos(alpha/2)*l3 + yOff, 0,
          Math.sin(alpha/2)*(l2+l3), Math.cos(alpha/2)*(l2+l3) + yOff, 0,
          cw, ch, 1
        ]);
        kps.push([
          Math.sin(-alpha/2)*l3, Math.cos(-alpha/2)*l3 + yOff, 0,
          Math.sin(-alpha/2)*(l2+l3), Math.cos(-alpha/2)*(l2+l3) + yOff, 0,
          cw, ch, -1
        ]);
      }}
      return kps;
    }}

    function getFoldKeypoints(segLen, foldAngle, nSegments, cw, ch, layer) {{
      const kps = [];
      let pos = [0, 0, 0];
      const direction = [0, 1, 0];
      for (let i = 0; i < nSegments; i++) {{
        const angle = (i % 2 === 0) ? foldAngle / 2 : -foldAngle / 2;
        const rot = rotZ3(angle);
        const segDir = matMul3(rot, direction);
        const end = pos.map((v, j) => v + segLen * segDir[j]);
        kps.push([...pos, ...end, cw, ch, layer]);
        pos = end;
      }}
      return kps;
    }}

    function transformFoldMembers(kps, centerOffset, jointOffset, theta) {{
      const R = rotY3(theta);
      for (const kp of kps) {{
        for (let i = 0; i < 2; i++) {{
          const v = [kp[3*i] - centerOffset[0], kp[3*i+1] - centerOffset[1], kp[3*i+2] - centerOffset[2]];
          const rv = matMul3(R, v);
          kp[3*i]   = rv[0] + jointOffset[0];
          kp[3*i+1] = rv[1] + jointOffset[1];
          kp[3*i+2] = rv[2] + jointOffset[2];
        }}
      }}
      return kps;
    }}

    function getPetKeypoints(l1, l2, l3, alpha_kp, beta_kp, theta, nCells, cw, ch, addEnds) {{
      // alpha_kp, beta_kp are already in keypoint convention (pi - model angle)
      // PET uses scissors on ALL sides (not folds)
      const kps = [];
      let shortLeft = getScissorKeypoints(2*l1, beta_kp, nCells, cw, ch, false, false);
      let shortRight = getScissorKeypoints(2*l1, beta_kp, nCells, cw, ch, false, false);
      // Flip layer tags: left = 1, right = -1
      for (const k of shortLeft) k[8] = 1;
      for (const k of shortRight) k[8] = -1;

      let jointL = [Math.sin(alpha_kp/2)*(l2+l3), Math.cos(alpha_kp/2)*(l2+l3), 0];
      let centerOff = [0, 0, 0];
      shortLeft = transformFoldMembers(shortLeft, centerOff, jointL, -(Math.PI/2 + theta/2));

      let jointR = [-Math.sin(alpha_kp/2)*(l2+l3), Math.cos(alpha_kp/2)*(l2+l3), 0];
      shortRight = transformFoldMembers(shortRight, centerOff, jointR, (Math.PI/2 + theta/2));

      const longKps = getScissorKeypoints(2*l3, alpha_kp, nCells, cw, ch, addEnds, true);
      const offsetKps = getOffsetKeypoints(l2, l3, alpha_kp, nCells, cw, ch);

      kps.push(longKps[0]);
      kps.push(...shortLeft);
      kps.push(...shortRight);
      kps.push(...longKps.slice(1, -1));
      kps.push(...offsetKps);
      kps.push(longKps[longKps.length - 1]);
      return kps;
    }}

    function getDartKeypoints(short, long, offset, hingePar, alpha_kp, beta_kp, theta, nCells, cw, ch, addEnds) {{
      // New model mapping: l2 = offset, l3 = long - offset, l1 = short/2
      // alpha_kp, beta_kp are already in keypoint convention (pi - model angle)
      const l1 = short / 2;
      const l2 = offset;
      const l3 = long - offset;
      const kps = [];

      let shortLeft = getFoldKeypoints(short, -beta_kp, 2*nCells, cw, ch, 1);
      let shortRight = getFoldKeypoints(short, beta_kp, 2*nCells, cw, ch, -1);
      const centerY = nCells * 2 * short * Math.cos(beta_kp / 2);

      let jointL = [Math.sin(alpha_kp/2)*(l2+l3), Math.cos(alpha_kp/2)*(l2+l3), 0];
      let centerOff = [0, 0, 0];
      shortLeft = transformFoldMembers(shortLeft, centerOff, jointL, -(Math.PI/2 + theta/2));

      let jointR = [-Math.sin(alpha_kp/2)*(l2+l3), Math.cos(alpha_kp/2)*(l2+l3), 0];
      shortRight = transformFoldMembers(shortRight, centerOff, jointR, (Math.PI/2 + theta/2));

      const longKps = getScissorKeypoints(2*l3, alpha_kp, nCells, cw, ch, addEnds, true);
      // Mark long layer as 2
      for (const k of longKps) k[8] = 2;
      const offsetKps = getOffsetKeypoints(l2, l3, alpha_kp, nCells, cw, ch);
      // Mark offset layer as 3
      for (const k of offsetKps) k[8] = 3;

      kps.push(longKps[0]);
      kps.push(...shortLeft);
      kps.push(...shortRight);
      kps.push(...longKps.slice(1, -1));
      kps.push(...offsetKps);
      kps.push(longKps[longKps.length - 1]);
      return kps;
    }}

    function downloadKeypoints(kps, filename) {{
      const header = "x1,y1,z1,x2,y2,z2,width,height,layer";
      const rows = kps.map(k => k.map((v, i) => i < 6 ? v.toFixed(6) : (i < 8 ? v.toFixed(4) : v)).join(","));
      const csv = [header, ...rows].join("\\n");
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    document.getElementById("exportPetBtn").addEventListener("click", function() {{
      if (!currentSol) return;
      const abt = getCurrentAlphaBetaTheta();
      if (!abt) return;
      const cw = parseFloat(document.getElementById("kpCrossWidth").value) || 0.1;
      const ch = parseFloat(document.getElementById("kpCrossHeight").value) || 0.1;
      const l1 = currentSol.short / 2;
      const l2 = currentSol.offset;
      const l3 = currentSol.long - currentSol.offset;
      const nCells = currentSol.n_cells;
      const alpha_kp = Math.PI - abt.alpha;
      const beta_kp = Math.PI - abt.beta;
      const kps = getPetKeypoints(l1, l2, l3, alpha_kp, beta_kp, abt.theta, nCells, cw, ch, false);
      downloadKeypoints(kps, "pet_keypoints.csv");
      const status = document.getElementById("kpExportStatus");
      status.textContent = "\\u2713 Exported pet_keypoints.csv (" + kps.length + " members)";
      status.style.opacity = 1;
      setTimeout(() => {{ status.style.opacity = 0; }}, 4000);
    }});

    document.getElementById("exportDartBtn").addEventListener("click", function() {{
      if (!currentSol) return;
      const abt = getCurrentAlphaBetaTheta();
      if (!abt) return;
      const cw = parseFloat(document.getElementById("kpCrossWidth").value) || 0.1;
      const ch = parseFloat(document.getElementById("kpCrossHeight").value) || 0.1;
      const nCells = currentSol.n_cells;
      const alpha_kp = Math.PI - abt.alpha;
      const beta_kp = Math.PI - abt.beta;
      const kps = getDartKeypoints(currentSol.short, currentSol.long, currentSol.offset, currentSol.hinge_par, alpha_kp, beta_kp, abt.theta, nCells, cw, ch, false);
      downloadKeypoints(kps, "dart_keypoints.csv");
      const status = document.getElementById("kpExportStatus");
      status.textContent = "\\u2713 Exported dart_keypoints.csv (" + kps.length + " members)";
      status.style.opacity = 1;
      setTimeout(() => {{ status.style.opacity = 0; }}, 4000);
    }});

    // ── FEA Integration ──
    const feaRunBtn = document.getElementById("feaRunBtn");
    const feaStatus = document.getElementById("feaStatus");
    const feaResults = document.getElementById("feaResults");
    const feaResultsBody = document.getElementById("feaResultsBody");
    const feaFreqChart = document.getElementById("feaFreqChart");
    const feaModeSelect = document.getElementById("feaModeSelect");
    const feaModeScale = document.getElementById("feaModeScale");
    const feaModeScaleVal = document.getElementById("feaModeScaleVal");
    let feaData = null;  // stores last FEA result
    let feaOverlayGroup = null;  // THREE.js group for FEA visualization
    let feaAnimFrame = null;

    function getCurrentKeypoints() {{
      // Extract member geometry directly from the 3D viewer's buildScene output
      // This ensures FEA geometry matches exactly what the user sees
      const state = window.getViewerState ? window.getViewerState() : null;
      if (!state || !state.sceneMembers || state.sceneMembers.length === 0) return null;

      // Convert collected members to keypoint format [x1,y1,z1, x2,y2,z2, w,h, layer]
      // Coordinates are already in viewer space (Three.js coords)
      // Tag members: structural get beam elements, hinge members collapse
      // into rigid connections (endpoints merged, no beam element created)
      const STRUCTURAL_MATS = {{
        'carbonFiber': 1,       // long arms
        'carbonFiberOffset': 2, // offset members
        'carbonShort': 3,       // short members
        'longeron': 6,          // longerons
      }};
      const HINGE_MATS = ['aluminum', 'aluminumGreen', 'aluminumDark'];

      return state.sceneMembers
        .filter(function(m) {{
          return STRUCTURAL_MATS[m.mat] !== undefined || HINGE_MATS.indexOf(m.mat) >= 0;
        }})
        .map(function(m) {{
          const isHinge = HINGE_MATS.indexOf(m.mat) >= 0;
          return [
            m.a[0], m.a[1], m.a[2],
            m.b[0], m.b[1], m.b[2],
            m.radius * 2, m.radius * 2,
            isHinge ? -1 : STRUCTURAL_MATS[m.mat]  // -1 = hinge (rigid connection)
          ];
        }});
    }}

    function clearFeaOverlay() {{
      if (feaAnimFrame) {{ cancelAnimationFrame(feaAnimFrame); feaAnimFrame = null; }}
      const state = window.getViewerState ? window.getViewerState() : null;
      if (state && state.scene && feaOverlayGroup) {{
        state.scene.remove(feaOverlayGroup);
        feaOverlayGroup = null;
      }}
    }}

    function showLoadArrows(data) {{
      const state = window.getViewerState ? window.getViewerState() : null;
      if (!state || !state.scene) return;
      clearFeaOverlay();
      feaOverlayGroup = new THREE.Group();
      feaOverlayGroup.name = "feaOverlay";

      // Fixed nodes: small red cubes
      if (data.fixed_nodes) {{
        const fixGeo = new THREE.BoxGeometry(0.03, 0.03, 0.03);
        const fixMat = new THREE.MeshBasicMaterial({{ color: 0xff4444 }});
        data.fixed_nodes.forEach(function(p) {{
          const m = new THREE.Mesh(fixGeo, fixMat);
          m.position.set(p[0], p[1], p[2]);
          feaOverlayGroup.add(m);
        }});
      }}

      // Load arrows at tip nodes
      const arrowLen = 0.3;
      if (data.load_arrows) {{
        data.load_arrows.forEach(function(a) {{
          const origin = new THREE.Vector3(a.pos[0], a.pos[1], a.pos[2]);
          const dir = new THREE.Vector3(a.dir[0], a.dir[1], a.dir[2]).normalize();
          const arrow = new THREE.ArrowHelper(dir, origin, arrowLen, 0xffcc00, 0.08, 0.04);
          feaOverlayGroup.add(arrow);
        }});
      }}

      state.scene.add(feaOverlayGroup);
    }}

    function showModeShape(modeIdx, scale) {{
      const state = window.getViewerState ? window.getViewerState() : null;
      if (!state || !state.scene || !feaData) return;
      clearFeaOverlay();
      if (modeIdx < 0 || modeIdx >= feaData.mode_shapes.length) {{
        // Just show load arrows
        showLoadArrows(feaData);
        return;
      }}

      feaOverlayGroup = new THREE.Group();
      feaOverlayGroup.name = "feaOverlay";

      const nodes = feaData.nodes;
      const members = feaData.members;
      const shape = feaData.mode_shapes[modeIdx];

      // Animate mode shape
      let phase = 0;
      const lineMat = new THREE.LineBasicMaterial({{ color: 0xff3333, linewidth: 2 }});

      function animateMode() {{
        // Remove old lines
        while (feaOverlayGroup.children.length > 0) {{
          feaOverlayGroup.remove(feaOverlayGroup.children[0]);
        }}

        phase += 0.04;
        const amp = Math.sin(phase) * scale;

        // Draw deformed members
        members.forEach(function(mem) {{
          const n0 = nodes[mem[0]];
          const n1 = nodes[mem[1]];
          const d0 = shape[mem[0]];
          const d1 = shape[mem[1]];

          const p0 = new THREE.Vector3(
            n0[0] + d0[0] * amp,
            n0[1] + d0[1] * amp,
            n0[2] + d0[2] * amp
          );
          const p1 = new THREE.Vector3(
            n1[0] + d1[0] * amp,
            n1[1] + d1[1] * amp,
            n1[2] + d1[2] * amp
          );

          const geo = new THREE.BufferGeometry().setFromPoints([p0, p1]);
          feaOverlayGroup.add(new THREE.Line(geo, lineMat));
        }});

        feaAnimFrame = requestAnimationFrame(animateMode);
      }}

      state.scene.add(feaOverlayGroup);
      animateMode();
    }}

    feaModeSelect.addEventListener("change", function() {{
      const idx = parseInt(this.value);
      const scale = parseFloat(feaModeScale.value);
      showModeShape(idx, scale);
    }});

    feaModeScale.addEventListener("input", function() {{
      feaModeScaleVal.textContent = parseFloat(this.value).toFixed(1) + "x";
      const idx = parseInt(feaModeSelect.value);
      if (idx >= 0 && feaData) {{
        clearFeaOverlay();
        showModeShape(idx, parseFloat(this.value));
      }}
    }});

    function renderFeaResults(data) {{
      if (data.error) {{
        feaStatus.textContent = "Error: " + data.error;
        feaStatus.style.color = "#e74c3c";
        feaResults.style.display = "none";
        return;
      }}

      feaData = data;
      feaResults.style.display = "block";

      const compWarn = (data.n_components && data.n_components > 1)
        ? " \\u26a0 " + data.n_components + " disconnected parts!"
        : " (connected)";
      const rows = [
        ["Nodes / Members", data.n_nodes + " / " + data.n_members + compWarn],
        ["Material", data.material_label],
        ["Mass", data.mass_kg.toFixed(3) + " kg"],
        ["f1", data.frequencies[0] ? data.frequencies[0].toFixed(4) + " Hz" : "N/A"],
        ["Max stress", data.sig_max_mpa.toFixed(1) + " MPa"],
        ["k_X", data.stiffness.X.toFixed(1) + " N/m"],
        ["k_Y", data.stiffness.Y.toFixed(1) + " N/m"],
        ["k_Z", data.stiffness.Z.toFixed(1) + " N/m"],
        ["Fixed / Tip nodes", (data.n_fixed_nodes || "?") + " / " + (data.n_tip_nodes || "?")],
      ];
      feaResultsBody.innerHTML = rows.map(function(r) {{
        return '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>';
      }}).join("");

      // Frequency bar chart
      if (data.frequencies && data.frequencies.length > 0) {{
        const fLabels = data.frequencies.map(function(_, i) {{ return "f" + (i+1); }});
        Plotly.newPlot(feaFreqChart, [{{
          x: fLabels,
          y: data.frequencies,
          type: "bar",
          marker: {{ color: "#e74c3c" }},
          hovertemplate: "%{{x}}: %{{y:.4f}} Hz<extra></extra>"
        }}], {{
          margin: {{ t: 4, b: 24, l: 40, r: 8 }},
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: {{ color: "#555", tickfont: {{ size: 9 }} }},
          yaxis: {{ color: "#555", tickfont: {{ size: 9 }}, title: {{ text: "Hz", font: {{ size: 9, color: "#555" }} }} }},
          height: 120,
        }}, {{ displayModeBar: false, responsive: true }});
      }}

      // Populate mode selector
      feaModeSelect.innerHTML = '<option value="-1">Load arrows</option>';
      data.frequencies.forEach(function(f, i) {{
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = "Mode " + (i+1) + " (" + f.toFixed(2) + " Hz)";
        feaModeSelect.appendChild(opt);
      }});

      // Show load arrows by default
      showLoadArrows(data);
    }}

    // Toggle custom Y range inputs
    document.getElementById("feaFixMode").addEventListener("change", function() {{
      document.getElementById("feaFixCustomRow").style.display = this.value === "custom" ? "block" : "none";
    }});

    feaRunBtn.addEventListener("click", async function() {{
      if (!currentSol) return;

      // Extract current keypoints from the viewer
      const kps = getCurrentKeypoints();
      if (!kps || kps.length < 2) {{
        feaStatus.textContent = "Need at least 2 members. Adjust cells/angle.";
        feaStatus.style.color = "#e74c3c";
        return;
      }}

      feaRunBtn.disabled = true;
      feaStatus.textContent = "Running FEA on " + kps.length + " members...";
      feaStatus.style.color = "#aaa";
      feaResults.style.display = "none";
      clearFeaOverlay();

      // Build boundary condition config
      const fixMode = document.getElementById("feaFixMode").value;
      const bcConfig = {{
        fix_mode: fixMode,
        load_at: document.getElementById("feaLoadAt").value,
        fix_dofs: document.getElementById("feaFixDofs").value,
      }};
      if (fixMode === "custom") {{
        const ymin = document.getElementById("feaFixYMin").value;
        const ymax = document.getElementById("feaFixYMax").value;
        if (ymin !== "") bcConfig.fix_y_min = parseFloat(ymin);
        if (ymax !== "") bcConfig.fix_y_max = parseFloat(ymax);
      }}

      try {{
        const resp = await fetch("/run-fea", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            keypoints: kps,
            material: document.getElementById("feaMaterial").value,
            profile_type: document.getElementById("feaProfile").value,
            hollow: document.getElementById("feaHollow").checked,
            outer_dim_mm: parseFloat(document.getElementById("feaDim").value),
            wall_mm: parseFloat(document.getElementById("feaWall").value),
            P_load: parseFloat(document.getElementById("feaLoad").value),
            load_dir: document.getElementById("feaLoadDir").value,
            bc: bcConfig,
          }})
        }});
        const data = await resp.json();
        if (resp.ok) {{
          feaStatus.textContent = "\\u2713 FEA complete (" + data.n_nodes + " nodes, " + data.n_members + " members)";
          feaStatus.style.color = "#2ecc71";
          renderFeaResults(data);
        }} else {{
          feaStatus.textContent = "Error: " + (data.error || "unknown");
          feaStatus.style.color = "#e74c3c";
        }}
      }} catch (err) {{
        feaStatus.textContent = "Request failed: " + err.message;
        feaStatus.style.color = "#e74c3c";
      }}

      feaRunBtn.disabled = false;
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

    parser.add_argument("--offset-min", type=float, default=0.01)
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

    print("args", args)
    output = solve_one(args.stow_width, args.stow_depth, args.expanded_depth, args.states, args.hinge_par, 0.015, args.n, args.short, args.long, None, args.tee)
    # import pdb; pdb.set_trace()

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