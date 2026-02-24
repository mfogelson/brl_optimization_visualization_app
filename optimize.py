"""
Scissor-linkage optimizer — run locally with Pyomo/bonmin.
Writes results to scissor_results.json for the 3D viewer.

Usage:
    python optimize.py --stow-width 1.0 --stow-depth 2.0 --expanded-depth 5.0
"""

import json
import math
import argparse
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import SolverFactory


def build_model(stow_width, stow_depth, expanded_depth, n_states=2, hinge_par=None, thickness=None, n=None, short=None, long=None, offset=None):
    m = pyo.ConcreteModel("ScissorLinkage")
    m.I = pyo.RangeSet(1, n_states)

    m.short = pyo.Var(bounds=(0.01, 10.0)) if short is None else short
    m.long = pyo.Var(bounds=(0.01, 20.0)) if long is None else long
    m.offset = pyo.Var(bounds=(0.01, 10.0)) if offset is None else offset
    m.thickness = pyo.Var(bounds=(0.001, 0.5)) if thickness is None else thickness
    m.hinge_par = pyo.Var(bounds=(0.0, 10.0)) if hinge_par is None else hinge_par
    
    n_initial = int(round(expanded_depth / stow_width))
    m.n = pyo.Var(bounds=(1, 2 * n_initial), domain=pyo.Integers, initialize=n_initial) if n is None else n

    m.sa = pyo.Var(m.I, bounds=(0.0, 1.0))
    m.ca = pyo.Var(m.I, bounds=(0.0, 1.0))
    m.sb = pyo.Var(m.I, bounds=(0.0, 1.0))
    m.cb = pyo.Var(m.I, bounds=(0.0, 1.0))

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

    @m.Constraint(m.I)
    def cell_depth_match(m, i):
        return m.c_long[i] == m.c_short[i]

    m.offset_def = pyo.Constraint(
        expr=m.offset == (m.long * m.sa[n_states] - m.short * m.sb[n_states]) / m.sa[n_states]
    )
    m.width_limit = pyo.Constraint(expr=2 * m.a[1] + 2 * m.hinge_par <= stow_width)
    m.stow_depth_limit = pyo.Constraint(expr=m.length[1] <= stow_depth)
    m.expanded_depth_target = pyo.Constraint(expr=m.length[n_states] == expanded_depth)
    m.initial_state = pyo.Constraint(expr=m.a[1] == m.b[1] + m.hinge_par)
    m.final_state = pyo.Constraint(expr=2.0 * m.a[n_states] == m.b[n_states])
    m.initial_angle_beta = pyo.Constraint(expr= m.sb[1] == (m.thickness/2.0)/m.short)
    # m.initial_angle_alpha = pyo.Constraint(expr= m.sa[1] == (m.thickness/2.0)/m.long)
    # m.stow_depth_constraint = pyo.Constraint(expr=m.thickness * m.n <= stow_depth)
    

    m.objective = pyo.Objective(expr=m.a[n_states], sense=pyo.maximize)

    return m


def solve(stow_width, stow_depth, expanded_depth, n_states=2, hinge_par=None, thickness=None, n=None, short=None, long=None, offset=None):
    model = build_model(stow_width, stow_depth, expanded_depth, n_states, hinge_par, thickness, n, short, long, offset)

    for name in ("bonmin", "ipopt"):
        solver = SolverFactory(name)
        if solver.available():
            if name == "bonmin":
                solver.options["honor_original_bounds"] = "yes"
            break
    else:
        raise RuntimeError("No solver found")

    result = solver.solve(model, tee=True)
    term = result.solver.termination_condition

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

    out = {
        "inputs": {
            "stow_width": stow_width,
            "stow_depth": stow_depth,
            "expanded_depth": expanded_depth,
        },
        "solution": {
            "long": value(model.long),
            "short": value(model.short),
            "offset": value(model.offset),
            "hinge_par": value(model.hinge_par),
            "n_cells": int(round(value(model.n))),
            "alpha_stow": angle(model.sa[1], model.ca[1]),
            "alpha_deploy": angle(model.sa[n_states], model.ca[n_states]),
            "beta_stow": angle(model.sb[1], model.cb[1]),
            "beta_deploy": angle(model.sb[n_states], model.cb[n_states]),
            "theta_deploy": theta(n_states),
            "a_stow": value(model.a[1]),
            "b_stow": value(model.b[1]),
            "c_stow": value(model.c_long[1]),
            "a_deploy": value(model.a[n_states]),
            "b_deploy": value(model.b[n_states]),
            "c_deploy": value(model.c_long[n_states]),
            "width_stow": value(2 * model.a[1] + 2 * model.hinge_par),
            "depth_stow": value(model.length[1]),
            "depth_deploy": value(model.length[n_states]),
            "thickness": value(model.thickness),
        },
        "status": str(term),
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stow-width", type=float, default=1.0)
    parser.add_argument("--stow-depth", type=float, default=2.0)
    parser.add_argument("--expanded-depth", type=float, default=5.0)
    parser.add_argument("--hinge-par", type=float, default=0.0125)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--short", type=float, default=None)
    parser.add_argument("--long", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--thickness", type=float, default=None)
    parser.add_argument("-o", "--output", default="scissor_results.json")
    args = parser.parse_args()

    result = solve(args.stow_width, args.stow_depth, args.expanded_depth, args.states, args.hinge_par, args.thickness, args.n, args.short, args.long, args.offset)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to {args.output}")
    print(json.dumps(result, indent=2))
