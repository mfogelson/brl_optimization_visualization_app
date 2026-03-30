"""
FEA on scissor-linkage geometry.

Takes keypoints (member start/end positions) from the current viewer state,
builds a beam FEA model, and solves for static response + natural frequencies.
Reuses the FEA assembly functions from the 3D Truss Optimizer.
"""

import sys
import os
import math
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh

# Import FEA primitives from the 3D Truss Optimizer
_TRUSS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '3d_Truss_Optimizer'))
if _TRUSS_DIR not in sys.path:
    sys.path.insert(0, _TRUSS_DIR)

from truss_optimizer import (
    beam_ke, compute_rotations, assemble_K, assemble_M,
    compute_member_forces, compute_cross_section_props,
    MATERIAL_PRESETS,
)
import warnings


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and infinity."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            if math.isinf(v) or math.isnan(v):
                return None
            return v
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return None
        return super().default(o)


# ---------------------------------------------------------------------------
#  Geometry: keypoints -> nodes + members
# ---------------------------------------------------------------------------

def _build_mesh_from_keypoints(keypoints, merge_tol=None):
    """Convert keypoints to a node array and member connectivity.

    Members with layer == -1 are treated as rigid connections: their
    endpoints are force-merged into a single node (collapsing the hinge
    chain) and no beam element is created for them. This connects the
    structural members that were linked through hinge details.

    Parameters
    ----------
    keypoints : list of [x1,y1,z1, x2,y2,z2, w, h, layer]
        layer -1 = rigid connection (hinge), others = structural beam.
    merge_tol : float or None
        Distance below which two endpoints are merged into one node.
        If None, auto-detect from structural member lengths.

    Returns
    -------
    nodes : ndarray (nn, 3)
    members : ndarray (nm, 2) int  -- indices into nodes (structural only)
    layers : list of int per member
    """
    pts = []
    raw_members = []
    raw_layers = []

    for kp in keypoints:
        p1 = np.array(kp[0:3], dtype=float)
        p2 = np.array(kp[3:6], dtype=float)
        layer = int(kp[8]) if len(kp) > 8 else 0
        raw_layers.append(layer)
        raw_members.append((p1, p2))
        pts.append(p1)
        pts.append(p2)

    if not pts:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int), []

    all_pts = np.array(pts)  # (2*nm, 3)

    # Auto-detect merge tolerance from STRUCTURAL member lengths only
    if merge_tol is None:
        struct_lens = [np.linalg.norm(p2 - p1)
                       for (p1, p2), ly in zip(raw_members, raw_layers) if ly >= 0]
        median_len = np.median(struct_lens) if struct_lens else 0.1
        merge_tol = max(1e-4, median_len * 0.02)

    # Phase 1: standard proximity merge
    node_map = np.full(len(all_pts), -1, dtype=int)
    nodes_list = []
    for i, p in enumerate(all_pts):
        if node_map[i] >= 0:
            continue
        idx = len(nodes_list)
        nodes_list.append(p)
        node_map[i] = idx
        dists = np.linalg.norm(all_pts[i + 1:] - p, axis=1)
        for j in np.where(dists < merge_tol)[0]:
            node_map[i + 1 + j] = idx

    # Phase 2: force-merge endpoints of hinge members (layer == -1)
    # This collapses hinge chains so structural members on either side share a node
    parent = list(range(len(nodes_list)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, layer in enumerate(raw_layers):
        if layer == -1:
            na = node_map[2 * i]
            nb = node_map[2 * i + 1]
            union(na, nb)

    # Rebuild node list with collapsed hinges
    root_map = {}
    new_nodes = []
    for old_idx in range(len(nodes_list)):
        r = find(old_idx)
        if r not in root_map:
            root_map[r] = len(new_nodes)
            new_nodes.append(nodes_list[r])
        # Map old_idx -> new collapsed idx
        root_map[old_idx] = root_map[r]

    # Remap node_map through the collapse
    final_map = np.array([root_map[find(node_map[i])] for i in range(len(all_pts))])

    nodes = np.array(new_nodes)

    # Phase 3: build member list (structural only, layer >= 0)
    members_out = []
    layers_out = []
    for i, layer in enumerate(raw_layers):
        if layer < 0:
            continue  # skip hinge members
        na = final_map[2 * i]
        nb = final_map[2 * i + 1]
        if na != nb:  # skip zero-length after merge
            members_out.append([na, nb])
            layers_out.append(layer)

    members = np.array(members_out, dtype=int) if members_out else np.empty((0, 2), dtype=int)
    return nodes, members, layers_out


# ---------------------------------------------------------------------------
#  Boundary conditions: find fixed nodes (base of the structure)
# ---------------------------------------------------------------------------

def _nodes_at_y_end(nodes, end='max', tol_frac=0.02):
    """Find nodes near the max or min Y."""
    y_vals = nodes[:, 1]
    span = y_vals.max() - y_vals.min()
    tol = max(0.01, span * tol_frac)
    if end == 'max':
        return np.where(y_vals >= y_vals.max() - tol)[0]
    else:
        return np.where(y_vals <= y_vals.min() + tol)[0]


def _nodes_at_y_mid(nodes, tol_frac=0.05):
    """Find nodes near the Y midpoint."""
    y_vals = nodes[:, 1]
    mid = (y_vals.max() + y_vals.min()) / 2
    span = y_vals.max() - y_vals.min()
    tol = max(0.01, span * tol_frac)
    return np.where(np.abs(y_vals - mid) <= tol)[0]


def _nodes_in_y_range(nodes, y_min, y_max):
    """Find nodes within a Y range."""
    y_vals = nodes[:, 1]
    return np.where((y_vals >= y_min) & (y_vals <= y_max))[0]


def _select_bc_nodes(nodes, bc):
    """Select fixed and loaded nodes based on BC config.

    bc : dict with keys:
        fix_mode : 'base', 'tip', 'both', 'base_wide', 'custom'
        load_at  : 'auto', 'base', 'tip', 'mid'
        fix_dofs : 'all', 'pin'
        fix_y_min, fix_y_max : float (for custom mode)

    Returns (fixed_nodes, tip_nodes, dofs_per_node)
    """
    fix_mode = bc.get('fix_mode', 'base')
    load_at = bc.get('load_at', 'auto')
    fix_dofs_mode = bc.get('fix_dofs', 'all')

    # Fixed nodes
    if fix_mode == 'base':
        fixed = _nodes_at_y_end(nodes, 'max')
    elif fix_mode == 'tip':
        fixed = _nodes_at_y_end(nodes, 'min')
    elif fix_mode == 'both':
        fixed = np.union1d(
            _nodes_at_y_end(nodes, 'max'),
            _nodes_at_y_end(nodes, 'min'))
    elif fix_mode == 'base_wide':
        fixed = _nodes_at_y_end(nodes, 'max', tol_frac=0.10)
    elif fix_mode == 'custom':
        y_min = bc.get('fix_y_min', nodes[:, 1].min())
        y_max = bc.get('fix_y_max', nodes[:, 1].max())
        fixed = _nodes_in_y_range(nodes, y_min, y_max)
    else:
        fixed = _nodes_at_y_end(nodes, 'max')

    # Loaded nodes
    if load_at == 'auto':
        # Opposite end from fixed
        if fix_mode in ('tip',):
            tip = _nodes_at_y_end(nodes, 'max')
        else:
            tip = _nodes_at_y_end(nodes, 'min')
    elif load_at == 'base':
        tip = _nodes_at_y_end(nodes, 'max')
    elif load_at == 'tip':
        tip = _nodes_at_y_end(nodes, 'min')
    elif load_at == 'mid':
        tip = _nodes_at_y_mid(nodes)
    else:
        tip = _nodes_at_y_end(nodes, 'min')

    # DOFs per fixed node
    dofs_per_node = 6 if fix_dofs_mode == 'all' else 3

    return fixed, tip, dofs_per_node


# ---------------------------------------------------------------------------
#  Core solver
# ---------------------------------------------------------------------------

def run_fea_on_keypoints(keypoints, material='aluminum_6061',
                         profile_type='square', hollow=True,
                         outer_dim_mm=10.0, wall_mm=1.0,
                         P_load=100.0, load_dir='Y',
                         n_modes=8, bc=None):
    """Run beam FEA on a set of keypoints.

    Parameters
    ----------
    keypoints : list of [x1,y1,z1, x2,y2,z2, w, h, layer]
        Member endpoints from the scissor viewer.
    material : str
        Key into MATERIAL_PRESETS.
    profile_type : str
        'square', 'rectangle', or 'circle'.
    hollow : bool
    outer_dim_mm : float
        Outer dimension of cross-section [mm].
    wall_mm : float
        Wall thickness [mm].
    P_load : float
        Applied tip load magnitude [N].
    load_dir : str
        'X', 'Y', or 'Z'.
    n_modes : int
        Number of vibration modes to compute.

    Returns
    -------
    dict with keys: nodes, members, frequencies, mode_shapes, stiffness,
         displacements, member_stresses, mass, etc.
    """
    # -- Material properties --
    mat = MATERIAL_PRESETS.get(material, MATERIAL_PRESETS['aluminum_6061'])
    E = mat['E']
    nu = mat.get('nu', 0.33)
    G = E / (2 * (1 + nu))
    rho = mat['rho']
    sigma_y = mat['sigma_y']

    # -- Cross-section --
    b = outer_dim_mm * 1e-3  # m
    t = wall_mm * 1e-3       # m
    if profile_type == 'square':
        dims = {'b': b, 't': t} if hollow else {'b': b}
    elif profile_type == 'circle':
        dims = {'d': b, 't': t} if hollow else {'d': b}
    elif profile_type == 'rectangle':
        dims = {'h': b, 'w': b, 't': t} if hollow else {'h': b, 'w': b}
    else:
        dims = {'b': b, 't': t} if hollow else {'b': b}

    A, Iy, Iz, J, c_out = compute_cross_section_props(profile_type, hollow, dims)

    # -- Build mesh --
    nodes, members, layers = _build_mesh_from_keypoints(keypoints)
    nn = nodes.shape[0]
    nm = members.shape[0]

    if nn < 2 or nm < 1:
        raise ValueError(f"Not enough geometry: {nn} nodes, {nm} members")

    # -- Member properties --
    dx = nodes[members[:, 1]] - nodes[members[:, 0]]
    memL = np.linalg.norm(dx, axis=1)
    memL = np.maximum(memL, 1e-10)  # avoid zero-length

    # Direction cosines
    cx = dx[:, 0] / memL
    cy = dx[:, 1] / memL
    cz = dx[:, 2] / memL

    rot = compute_rotations(cx, cy, cz)

    # Uniform properties for all members
    E_arr = np.full(nm, E)
    A_arr = np.full(nm, A)
    Iy_arr = np.full(nm, Iy)
    Iz_arr = np.full(nm, Iz)
    J_arr = np.full(nm, J)
    G_arr = np.full(nm, G)

    # -- Assembly --
    nd = 6 * nn
    K = assemble_K(nodes, members, E_arr, A_arr, Iy_arr, Iz_arr, J_arr, G_arr, memL, rot)
    Mm = assemble_M(nodes, members, A_arr, Iy_arr, Iz_arr, J_arr, memL, rho)

    # -- Boundary conditions --
    if bc is None:
        bc = {}
    fixed_nodes, tip_nodes, dofs_per_fixed = _select_bc_nodes(nodes, bc)

    if len(fixed_nodes) == 0:
        raise ValueError("No fixed nodes found for the selected BC mode")
    if len(tip_nodes) == 0:
        raise ValueError("No loaded nodes found for the selected load location")

    # Build fixed DOF list
    if dofs_per_fixed == 6:
        fixed_dofs = np.concatenate([np.arange(6) + n * 6 for n in fixed_nodes])
    else:
        # Pin: fix only translational DOFs (0,1,2), leave rotations free
        fixed_dofs = np.concatenate([np.arange(3) + n * 6 for n in fixed_nodes])

    all_dofs = np.arange(nd)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    if len(free_dofs) < 2:
        raise ValueError("Too few free DOFs for analysis")

    Kff = K[np.ix_(free_dofs, free_dofs)]

    def _safe_solve(K_sub, F_sub):
        """Solve K*U=F, returning zeros if singular."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                u = spsolve(K_sub, F_sub)
                if np.any(~np.isfinite(u)):
                    return np.zeros_like(F_sub)
                return u
            except Exception:
                return np.zeros_like(F_sub)

    def _sanitize(arr):
        """Replace NaN/Inf with 0."""
        a = np.asarray(arr, dtype=float)
        a[~np.isfinite(a)] = 0.0
        return a

    # -- Static solve (3 load directions) --
    load_map = {'X': 0, 'Y': 1, 'Z': 2}
    results_by_dir = {}
    for dir_name, dof_idx in load_map.items():
        F = np.zeros(nd)
        Pn = P_load / len(tip_nodes)
        for tn in tip_nodes:
            F[tn * 6 + dof_idx] = Pn
        U = np.zeros(nd)
        U[free_dofs] = _safe_solve(Kff, F[free_dofs])
        axial, moment = compute_member_forces(
            nodes, members, E_arr, A_arr, Iy_arr, Iz_arr, J_arr, G_arr, memL, rot, U)
        sig = np.abs(axial / A) + np.abs(moment * c_out / Iy)

        tip_disp = np.mean(np.abs(U[tip_nodes * 6 + dof_idx]))
        stiffness = P_load / tip_disp if tip_disp > 1e-30 else 0.0

        results_by_dir[dir_name] = dict(
            U=U, axial=axial, moment=moment, stress=sig, stiffness=stiffness, F=F)

    # Primary load direction
    pri = results_by_dir.get(load_dir, results_by_dir['Y'])

    # -- Modal analysis --
    n_modes_actual = min(n_modes, len(free_dofs) - 2)
    Mff = Mm[np.ix_(free_dofs, free_dofs)]
    eps_m = 1e-12 * sparse.eye(Mff.shape[0])

    try:
        eigenvalues, V = eigsh(Kff, k=max(1, n_modes_actual),
                               M=Mff + eps_m, sigma=1e-2, which='LM')
        si = np.argsort(np.abs(eigenvalues))
        eigenvalues = np.abs(eigenvalues[si])
        V = V[:, si]
        freq = np.sqrt(eigenvalues) / (2 * np.pi)
    except Exception:
        freq = np.zeros(max(1, n_modes_actual))
        V = np.zeros((len(free_dofs), max(1, n_modes_actual)))

    # Expand mode shapes to full DOF vector, normalize
    modes = np.zeros((nd, len(freq)))
    for i in range(len(freq)):
        modes[free_dofs, i] = V[:, i]
        mx = np.max(np.abs(modes[:, i]))
        if mx > 0:
            modes[:, i] /= mx

    # -- Extract node displacements for mode shapes --
    # Reshape modes to (nn, 6, n_modes) -> take XYZ translation (first 3 DOFs)
    mode_shapes_xyz = []
    for i in range(len(freq)):
        shape = _sanitize(modes[:, i]).reshape(nn, 6)[:, :3]  # (nn, 3)
        mode_shapes_xyz.append(shape.tolist())

    # Primary displacement as node XYZ
    pri_disp_xyz = _sanitize(pri['U']).reshape(nn, 6)[:, :3]  # (nn, 3)

    # Mass
    mass = rho * A * np.sum(memL)

    # Load arrow data: tip nodes positions and force direction
    load_arrows = []
    for tn in tip_nodes:
        pos = nodes[tn].tolist()
        direction = [0, 0, 0]
        direction[load_map[load_dir]] = 1.0
        load_arrows.append({'pos': pos, 'dir': direction})

    # Fixed node indicators
    fixed_positions = nodes[fixed_nodes].tolist()

    return dict(
        # Geometry
        nodes=nodes.tolist(),
        members=members.tolist(),
        layers=layers,
        n_nodes=nn,
        n_members=nm,

        # Material
        material=material,
        material_label=mat['label'],
        E_GPa=round(E / 1e9, 1),
        rho=rho,
        sigma_y_MPa=round(sigma_y / 1e6, 1),

        # Cross section
        profile_type=profile_type,
        hollow=hollow,
        outer_dim_mm=outer_dim_mm,
        wall_mm=wall_mm,

        # Mass
        mass_kg=round(mass, 4),

        # Frequencies
        frequencies=[round(float(f), 4) for f in freq],

        # Mode shapes: list of (nn x 3) displacement arrays, one per mode
        mode_shapes=mode_shapes_xyz,

        # Static response (primary load direction)
        P_load=P_load,
        load_dir=load_dir,
        displacement=pri_disp_xyz.tolist(),
        member_stress_mpa=[round(float(s) / 1e6, 2) for s in _sanitize(pri['stress'])],
        member_axial=[round(float(a), 2) for a in _sanitize(pri['axial'])],
        sig_max_mpa=round(float(np.max(_sanitize(pri['stress']))) / 1e6, 2),

        # Stiffness (N/m for each direction)
        stiffness={
            d: round(float(_sanitize(np.array([results_by_dir[d]['stiffness']]))[0]), 2)
            for d in ['X', 'Y', 'Z']
        },

        # Visualization helpers
        load_arrows=load_arrows,
        fixed_nodes=fixed_positions,
        tip_nodes=nodes[tip_nodes].tolist(),

        # Diagnostics
        n_fixed_nodes=len(fixed_nodes),
        n_tip_nodes=len(tip_nodes),
        n_free_dofs=len(free_dofs),
        n_fixed_dofs=len(fixed_dofs),
        n_components=_count_components(nn, members),
    )


def _count_components(nn, members):
    """Count connected components via union-find."""
    parent = list(range(nn))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for m in members:
        union(int(m[0]), int(m[1]))
    return len(set(find(i) for i in range(nn)))


def get_material_presets():
    """Return material presets for the UI."""
    return {
        name: {
            'label': m['label'],
            'E_GPa': round(m['E'] / 1e9, 1),
            'sigma_y_MPa': round(m['sigma_y'] / 1e6, 1),
            'rho': m['rho'],
        }
        for name, m in MATERIAL_PRESETS.items()
    }
