import numpy as np
import pygismo as gs
from scipy import sparse
from scipy.spatial.distance import cdist


def gauss_legendre_01(n):
    pts, wts = np.polynomial.legendre.leggauss(n)
    return 0.5 * (pts + 1), 0.5 * wts


def get_side_dofs(basis, side_idx):
    n0 = basis.component(0).size()
    n1 = basis.component(1).size()
    if side_idx == 1:
        return list(range(0, n0 * n1, n0))
    if side_idx == 2:
        return list(range(n0 - 1, n0 * n1, n0))
    if side_idx == 3:
        return list(range(0, n0))
    if side_idx == 4:
        return list(range(n0 * (n1 - 1), n0 * n1))
    return []


def build_multipatch_mapper(mp):
    mb = gs.core.gsMultiBasis(mp)
    mapper = gs.core.gsDofMapper(mb)

    for iface in mp.interfaces():
        ps1, ps2 = iface.first(), iface.second()
        p1, s1 = ps1.patchIndex(), ps1.side().index()
        p2, s2 = ps2.patchIndex(), ps2.side().index()

        b1, b2 = mp.patch(p1).basis(), mp.patch(p2).basis()
        dofs1 = get_side_dofs(b1, s1)
        dofs2 = get_side_dofs(b2, s2)

        def dof_to_phys(patch, basis, dof):
            n0 = basis.component(0).size()
            grev0 = basis.knots(0).greville().flatten()
            grev1 = basis.knots(1).greville().flatten()
            i, j = dof % n0, dof // n0
            return patch.eval(np.array([[grev0[i]], [grev1[j]]])).flatten()

        coords1 = np.array([dof_to_phys(mp.patch(p1), b1, d) for d in dofs1])
        coords2 = np.array([dof_to_phys(mp.patch(p2), b2, d) for d in dofs2])
        match_idx = cdist(coords1, coords2).argmin(axis=1)

        arr1 = np.array(dofs1, dtype=np.int32).reshape(-1, 1)
        arr2 = np.array([dofs2[j] for j in match_idx], dtype=np.int32).reshape(-1, 1)
        mapper.matchDofs(p1, arr1, p2, arr2, 0)

    for b in mp.boundaries():
        p_idx = b.patchIndex()
        s_idx = b.side().index()
        side_dofs = get_side_dofs(mp.patch(p_idx).basis(), s_idx)
        arr = np.array(side_dofs, dtype=np.int32).reshape(-1, 1)
        mapper.markBoundary(p_idx, arr, 0)

    mapper.finalize()
    return mapper


def assemble_poisson_patch(patch, basis, f_func, mapper, patch_idx, n_global, n_gauss=None):
    p = max(basis.degree(0), basis.degree(1))
    if n_gauss is None:
        n_gauss = p + 2

    gp_1d, gw_1d = gauss_legendre_01(n_gauss)
    breaks_0 = sorted(set(basis.knots(0).get()))
    breaks_1 = sorted(set(basis.knots(1).get()))

    use_gismo_eval = hasattr(f_func, "eval") and hasattr(f_func, "domainDim")

    rows, cols, vals_k = [], [], []
    f_vec = np.zeros(n_global)

    for xi_lo, xi_hi in zip(breaks_0[:-1], breaks_0[1:]):
        h_xi = xi_hi - xi_lo
        for eta_lo, eta_hi in zip(breaks_1[:-1], breaks_1[1:]):
            h_eta = eta_hi - eta_lo
            for i_gp in range(n_gauss):
                xi_q = xi_lo + h_xi * gp_1d[i_gp]
                for j_gp in range(n_gauss):
                    eta_q = eta_lo + h_eta * gp_1d[j_gp]
                    w_q = gw_1d[i_gp] * gw_1d[j_gp] * h_xi * h_eta
                    pt = np.array([[xi_q], [eta_q]])

                    act_local = basis.active(pt).flatten()
                    n_act = len(act_local)
                    n_vals = basis.eval(pt).flatten()
                    d_vals = basis.deriv(pt).flatten().reshape(n_act, 2)

                    jac = patch.jacobian(pt)
                    det_j = np.linalg.det(jac)
                    jinv_t = np.linalg.inv(jac).T
                    grad_phys = (jinv_t @ d_vals.T).T

                    xy = patch.eval(pt)
                    factor = np.abs(det_j) * w_q
                    gdofs = [mapper.index(int(d), patch_idx, 0) for d in act_local]

                    k_local = grad_phys @ grad_phys.T * factor
                    for a in range(n_act):
                        for b in range(n_act):
                            rows.append(gdofs[a])
                            cols.append(gdofs[b])
                            vals_k.append(k_local[a, b])

                    if use_gismo_eval:
                        f_val = f_func.eval(xy).flatten()[0]
                    else:
                        f_val = f_func(xy[0, 0], xy[1, 0])
                    for a in range(n_act):
                        f_vec[gdofs[a]] += f_val * n_vals[a] * factor

    k_mat = sparse.coo_matrix((vals_k, (rows, cols)), shape=(n_global, n_global)).tocsr()
    return k_mat, f_vec


class _IdentityMapper:
    def __init__(self, n):
        self._n = n

    def index(self, d, p, c):
        return d

    def size(self):
        return self._n

    def freeSize(self, c):
        return self._n

    def is_free(self, d, p, c):
        return True

    def boundarySize(self):
        return 0


def assemble_poisson(mp_or_patch, f_func, mapper=None, n_gauss=None):
    if hasattr(mp_or_patch, "nPatches"):
        mp_loc = mp_or_patch
        if mapper is None:
            mapper = build_multipatch_mapper(mp_loc)
        n_global = mapper.size()
        k_total = sparse.csr_matrix((n_global, n_global))
        f_total = np.zeros(n_global)
        for p_idx in range(mp_loc.nPatches()):
            patch = mp_loc.patch(p_idx)
            basis = patch.basis()
            k_patch, f_patch = assemble_poisson_patch(
                patch, basis, f_func, mapper, p_idx, n_global, n_gauss
            )
            k_total = k_total + k_patch
            f_total += f_patch
        return k_total, f_total, mapper

    patch = mp_or_patch
    basis = patch.basis()
    n_dofs = basis.size()
    mapper = _IdentityMapper(n_dofs)
    k_mat, f_vec = assemble_poisson_patch(patch, basis, f_func, mapper, 0, n_dofs, n_gauss)
    return k_mat, f_vec, mapper


def assemble_boundary_integral(patch, basis, side, func, n_gauss=None):
    n_dofs = basis.size()
    p = max(basis.degree(0), basis.degree(1))
    if n_gauss is None:
        n_gauss = p + 2

    gp_1d, gw_1d = gauss_legendre_01(n_gauss)

    if side in ("west", "east"):
        fixed_val = 0.0 if side == "west" else 1.0
        fixed_dir = 0
        breaks = sorted(set(basis.knots(1).get()))
        tangent_col = 1
    else:
        fixed_val = 0.0 if side == "south" else 1.0
        fixed_dir = 1
        breaks = sorted(set(basis.knots(0).get()))
        tangent_col = 0

    f_bdry = np.zeros(n_dofs)

    for t_lo, t_hi in zip(breaks[:-1], breaks[1:]):
        h_t = t_hi - t_lo
        for i_gp in range(n_gauss):
            t_q = t_lo + h_t * gp_1d[i_gp]
            w_q = gw_1d[i_gp] * h_t

            if fixed_dir == 0:
                pt = np.array([[fixed_val], [t_q]])
            else:
                pt = np.array([[t_q], [fixed_val]])

            act = basis.active(pt).flatten()
            n_vals = basis.eval(pt).flatten()
            jac = patch.jacobian(pt)
            tangent = jac[:, tangent_col]
            ds = np.linalg.norm(tangent) * w_q
            xy = patch.eval(pt).flatten()
            f_val = func(xy[0], xy[1])

            for a_idx in range(len(act)):
                f_bdry[act[a_idx]] += f_val * n_vals[a_idx] * ds

    return f_bdry


def assemble_boundary_mass(patch, basis, side, alpha_func, n_gauss=None):
    n_dofs = basis.size()
    p = max(basis.degree(0), basis.degree(1))
    if n_gauss is None:
        n_gauss = p + 2

    gp_1d, gw_1d = gauss_legendre_01(n_gauss)

    if side in ("west", "east"):
        fixed_val = 0.0 if side == "west" else 1.0
        fixed_dir = 0
        breaks = sorted(set(basis.knots(1).get()))
        tangent_col = 1
    else:
        fixed_val = 0.0 if side == "south" else 1.0
        fixed_dir = 1
        breaks = sorted(set(basis.knots(0).get()))
        tangent_col = 0

    rows, cols, vals = [], [], []

    for t_lo, t_hi in zip(breaks[:-1], breaks[1:]):
        h_t = t_hi - t_lo
        for i_gp in range(n_gauss):
            t_q = t_lo + h_t * gp_1d[i_gp]
            w_q = gw_1d[i_gp] * h_t

            if fixed_dir == 0:
                pt = np.array([[fixed_val], [t_q]])
            else:
                pt = np.array([[t_q], [fixed_val]])

            act = basis.active(pt).flatten()
            n_vals = basis.eval(pt).flatten()
            jac = patch.jacobian(pt)
            tangent = jac[:, tangent_col]
            ds = np.linalg.norm(tangent) * w_q
            xy = patch.eval(pt).flatten()
            alpha_val = alpha_func(xy[0], xy[1])

            for a in range(len(act)):
                for b in range(len(act)):
                    rows.append(act[a])
                    cols.append(act[b])
                    vals.append(alpha_val * n_vals[a] * n_vals[b] * ds)

    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()


def assemble_poisson_variable(patch, basis, lam_func, f_func, n_gauss=None):
    n_dofs = basis.size()
    p = max(basis.degree(0), basis.degree(1))
    if n_gauss is None:
        n_gauss = p + 2

    gp_1d, gw_1d = gauss_legendre_01(n_gauss)
    breaks_0 = sorted(set(basis.knots(0).get()))
    breaks_1 = sorted(set(basis.knots(1).get()))

    use_gismo_lam = hasattr(lam_func, "eval") and hasattr(lam_func, "domainDim")
    use_gismo_f = hasattr(f_func, "eval") and hasattr(f_func, "domainDim")

    rows, cols, vals_k = [], [], []
    f_vec = np.zeros(n_dofs)

    for xi_lo, xi_hi in zip(breaks_0[:-1], breaks_0[1:]):
        h_xi = xi_hi - xi_lo
        for eta_lo, eta_hi in zip(breaks_1[:-1], breaks_1[1:]):
            h_eta = eta_hi - eta_lo
            for i_gp in range(n_gauss):
                xi_q = xi_lo + h_xi * gp_1d[i_gp]
                for j_gp in range(n_gauss):
                    eta_q = eta_lo + h_eta * gp_1d[j_gp]
                    w_q = gw_1d[i_gp] * gw_1d[j_gp] * h_xi * h_eta
                    pt = np.array([[xi_q], [eta_q]])

                    act = basis.active(pt).flatten()
                    n_act = len(act)
                    n_vals = basis.eval(pt).flatten()
                    d_vals = basis.deriv(pt).flatten().reshape(n_act, 2)

                    jac = patch.jacobian(pt)
                    det_j = np.linalg.det(jac)
                    jinv_t = np.linalg.inv(jac).T
                    grad_phys = (jinv_t @ d_vals.T).T

                    xy = patch.eval(pt)
                    factor = np.abs(det_j) * w_q

                    lam_val = (
                        lam_func.eval(xy).flatten()[0]
                        if use_gismo_lam
                        else lam_func(xy[0, 0], xy[1, 0])
                    )
                    k_local = lam_val * (grad_phys @ grad_phys.T) * factor
                    for a in range(n_act):
                        for b in range(n_act):
                            rows.append(act[a])
                            cols.append(act[b])
                            vals_k.append(k_local[a, b])

                    f_val = (
                        f_func.eval(xy).flatten()[0]
                        if use_gismo_f
                        else f_func(xy[0, 0], xy[1, 0])
                    )
                    for a in range(n_act):
                        f_vec[act[a]] += f_val * n_vals[a] * factor

    k_mat = sparse.coo_matrix((vals_k, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return k_mat, f_vec
