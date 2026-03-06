"""
Parallel IGA Poisson assembly using mpi4py.

Usage: mpirun -np 4 python3.11 parallel_assembly.py [xml_file] [p_elevate] [num_refine]

Each rank assembles its share of elements, then results are gathered on rank 0.
"""
import sys
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpi4py import MPI

import pygismo as gs


def gauss_legendre_01(n):
    pts, wts = np.polynomial.legendre.leggauss(n)
    return 0.5 * (pts + 1), 0.5 * wts


def assemble_elements(patch, basis, f_func, elements, n_gauss):
    """Assemble stiffness and load for a list of elements."""
    n_dofs = basis.size()
    gp_1d, gw_1d = gauss_legendre_01(n_gauss)

    rows, cols, vals_K = [], [], []
    F = np.zeros(n_dofs)

    for (xi_lo, xi_hi, eta_lo, eta_hi) in elements:
        h_xi = xi_hi - xi_lo
        h_eta = eta_hi - eta_lo
        for i_gp in range(n_gauss):
            xi_q = xi_lo + h_xi * gp_1d[i_gp]
            for j_gp in range(n_gauss):
                eta_q = eta_lo + h_eta * gp_1d[j_gp]
                w_q = gw_1d[i_gp] * gw_1d[j_gp] * h_xi * h_eta
                pt = np.array([[xi_q], [eta_q]])

                act = basis.active(pt).flatten()
                n_act = len(act)
                N = basis.eval(pt).flatten()
                dN = basis.deriv(pt).flatten().reshape(n_act, 2)

                J = patch.jacobian(pt)
                detJ = np.linalg.det(J)
                Jinv_T = np.linalg.inv(J).T
                grad_phys = (Jinv_T @ dN.T).T

                xy = patch.eval(pt)
                factor = np.abs(detJ) * w_q

                K_local = grad_phys @ grad_phys.T * factor
                for a in range(n_act):
                    for b in range(n_act):
                        rows.append(act[a])
                        cols.append(act[b])
                        vals_K.append(K_local[a, b])

                f_val = f_func.eval(xy).flatten()[0]
                for a in range(n_act):
                    F[act[a]] += f_val * N[a] * factor

    K = sparse.coo_matrix((vals_K, (rows, cols)),
                          shape=(n_dofs, n_dofs)).tocsr()
    return K, F


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse arguments
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "geometries/square.xml"
    p_elevate = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    num_refine = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    xml_path = Path(xml_file)
    if not xml_path.exists():
        fallback = Path("geometries") / xml_file
        if fallback.exists():
            xml_path = fallback
        else:
            raise FileNotFoundError(f"Could not find geometry file: {xml_file}")

    # Every rank loads the same geometry (shared-nothing)
    mp_loc = gs.core.gsMultiPatch()
    gs.io.gsReadFile(str(xml_path), mp_loc)
    mp_loc.degreeElevate(p_elevate)
    mp_loc.uniformRefine(num_refine)

    patch = mp_loc.patch(0)
    basis = patch.basis()
    n_dofs = basis.size()
    p = max(basis.degree(0), basis.degree(1))
    n_gauss = p + 2

    f_func = gs.core.gsFunctionExpr('2*pi*pi*sin(pi*x)*sin(pi*y)', 2)

    # Build element list
    breaks_0 = sorted(set(basis.knots(0).get()))
    breaks_1 = sorted(set(basis.knots(1).get()))
    all_elements = [(xi_lo, xi_hi, eta_lo, eta_hi)
                    for xi_lo, xi_hi in zip(breaks_0[:-1], breaks_0[1:])
                    for eta_lo, eta_hi in zip(breaks_1[:-1], breaks_1[1:])]
    n_el = len(all_elements)

    # Partition elements across ranks (round-robin)
    my_elements = all_elements[rank::size]

    if rank == 0:
        print(f"MPI parallel assembly: {size} ranks, {n_el} elements, {n_dofs} DOFs, p={p}")
        print(f"  Rank 0 handles {len(my_elements)} elements")

    comm.Barrier()
    t0 = MPI.Wtime()

    # Each rank assembles its local contribution
    K_local, F_local = assemble_elements(patch, basis, f_func, my_elements, n_gauss)

    # Reduce: sum all local load vectors on rank 0
    F_global = np.zeros(n_dofs) if rank == 0 else None
    comm.Reduce(F_local, F_global, op=MPI.SUM, root=0)

    # Reduce stiffness matrix: convert to dense, sum, convert back
    # (For large problems, use distributed sparse formats instead)
    K_dense_local = K_local.toarray()
    K_dense_global = np.zeros_like(K_dense_local) if rank == 0 else None
    comm.Reduce(K_dense_local, K_dense_global, op=MPI.SUM, root=0)

    t_assembly = MPI.Wtime() - t0

    # Rank 0 solves the system
    if rank == 0:
        print(f"  Assembly time: {t_assembly:.4f}s")

        K_global = sparse.csr_matrix(K_dense_global)

        # Boundary DOFs
        n0 = basis.component(0).size()
        n1 = basis.component(1).size()
        bdry = set()
        for j in range(n1):
            bdry.add(j * n0)
            bdry.add(j * n0 + n0 - 1)
        for i in range(n0):
            bdry.add(i)
            bdry.add((n1 - 1) * n0 + i)
        bdry = sorted(bdry)
        inner = sorted(set(range(n_dofs)) - set(bdry))

        u = np.zeros(n_dofs)
        K_ff = K_global[np.ix_(inner, inner)]
        F_ff = F_global[inner]
        u[inner] = spsolve(K_ff.tocsc(), F_ff)

        # Compute L2 error
        u_exact = gs.core.gsFunctionExpr('sin(pi*x)*sin(pi*y)', 2)
        gp_err, gw_err = gauss_legendre_01(p + 3)
        err_sq = 0.0
        for xi_lo, xi_hi in zip(breaks_0[:-1], breaks_0[1:]):
            h_xi = xi_hi - xi_lo
            for eta_lo, eta_hi in zip(breaks_1[:-1], breaks_1[1:]):
                h_eta = eta_hi - eta_lo
                for ig in range(len(gp_err)):
                    xq = xi_lo + h_xi * gp_err[ig]
                    for jg in range(len(gp_err)):
                        eq = eta_lo + h_eta * gp_err[jg]
                        wq = gw_err[ig] * gw_err[jg] * h_xi * h_eta
                        pt = np.array([[xq], [eq]])
                        act = basis.active(pt).flatten()
                        N = basis.eval(pt).flatten()
                        u_h = np.dot(N, u[act])
                        J = patch.jacobian(pt)
                        dJ = np.abs(np.linalg.det(J))
                        xy = patch.eval(pt)
                        u_ex = u_exact.eval(xy).flatten()[0]
                        err_sq += (u_h - u_ex)**2 * dJ * wq

        print(f"  L2 error: {np.sqrt(err_sq):.4e}")
        print(f"  Solution range: [{u.min():.6f}, {u.max():.6f}]")


if __name__ == "__main__":
    main()
