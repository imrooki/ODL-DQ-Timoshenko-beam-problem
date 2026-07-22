import numpy as np

from .dq_core import interpolation_row

def extract_center_deflection(displacement_full, system):

    Nx, Ny, Ng = system["Nx"], system["Ny"], system["Ng"]
    a, b = system["a"], system["b"]
    Wblock = np.asarray(displacement_full, dtype=float).ravel()[2 * Ng:3 * Ng]
    Wgrid = Wblock.reshape(Nx, Ny)
    row_x = interpolation_row(system["x_nodes"], 0.5 * a)
    row_y = interpolation_row(system["y_nodes"], 0.5 * b)
    return float(row_x @ Wgrid @ row_y)

def navier_stiffness(E, nu, rho, h, kappa):
    G = E / (2.0 * (1.0 + nu))
    D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    return {
        "E": E, "nu": nu, "rho": rho, "h": h, "kappa": kappa, "G": G,
        "D": D, "D12": nu * D, "D66": (1.0 - nu) / 2.0 * D,
        "A": E * h / (1.0 - nu ** 2), "As": kappa * G * h,
        "I0": rho * h, "I2": rho * h ** 3 / 12.0,
    }

def navier_S(m, n, a, b, s):

    k = m * np.pi / a
    l = n * np.pi / b
    As, D, D12, D66 = s["As"], s["D"], s["D12"], s["D66"]
    return np.array([
        [As * (k ** 2 + l ** 2), As * k, As * l],
        [As * k, D * k ** 2 + D66 * l ** 2 + As, (D12 + D66) * k * l],
        [As * l, (D12 + D66) * k * l, D * l ** 2 + D66 * k ** 2 + As],
    ])

def navier_center_uniform(a, b, q, s, Mmax=39):

    wc = 0.0
    for m in range(1, Mmax + 1, 2):
        for n in range(1, Mmax + 1, 2):
            Qmn = 16.0 * q / (np.pi ** 2 * m * n)
            amp = np.linalg.solve(navier_S(m, n, a, b, s), np.array([Qmn, 0.0, 0.0]))
            wc += amp[0] * np.sin(m * np.pi / 2.0) * np.sin(n * np.pi / 2.0)
    return float(wc)

def navier_frequencies(a, b, s, Mmax=6, nmodes=6):

    Mmn = np.diag([s["I0"], s["I2"], s["I2"]])
    vals = []
    for m in range(1, Mmax + 1):
        for n in range(1, Mmax + 1):
            S = navier_S(m, n, a, b, s)
            ev = np.linalg.eigvals(np.linalg.solve(Mmn, S))
            vals.append(np.sqrt(np.min(ev.real)))
    vals = np.sort(np.array(vals))
    return vals[:nmodes]

def navier_step_center_history(t_array, a, b, q, s, Mmax=11):

    t = np.asarray(t_array, dtype=float).ravel()
    wc = np.zeros(t.size)
    Mmn = np.diag([s["I0"], s["I2"], s["I2"]])
    for m in range(1, Mmax + 1, 2):
        for n in range(1, Mmax + 1, 2):
            Qmn = 16.0 * q / (np.pi ** 2 * m * n)
            S = navier_S(m, n, a, b, s)
            evals, Phi = np.linalg.eig(np.linalg.solve(Mmn, S))
            om = np.sqrt(np.maximum(evals.real, 0.0))
            F = np.array([Qmn, 0.0, 0.0])
            sphase = np.sin(m * np.pi / 2.0) * np.sin(n * np.pi / 2.0)
            for r in range(3):
                phir = Phi[:, r].real
                mr = phir @ (Mmn @ phir)
                gr = (phir @ F) / mr
                wc += sphase * phir[0] * (gr / om[r] ** 2) * (1.0 - np.cos(om[r] * t))
    return wc

def navier_buckling_kx(a, b, s, Mmax=6):

    best = np.inf
    for m in range(1, Mmax + 1):
        for n in range(1, Mmax + 1):
            kx = m * np.pi / a
            S = navier_S(m, n, a, b, s)
            schur = S[0, 0] - S[0, 1:3] @ np.linalg.solve(S[1:3, 1:3], S[1:3, 0])
            kk = (schur / kx ** 2) * a ** 2 / (np.pi ** 2 * s["D"])
            best = min(best, kk)
    return float(best)
