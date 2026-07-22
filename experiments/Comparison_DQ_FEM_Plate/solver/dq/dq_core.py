import numpy as np

def chebyshev_lobatto_nodes(N, length_value=1.0):

    if N < 2:
        raise ValueError("chebyshev_lobatto_nodes: N must be >= 2.")
    j = np.arange(N)
    x = np.cos(np.pi * j / (N - 1))
    return 0.5 * length_value * (1.0 - x)

def dq_first_derivative_matrix(nodes):

    x = np.asarray(nodes, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("dq_first_derivative_matrix: need at least two nodes.")
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n))
    dX = x[:, None] - x[None, :]
    D = (c[:, None] * (1.0 / c)[None, :]) / (dX + np.eye(n))
    D = D - np.diag(D.sum(axis=1))
    return D

def dq_second_derivative_matrix(nodes):

    D1 = dq_first_derivative_matrix(nodes)
    return D1 @ D1

def build_2d_operators(x_nodes, y_nodes):

    Nx = np.asarray(x_nodes).size
    Ny = np.asarray(y_nodes).size
    D1x = dq_first_derivative_matrix(x_nodes)
    D2x = dq_second_derivative_matrix(x_nodes)
    D1y = dq_first_derivative_matrix(y_nodes)
    D2y = dq_second_derivative_matrix(y_nodes)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    return {
        "Dx": np.kron(D1x, Iy),
        "Dy": np.kron(Ix, D1y),
        "Dxx": np.kron(D2x, Iy),
        "Dyy": np.kron(Ix, D2y),
        "Dxy": np.kron(D1x, D1y),
        "D1x": D1x,
        "D2x": D2x,
        "D1y": D1y,
        "D2y": D2y,
    }

def quadrature_weights_for_nodes(nodes):

    x = np.asarray(nodes, dtype=float).ravel()
    n_nodes = x.size
    if n_nodes < 2:
        raise ValueError("quadrature_weights_for_nodes: need at least two nodes.")
    a = x[0]
    b = x[-1]
    n = n_nodes - 1
    if n == 1:
        return np.array([0.5 * (b - a), 0.5 * (b - a)])

    theta = np.pi * np.arange(n + 1) / n
    weights = np.zeros(n_nodes)
    interior = np.arange(1, n)
    v = np.ones(interior.size)

    if n % 2 == 0:
        edge_weight = 1.0 / (n ** 2 - 1.0)
        for k in range(1, n // 2):
            v = v - 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)
        v = v - np.cos(n * theta[interior]) / (n ** 2 - 1.0)
    else:
        edge_weight = 1.0 / (n ** 2)
        for k in range(1, (n + 1) // 2):
            v = v - 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)

    weights[0] = edge_weight
    weights[-1] = edge_weight
    weights[interior] = 2.0 * v / n
    return 0.5 * (b - a) * weights

def barycentric_weights(nodes):

    x = np.asarray(nodes, dtype=float).ravel()
    n = x.size
    w = np.ones(n)
    for j in range(n):
        diff = x[j] - np.delete(x, j)
        w[j] = 1.0 / np.prod(diff)
    return w

def interpolation_row(source_nodes, target):

    x = np.asarray(source_nodes, dtype=float).ravel()
    w = barycentric_weights(x)
    diff = target - x
    hit = np.where(np.abs(diff) < 1.0e-14)[0]
    row = np.zeros(x.size)
    if hit.size:
        row[hit[0]] = 1.0
    else:
        ratio = w / diff
        row = ratio / ratio.sum()
    return row
