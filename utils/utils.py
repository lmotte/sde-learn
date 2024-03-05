import numpy as np


def rho(X, Y, mu):
    """
    Evaluate the function rho(X_i, Y_i) for each pair (X_i, Y_i) between X and Y.

    Parameters:
    - X (numpy.ndarray): Array of shape (nx, n) containing the points X_i.
    - Y (numpy.ndarray): Array of shape (ny, n) containing the points Y_i.
    - mu (float): Parameter of rho.

    Returns:
    - numpy.ndarray: Array of shape (nx, ny) containing the pairwise evaluation.

    """
    _, n = X.shape

    # Expand dimensions to enable subtraction
    X_expanded = X[:, np.newaxis, :]
    Y_expanded = Y[np.newaxis, :, :]

    # Calculate Euclidean distance between each pair of elements
    distances = np.linalg.norm(X_expanded - Y_expanded, axis=2)

    # Calculate the resulting matrix
    rho_xy = mu**n * (2 * np.pi) ** (-n / 2) * np.exp(-0.5 * mu**2 * distances**2)

    return rho_xy


def rho_partial_derivative(X, Y, mu, k):
    """
    Compute the partial derivative of the function rho with respect to x_k
    for all pairs (X_i, Y_j)  of rows of matrices X and Y.

    Parameters:
    - X (numpy.ndarray): Array of shape (nx, n) containing the points X_i.
    - Y (numpy.ndarray): Array of shape (ny, n) containing the points Y_i.
    - mu (float): Parameter of rho.
    - k (int): Index for the partial derivative (1 <= k < n)

    Returns:
        numpy.ndarray: Array of shape (nx, ny) containing the pairwise evaluation.
    """
    _, n = X.shape

    # Expand dimensions to enable subtraction
    X_exp = X[:, np.newaxis, :]
    Y_exp = Y[np.newaxis, :, :]

    # Calculate Euclidean distance between each pair of elements
    distances_squared = np.linalg.norm(X_exp - Y_exp, axis=2) ** 2

    # Compute the partial derivative matrix w.r.t. x_k
    d_rho_xy = (
        -(mu ** (n + 2))
        * (2 * np.pi) ** (-n / 2)
        * (X_exp - Y_exp)[:, :, k]
        * np.exp(-0.5 * mu**2 * distances_squared)
    )

    return d_rho_xy


def rho_second_order_partial_derivative(X, Y, mu, k, q=None):
    """
    Compute the second-order partial or mixed partial derivatives of the function rho
    for all pairs (X_i, Y_j) of rows of matrices X and Y.

    Parameters:
    - X (numpy.ndarray): Array of shape (nx, n) containing the points X_i.
    - Y (numpy.ndarray): Array of shape (ny, n) containing the points Y_i.
    - mu (float): Parameter of rho.
    - k: Index for the partial derivative (1 <= k <= n)
    - q: Index for the second variable in the mixed partial derivative (1 <= q <= n), optional

    Returns:
    - numpy.ndarray: Array of shape (nx, ny) containing the pairwise evaluation.
    """
    _, n = X.shape

    # Expand dimensions to enable subtraction
    X_expanded = X[:, np.newaxis, :]
    Y_expanded = Y[np.newaxis, :, :]

    # Calculate Euclidean distance squared between each pair of elements
    distances_squared = np.linalg.norm(X_expanded - Y_expanded, axis=2) ** 2

    if q is None or k == q:
        # Compute the second-order partial derivative matrix
        d2_rho_xy = (
            mu ** (n + 2)
            * (2 * np.pi) ** (-n / 2)
            * (mu**2 * (X_expanded - Y_expanded)[:, :, k] ** 2 - 1)
            * np.exp(-0.5 * mu**2 * distances_squared)
        )
    else:
        # Compute the second-order mixed partial derivative matrix
        d2_rho_xy = (
            mu ** (n + 4)
            * (2 * np.pi) ** (-n / 2)
            * (X_expanded - Y_expanded)[:, :, k]
            * (X_expanded - Y_expanded)[:, :, q]
            * np.exp(-0.5 * mu**2 * distances_squared)
        )

    return d2_rho_xy


def mse(p1, p2, T, x_dim, R, n_t, n_x):
    """
    Estimate the L2 norm of the difference between two functions p1 and p2
    over the temporal domain [0, T] and spatial domain R^n.

    Parameters:
    - p1 (function): First function p1(t, x).
    - p2 (function): Second function p2(t, x).
    - T (float): Upper limit of the temporal domain.
    - x_dim (int): Dimension of the spatial domain.
    - R (float): Spatial domain range.
    - n_t (int): Number of points in the temporal domain.
    - n_x (int): Number of points in the spatial domain.

    Returns:
    - float: L2 norm of the difference between p1 and p2.
    """

    # Generate n_t points t in [0, T]
    T_te = np.random.uniform(0, T, size=n_t).reshape(-1, 1)

    # Generate n_x points x in [-R, R]^n
    X_te = np.random.uniform(-R, R, size=(n_x, x_dim))

    # Compute MSE over T_grid and X_te
    M = p1(T_te, X_te.T) - p2(T_te, X_te.T)
    MSE = 1 / (n_t * n_x) * np.linalg.norm(M) ** 2

    return MSE


def cartesian_products_of_rows(A, B):
    """Compute the cartesian product of the rows of A and B

    Example
    =======

    A:           B:
    [[1,2,3],       [[1,2],
    [4,5,6],        [3,4]]
    [7,8,9]]

    Returns:

    [[1,2,3,1,2],
    [1,2,3,3,4],
    [4,5,6,1,2],
    [4,5,6,3,4],
    [7,8,9,1,2],
    [7,8,9,3,4]]
    """

    A_repeated = np.repeat(A, repeats=B.shape[0], axis=0)
    B_tiled = np.tile(B, (A.shape[0], 1))
    cartesian_product = np.hstack((A_repeated, B_tiled))
    return cartesian_product
