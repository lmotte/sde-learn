import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from utils.utils import cartesian_products_of_rows
import cvxopt as opt


class FPEstimator:
    """
    FPEstimator Class

    Overview: The FPEstimator class is designed for estimating the coefficients of a
    Stochastic Differential Equation (SDE) from a given probability density by matching of the Fokker-Planck equation.
    """

    def __init__(self):
        """
        Initialize an instance of the FPEstimator class.
        """

        self.mu = None
        self.T = None
        self.n_t = None
        self.n = None
        self.T_fp = None
        self.Z_fp = None
        self.n_x = None
        self.n_z = None
        self.P = None
        self.d_P = None
        self.d_2_P = None
        self.d_t = None
        self.K_tilde_inv = None
        self.R_pos_sigma = None
        self.eps = None
        self.n_pos = None
        self.idx_pos = None

        # Hyperparameters
        self.la = None
        self.be = None
        self.gamma_z = None
        self.kde = None
        self.c_kernel = None

    def fit(self, T_fp, X_fp, p):
        """
        Fit the model with training data and a kernel density estimator.

        Parameters:
        - T_fp (numpy.ndarray): Time training points, shape (n_t).
        - X_fp (numpy.ndarray): Position training points, shape (n_x, d_x) or (n_x, d_x, n_t)
        - p(t, x) (object): probability density.

        Computes for model fitting:
        - Cartesian products of T_fp and X_fp.
        - Gram matrices and their derivatives.
        - values of p and its partial derivatives on the train set.
        - Least-squares solution : inverse of the regularized Gram matrix.
        """

        self.kde = p

        if X_fp.ndim == 2:
            self.Z_fp = cartesian_products_of_rows(T_fp, X_fp)
            self.n_x, self.n = X_fp.shape
            self.n_t = T_fp.shape[0]
            self.n_z = self.n_t * self.n_x
            self.T_fp = T_fp
            self.P, self.d_P, self.d_2_P, self.d_t = p(T_fp, X_fp, partial=True)

        elif X_fp.ndim == 3:
            self.n_x, self.n = X_fp.shape[0], X_fp.shape[1]
            self.n_t = T_fp.shape[0]
            self.n_z = self.n_t * self.n_x
            self.P = np.zeros((self.n_t, self.n_x))
            self.d_P = np.zeros((self.n, self.n_t, self.n_x))
            self.d_2_P = np.zeros((self.n, self.n, self.n_t, self.n_x))
            self.d_t = np.zeros((self.n_t, self.n_x))
            self.Z_fp = np.zeros((self.n_z, self.n + 1))
            for i in range(self.n_t):
                P_ti, d_P_ti, d_2_P_ti, d_t_ti = p(
                    T_fp[i : i + 1], X_fp[:, :, i], partial=True
                )
                (
                    self.P[i : i + 1, :],
                    self.d_P[:, i : i + 1, :],
                    self.d_2_P[:, :, i : i + 1, :],
                    self.d_t[i, :],
                ) = (P_ti, d_P_ti, d_2_P_ti, d_t_ti)
                self.Z_fp[
                    i * self.n_x : (i + 1) * self.n_x
                ] = cartesian_products_of_rows(T_fp[i : i + 1], X_fp[:, :, i])

        K, d_K, d_2_K, d_3_K, d_4_K = self.compute_grams_K_tr(self.Z_fp)

        K_tilde = self.compute_K_tilde(
            K, d_K, d_2_K, d_3_K, d_4_K, self.P, self.d_P, self.d_2_P
        )

        self.K_tilde_inv = np.linalg.inv(
            K_tilde + self.n_z * self.la / (self.T * self.be) * np.eye(K_tilde.shape[0])
        )

        self.n_pos = self.Z_fp.shape[0]
        self.idx_pos = np.random.choice(
            self.Z_fp.shape[0], size=self.n_pos, replace=False
        )

        K_pos, d_K_pos, d_2_K_pos = self.compute_grams_K_te(self.Z_fp[self.idx_pos])
        R_pos_b, R_pos_sigma = self.compute_R_te(
            K_pos, d_K_pos, d_2_K_pos, self.P, self.d_P, self.d_2_P
        )
        self.R_pos_sigma = R_pos_sigma
        d_t = self.d_t.reshape((-1, 1))
        self.mu = self.n_z * self.la / (self.T * self.be)
        self.eps = self.optimal_gamma(
            self.mu, K_pos[self.idx_pos, :], self.R_pos_sigma, self.K_tilde_inv, d_t
        )

    def predict(self, T_te, X_te, thresholding=False):
        """
        Predict values of the drift and diffusion coefficients using the fitted model.
        """

        Z_te = cartesian_products_of_rows(T_te, X_te)
        K_te, d_K_te, d_2_K_te = self.compute_grams_K_te(Z_te)
        R_te_b, R_te_sigma = self.compute_R_te(
            K_te, d_K_te, d_2_K_te, self.P, self.d_P, self.d_2_P
        )

        # Reshape
        d_t = self.d_t.reshape((-1, 1))

        # Compute predictions without positivity constraints
        M = d_t.T.dot(self.K_tilde_inv)
        B_pred = M.dot(R_te_b)
        S_pred = M.dot(R_te_sigma)

        # Compute predictions with positivity constraints
        mu = self.n_z * self.la / (self.T * self.be)
        add_S = (1 / mu) * self.eps.T.dot(
            K_te[self.idx_pos, :]
            - self.R_pos_sigma.T.dot(self.K_tilde_inv).dot(R_te_sigma)
        )
        add_B = -(1 / mu) * self.eps.T.dot(self.R_pos_sigma.T).dot(
            self.K_tilde_inv
        ).dot(R_te_b)
        S_pos = S_pred + add_S
        B_pos = B_pred + add_B

        if thresholding:
            S_pos[S_pos < 0] = 0
            S_pred[S_pred < 0] = 0
            S_pred = S_pred ** (1 / 2)
            S_pos = S_pos ** (1 / 2)

        return B_pos, S_pos, B_pred, S_pred

    def optimal_gamma(self, mu, K, R_tr_sigma, M, d):
        # Calculate A and b based on the given formulas
        A0 = R_tr_sigma.T @ M @ R_tr_sigma
        A = (mu**-1) * (K - A0)
        b = 2 * R_tr_sigma.T @ M @ d

        # Convert A, b to CVXOPT matrices
        P = opt.matrix(2 * A)
        q = opt.matrix(b)

        # Define the inequality constraints (Gx <= h)
        n = b.size
        G = opt.matrix(-np.eye(n))
        h = opt.matrix(np.zeros((n, 1)))

        # Solve the quadratic program
        opt.solvers.options["show_progress"] = False
        solution = opt.solvers.qp(P, q, G, h)
        gamma_optimal = np.array(solution["x"]).flatten()

        return gamma_optimal

    def compute_mse_tr(self):
        n_t = self.n_t
        Z_tr = self.Z_fp

        n_tr = Z_tr.shape[0]
        K_te, d_K_te, d_2_K_te, d_3_K_te, d_4_K_te = self.compute_grams_K_tr(Z_tr)

        K_tilde_tr = self.compute_K_tilde(
            K_te, d_K_te, d_2_K_te, d_3_K_te, d_4_K_te, self.P, self.d_P, self.d_2_P
        )

        d_p_tr_direct = self.d_t
        d_p_tr_kolmogorov = K_tilde_tr.dot(self.K_tilde_inv).dot(
            self.d_t.reshape((-1, 1))
        )
        L = d_p_tr_direct.reshape((-1, 1)) - d_p_tr_kolmogorov
        MSE = 1 / n_tr * np.linalg.norm(L) ** 2
        d_p_tr_kolmogorov = d_p_tr_kolmogorov.reshape((n_t, -1))

        return d_p_tr_kolmogorov, d_p_tr_direct, MSE

    def compute_mse_te(self, T_te, X_te):
        n_t = T_te.shape[0]
        Z_te = cartesian_products_of_rows(T_te, X_te)
        n_te = Z_te.shape[0]
        K_te, d_K_te, d_2_K_te, d_3_K_te, d_4_K_te = self.compute_grams_K_te(
            Z_te, full=True
        )
        P_te, d_P_te, d_2_P_te, d_p_te_direct = self.kde(T_te, X_te, partial=True)

        K_tilde_te = self.compute_K_tilde_te(
            K_te,
            d_K_te,
            d_2_K_te,
            d_3_K_te,
            d_4_K_te,
            self.P,
            self.d_P,
            self.d_2_P,
            P_te,
            d_P_te,
            d_2_P_te,
        )

        d_p_te_kolmogorov = K_tilde_te.T.dot(self.K_tilde_inv).dot(
            self.d_t.reshape((-1, 1))
        )

        L = d_p_te_direct.reshape((-1, 1)) - d_p_te_kolmogorov
        MSE = 1 / n_te * np.linalg.norm(L) ** 2

        d_p_te_kolmogorov = d_p_te_kolmogorov.reshape((n_t, -1))
        norms = (
            1 / n_te * np.linalg.norm(d_p_te_kolmogorov) ** 2,
            1 / n_te * np.linalg.norm(d_p_te_direct.reshape((-1, 1))) ** 2,
        )

        return d_p_te_kolmogorov, d_p_te_direct, MSE, norms

    def compute_K_tilde(self, K, d_K, d_2_K, d_3_K, d_4_K, P, d_P, d_2_P):
        """
        Compute the Gram matrix tilde K (See, Lemma 4.3).

        Parameters:
        - K: numpy array, Gram matrix
        - dK, d2K, d3K, d4K: numpy arrays, partial derivatives Gram matrices
        - P, dP, d2P: numpy arrays, partial derivatives probability density

        Returns:
        - tilde_K: numpy array
        """

        tilde_K = np.zeros(K.shape)

        # Concatenate the probability density predictions
        P = P.reshape((-1, 1))
        d_P = d_P.reshape((self.n, -1, 1))
        d_2_P = d_2_P.reshape((self.n, self.n, -1, 1))

        # Add first order partial derivative terms
        for i in range(self.n):
            K_i, K_ii, P_i = d_K[i], d_2_K[i, i], d_P[i]
            tilde_K_i = (
                K * np.outer(P_i, P_i)
                + K_i * np.outer(P, P_i)
                + (-K_i) * np.outer(P_i, P)
                + (-K_ii) * np.outer(P, P)
            )
            tilde_K += tilde_K_i

        # Add second order partial derivative terms
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        K_i, K_j = d_K[i], d_K[j]
                        K_k, K_l = d_K[k], d_K[l]
                        K_ij, K_il = d_2_K[i, j], d_2_K[i, l]
                        K_kl, K_ki = d_2_K[k, l], d_2_K[k, i]
                        K_kj, K_jl = d_2_K[k, j], d_2_K[j, l]
                        K_ki_j, K_kl_j = d_3_K[k, i, j], d_3_K[k, l, j]
                        K_kl_i, K_li_j = d_3_K[k, l, i], d_3_K[l, i, j]
                        K_kl_ij = d_4_K[k, l, i, j]
                        P_i, P_j, P_ij = d_P[i], d_P[j], d_2_P[i, j]
                        P_k, P_l, P_kl = d_P[k], d_P[l], d_2_P[k, l]

                        tilde_K_kl_ij = (
                            K * np.outer(P_kl, P_ij)
                            + K_k * np.outer(P_l, P_ij)
                            + K_l * np.outer(P_k, P_ij)
                            + K_kl * np.outer(P, P_ij)
                            + (-K_i) * np.outer(P_kl, P_j)
                            + (-K_ki) * np.outer(P_l, P_j)
                            + (-K_il) * np.outer(P_k, P_j)
                            + K_kl_i * np.outer(P, P_j)
                            + (-K_j) * np.outer(P_kl, P_i)
                            + (-K_kj) * np.outer(P_l, P_i)
                            + (-K_jl) * np.outer(P_k, P_i)
                            + K_kl_j * np.outer(P, P_i)
                            + K_ij * np.outer(P_kl, P)
                            + (-K_ki_j) * np.outer(P_l, P)
                            + (-K_li_j) * np.outer(P_k, P)
                            + K_kl_ij * np.outer(P, P)
                        )

                        tilde_K += (1 / 4) * tilde_K_kl_ij

        return tilde_K

    def compute_grams_K_tr(self, Z_tr):
        Z_tr = Z_tr.copy()

        K = rbf_kernel(Z_tr, Z_tr, self.gamma_z) + self.c_kernel

        # Calculate difference between each pair of elements
        X_tr = Z_tr[:, 1:]
        D_x = -(X_tr[np.newaxis, :, :] - X_tr[:, np.newaxis, :])

        # Apply recursive formulas to compute partial derivatives Gram matrices
        n_z = Z_tr.shape[0]
        d_K = np.zeros((self.n, n_z, n_z))
        d_2_K = np.zeros((self.n, self.n, n_z, n_z))
        d_3_K = np.zeros((self.n, self.n, self.n, n_z, n_z))
        d_4_K = np.zeros((self.n, self.n, self.n, self.n, n_z, n_z))
        for i in range(self.n):
            d_K[i] = -2 * self.gamma_z * D_x[:, :, i] * K
            for j in range(self.n):
                d_2_K[i, j] = -2 * self.gamma_z * (D_x[:, :, j] * d_K[i] + (i == j) * K)

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    K_k_ij = (
                        -2
                        * self.gamma_z
                        * (
                            D_x[:, :, j] * d_2_K[i, k]
                            + (k == j) * d_K[i]
                            + (i == j) * d_K[k]
                        )
                    )
                    K_ij_k = -K_k_ij
                    d_3_K[i, j, k] = K_ij_k

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        K_kl_ij = (
                            -2
                            * self.gamma_z
                            * (
                                -D_x[:, :, j] * d_3_K[k, l, i]
                                + (l == j) * d_2_K[i, k]
                                + (k == j) * d_2_K[i, l]
                                + (i == j) * d_2_K[k, l]
                            )
                        )
                        d_4_K[k, l, i, j] = K_kl_ij

        return K, d_K, d_2_K, d_3_K, d_4_K

    def compute_grams_K_te(self, Z_te, full=False):
        Z_fp = self.Z_fp.copy()
        Z_te = Z_te.copy()
        K_te = rbf_kernel(Z_fp, Z_te, self.gamma_z) + self.c_kernel
        n_te = Z_te.shape[0]

        # Calculate difference between each pair of elements
        X_te = Z_te[:, 1:]
        X_fp = Z_fp[:, 1:]
        D_x = -(X_te[np.newaxis, :, :] - X_fp[:, np.newaxis, :])

        # Apply recursive formulas to compute partial derivatives Gram matrices
        n_z = self.n_t * self.n_x
        d_K_te = np.zeros((self.n, n_z, n_te))
        d_2_K_te = np.zeros((self.n, self.n, n_z, n_te))
        for i in range(self.n):
            d_K_te[i] = -2 * self.gamma_z * D_x[:, :, i] * K_te
            for j in range(self.n):
                d_2_K_te[i, j] = (
                    -2 * self.gamma_z * (D_x[:, :, j] * d_K_te[i] + (i == j) * K_te)
                )

        if not full:
            return K_te, d_K_te, d_2_K_te
        else:
            d_3_K_te = np.zeros((self.n, self.n, self.n, n_z, n_te))
            d_4_K_te = np.zeros((self.n, self.n, self.n, self.n, n_z, n_te))

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        K_k_ij = (
                            -2
                            * self.gamma_z
                            * (
                                D_x[:, :, j] * d_2_K_te[i, k]
                                + (k == j) * d_K_te[i]
                                + (i == j) * d_K_te[k]
                            )
                        )
                        K_ij_k = -K_k_ij
                        d_3_K_te[i, j, k] = K_ij_k

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        for l in range(self.n):
                            K_kl_ij = (
                                -2
                                * self.gamma_z
                                * (
                                    -D_x[:, :, j] * d_3_K_te[k, l, i]
                                    + (l == j) * d_2_K_te[i, k]
                                    + (k == j) * d_2_K_te[i, l]
                                    + (i == j) * d_2_K_te[k, l]
                                )
                            )
                            d_4_K_te[k, l, i, j] = K_kl_ij

            return K_te, d_K_te, d_2_K_te, d_3_K_te, d_4_K_te

    def compute_R_te(self, K_te, d_K_te, d_2_K_te, P, d_P, d_2_P):
        n_te = K_te.shape[1]
        n_z = self.n_x * self.n_t

        # Concatenate the probability density predictions
        P = P.reshape((-1, 1))
        d_P = d_P.reshape((self.n, -1, 1))
        d_2_P = d_2_P.reshape((self.n, self.n, -1, 1))

        R_te_b = np.zeros((self.n, n_z, n_te))
        R_te_sigma = np.zeros((n_z, n_te))

        for i in range(self.n):
            P_i = d_P[i]
            R_te_b_i = K_te * np.outer(P_i, np.ones(n_te).T) + d_K_te[i] * np.outer(
                P, np.ones(n_te).T
            )
            R_te_b[i] = -R_te_b_i
            for j in range(self.n):
                P_j, P_ij = d_P[j], d_2_P[i, j]
                R_te_sigma_ij = K_te * np.outer(P_ij, np.ones(n_te).T)
                R_te_sigma_ij += d_K_te[i] * np.outer(P_j, np.ones(n_te).T)
                R_te_sigma_ij += d_K_te[j] * np.outer(P_i, np.ones(n_te).T)
                R_te_sigma_ij += d_2_K_te[i, j] * np.outer(P, np.ones(n_te).T)
                R_te_sigma += 1 / 2 * R_te_sigma_ij

        return R_te_b, R_te_sigma

    def compute_K_tilde_te(
        self,
        K_te,
        d_K_te,
        d_2_K_te,
        d_3_K_te,
        d_4_K_te,
        P,
        d_P,
        d_2_P,
        P_te,
        d_P_te,
        d_2_P_te,
    ):
        """
        Compute the Gram matrix tilde K (See, Lemma 4.3).

        Parameters:
        - K: numpy array, Gram matrix
        - dK, d2K, d3K, d4K: numpy arrays, partial derivatives Gram matrices
        - P, dP, d2P: numpy arrays, partial derivatives probability density

        Returns:
        - tilde_K: numpy array
        """

        tilde_K = np.zeros(K_te.shape)

        # Concatenate the probability density predictions
        P = P.reshape((-1, 1))
        d_P = d_P.reshape((self.n, -1, 1))
        d_2_P = d_2_P.reshape((self.n, self.n, -1, 1))
        P_te = P_te.reshape((-1, 1))
        d_P_te = d_P_te.reshape((self.n, -1, 1))
        d_2_P_te = d_2_P_te.reshape((self.n, self.n, -1, 1))

        # Add first order partial derivative terms
        for i in range(self.n):
            P_i = d_P[i]
            K_i, K_ii, P_i_te = d_K_te[i], d_2_K_te[i, i], d_P_te[i]
            tilde_K_i = (
                K_te * np.outer(P_i, P_i_te)
                + K_i * np.outer(P, P_i_te)
                + (-K_i) * np.outer(P_i, P_te)
                + (-K_ii) * np.outer(P, P_te)
            )
            tilde_K += tilde_K_i

        # Add second order partial derivative terms
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        K_i, K_j = d_K_te[i], d_K_te[j]
                        K_k, K_l = d_K_te[k], d_K_te[l]
                        K_ij, K_il = d_2_K_te[i, j], d_2_K_te[i, l]
                        K_kl, K_ki = d_2_K_te[k, l], d_2_K_te[k, i]
                        K_kj, K_jl = d_2_K_te[k, j], d_2_K_te[j, l]
                        K_ki_j, K_kl_j = d_3_K_te[k, i, j], d_3_K_te[k, l, j]
                        K_kl_i, K_li_j = d_3_K_te[k, l, i], d_3_K_te[l, i, j]
                        K_kl_ij = d_4_K_te[k, l, i, j]
                        P_i_te, P_j_te, P_ij_te = d_P_te[i], d_P_te[j], d_2_P_te[i, j]
                        P_k, P_l, P_kl = d_P[k], d_P[l], d_2_P[k, l]

                        tilde_K_kl_ij = (
                            K_te * np.outer(P_kl, P_ij_te)
                            + K_k * np.outer(P_l, P_ij_te)
                            + K_l * np.outer(P_k, P_ij_te)
                            + K_kl * np.outer(P, P_ij_te)
                            + (-K_i) * np.outer(P_kl, P_j_te)
                            + (-K_ki) * np.outer(P_l, P_j_te)
                            + (-K_il) * np.outer(P_k, P_j_te)
                            + K_kl_i * np.outer(P, P_j_te)
                            + (-K_j) * np.outer(P_kl, P_i_te)
                            + (-K_kj) * np.outer(P_l, P_i_te)
                            + (-K_jl) * np.outer(P_k, P_i_te)
                            + K_kl_j * np.outer(P, P_i_te)
                            + K_ij * np.outer(P_kl, P_te)
                            + (-K_ki_j) * np.outer(P_l, P_te)
                            + (-K_li_j) * np.outer(P_k, P_te)
                            + K_kl_ij * np.outer(P, P_te)
                        )

                        tilde_K += (1 / 4) * tilde_K_kl_ij

        return tilde_K
