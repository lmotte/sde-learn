import numpy as np
from utils.utils import rho
from sklearn.metrics.pairwise import rbf_kernel


class ProbaDensityEstimator:
    """
    ProbaDensityEstimator Class

    Overview:
    ---------
    The ProbaDensityEstimator class is designed for probability density estimation.
    It implements evaluations for p(t, x), and its i-th order partial derivatives.
    """

    def __init__(self):
        """
        Initialize an instance of the ProbaDensityEstimator class.
        """
        self.gamma_t = None
        self.L_t = None
        self.mu_x = None
        self.n_tr = None
        self.T_tr = None
        self.X_tr = None
        self.K_t_inv = None
        self.x_dim = None
        self.c_kernel = None

    def fit(self, T_tr, X_tr):
        """
        Fit the model using the provided training data.
        """
        self.n_tr = X_tr.shape[0]
        self.x_dim = X_tr.shape[-1]
        self.T_tr = T_tr
        self.X_tr = X_tr
        K_t = rbf_kernel(T_tr, T_tr, gamma=self.gamma_t) + self.c_kernel
        self.K_t_inv = np.linalg.inv(K_t + self.L_t * np.eye(K_t.shape[0]))

    def predict(self, T_te, X_te, partial=False):
        """
        Predict the probability density values using the fitted model.
        """
        if self.X_tr is not None:
            K_t_te_tr = rbf_kernel(T_te, self.T_tr, gamma=self.gamma_t) + self.c_kernel
            X_tr_concat = self.X_tr.reshape((-1, self.x_dim))
            X_te_concat = X_te.reshape((-1, self.x_dim))
            n_x_te = X_te_concat.shape[0]
            n_t_te = T_te.shape[0]
            R_tr_te = rho(X_tr_concat, X_te_concat, mu=self.mu_x)
            R_tr_te_deconcat = R_tr_te.reshape((self.n_tr, -1, n_x_te))
            G_te = np.mean(R_tr_te_deconcat, axis=0)
            P_hat = np.dot(K_t_te_tr, np.dot(self.K_t_inv, G_te))

            if not partial:
                return P_hat

            else:
                R_tr_te = rho(X_tr_concat, X_te_concat, mu=self.mu_x)

                # Calculate difference between each pair of elements
                D_x = X_te[np.newaxis, :, :] - X_tr_concat[:, np.newaxis, :]

                # Apply recursive formulas to compute partial derivatives Gram matrices
                d_P_hat = np.zeros((self.x_dim, n_t_te, n_x_te))
                d_2_P_hat = np.zeros((self.x_dim, self.x_dim, n_t_te, n_x_te))
                for i in range(self.x_dim):
                    R_i_tr_te = R_tr_te * (-self.mu_x**2 * D_x[:, :, i])
                    R_i_tr_te_deconcat = R_i_tr_te.reshape((self.n_tr, -1, n_x_te))
                    G_i_te = np.mean(R_i_tr_te_deconcat, axis=0)
                    d_P_hat[i] = np.dot(K_t_te_tr, np.dot(self.K_t_inv, G_i_te))

                    for j in range(self.x_dim):
                        R_ij_tr_te = (-self.mu_x**2) * (
                            D_x[:, :, j] * R_i_tr_te + (i == j) * R_tr_te
                        )
                        R_ij_tr_te_deconcat = R_ij_tr_te.reshape(
                            (self.n_tr, -1, n_x_te)
                        )
                        G_ij_te = np.mean(R_ij_tr_te_deconcat, axis=0)
                        d_2_P_hat[i, j] = np.dot(
                            K_t_te_tr, np.dot(self.K_t_inv, G_ij_te)
                        )

                # Calculate difference between each pair of elements
                D_t = T_te[np.newaxis, :, :] - self.T_tr[:, np.newaxis, :]
                d_K_t_te_tr = -2 * self.gamma_t * D_t[:, :, 0].T * K_t_te_tr
                # d_K_t_te_tr = K_t_te_tr
                d_t = np.dot(d_K_t_te_tr, np.dot(self.K_t_inv, G_te))

                return P_hat, d_P_hat, d_2_P_hat, d_t

        else:
            raise ValueError("Fit the model first using the 'fit' method.")

    def log_likelihood(self, P):
        """
        Compute the log likelihood.

        Parameters:
            P: A numpy array of probability density's values.

        """

        # Ensure all predicted densities are positive to avoid log(0)
        P = np.clip(P, a_min=1e-9, a_max=None)

        # Compute the log likelihood
        log_likelihood = np.sum(np.log(P))

        return log_likelihood
