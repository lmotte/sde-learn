import numpy as np


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
        self.T = None
        self.k_t = None
        self.rho = None
        self.n_tr = None
        self.T_tr = None
        self.X_tr = None
        self.K_t_inv = None
        self.x_dim = None

    def fit(self, T_tr, X_tr):
        """
        Fit the model using the provided training data.
        """
        self.n_tr = X_tr.shape[0]
        self.x_dim = X_tr.shape[-1]
        self.T_tr = T_tr
        self.X_tr = X_tr
        K_t = self.k_t(T_tr, T_tr)
        self.K_t_inv = np.linalg.inv(K_t)

    def predict(self, T_te, X_te):
        """
        Predict the probability density values using the fitted model.
        """
        if self.X_tr is not None:
            K_t_te_tr = self.k_t(T_te, self.T_tr)
            X_tr_concat = self.X_tr.reshape((-1, self.x_dim))
            X_te_concat = X_te.reshape((-1, self.x_dim))
            n_te = X_te_concat.shape[0]
            R_i_te_tr = self.rho(X_tr_concat, X_te_concat)
            R_i_te_tr_deconcat = R_i_te_tr.reshape((self.n_tr, -1, n_te))
            G_i_te = np.mean(R_i_te_tr_deconcat, axis=0)
            p_hat = np.dot(K_t_te_tr, np.dot(self.K_t_inv, G_i_te))
            return p_hat
        else:
            raise ValueError("Fit the model first using the 'fit' method.")
