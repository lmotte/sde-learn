import time
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from utils.utils import cartesian_products_of_rows, qp_solver


class FPEstimator:
    """
    FPEstimator class for estimating Stochastic Differential Equation (SDE) coefficients.

    This class implements the estimation of drift and diffusion coefficients of an SDE from a given
    probability density. The estimation is performed by mathching of the corresponding Fokker-Planck equation
    using kernel ridge regression techniques, including both standard and Nyström approximation methods.

    Attributes
    ----------
    la : float, optional
        Regularization parameter for ridge regression. Default is None.
    gamma_z : float, optional
        Parameter for the Radial Basis Function (RBF) kernel. Default is None.
    kde : callable, optional
        Kernel density estimator function. Default is None.
    c_kernel : float, optional
        Constant term added to the kernel function. Default is None.
        T : None or array-like
        Training time points, set during fitting.
    n : int
        Dimension of the SDE.
    n_t : int
        Number of time training points, set during fitting.
    n_x : int
        Number of spatial training points, set during fitting.
    n_z : int
        Total number of training data points (time-spatial).
    n_pc : None or int
        Number of positivity constraints, set during fitting.
    Z_tr : array-like
        Training data points (time-spatial), set during fitting.
    P_tr : array-like
        Probability densities at training points, set during fitting.
    d_P_tr : array-like
        First derivative of probability densities at training points, set during fitting.
    d_2_P_tr : array-like
        Second derivative of probability densities at training points, set during fitting.
    d_t_P_tr : array-like
        Time derivative of probability densities at training points, set during fitting.
    idx_pc : array-like
        Indices for positivity constraints, set during fitting.
    eps : array-like
        Corrections for positivity constraints, set during fitting.
    alpha : array-like
        Coefficients for drift and diffusion estimations without positivity constraints, set during fitting.
    alpha_pc : array-like
        Coefficients for drift and diffusion estimations with positivity constraints, set during fitting.
    idx_ny : array-like
        Indices used in Nyström approximation, set during fitting if Nyström approximation is used.

    Methods
    -------
    __init__(self)
        Initializes the FPEstimator instance.

    fit(self, T_tr, X_tr, p, nystrom=-1)
        Fits the model using provided training data and probability density function.

    [Other methods are not included here for brevity.]

    Examples
    --------
    fpe = FPEstimator()
    fpe.fit(T_tr, X_tr, p_density, nystrom=100)
    Fit the FPEstimator model using training data `T_tr`, `X_tr`, a probability density function `p_density`,
    and using Nyström approximation with 100 anchor points.
    """

    def __init__(self):
        """
        Initialize an instance of the FPEstimator class.
        """

        # Parameters.
        self.la = None
        self.gamma_z = None
        self.kde = None
        self.c_kernel = None

        # Attributes computed and save at train.
        self.T = None
        self.n = None
        self.n_t = None
        self.n_x = None
        self.n_z = None
        self.n_pc = None
        self.Z_tr = None
        self.P_tr = None
        self.d_P_tr = None
        self.d_2_P_tr = None
        self.d_t_P_tr = None
        self.idx_pc = None
        self.eps = None
        self.alpha = None
        self.alpha_pc = None
        self.idx_ny = None

    def fit(self, T_tr, X_tr, p, nystrom=-1):
        """
        Fit the model with training data and a kernel density estimator.

        Parameters
        ----------
        T_tr : numpy.ndarray
            Time training points with shape (n_t,).
        X_tr : numpy.ndarray
            Position training points with shape (n_x, n) for 2D input or (n_x, n, n_t) for 3D input, where `n_x` is
            the number of spatial points, `n` is the dimensionality of the points, and `n_t` is the number of time
            points.
        p : callable
            Probability density function, which takes time `t` and position `x` as inputs and returns the probability
            density value at those points.
        nystrom : int, optional
            Number of anchors for Nyström approximation. A value of -1 (default) indicates that no approximation is
            used, and standard KRR is performed.

        Computes for model fitting:
            Training set Z_tr defined as the cartesian products of T_tr and X_fp.
            Values of p and its partial derivatives on the train set Z_tr.
            Gram matrices and their derivatives.
            Least-squares solution: vector alpha of shape (n_tr, 1) (or (nystrom, 1) if approximation) of the kernel
            model, and vector eps of shape (n_pc, 1) (or (nystrom, 1) if approximation) for the positivity constraints
            correction.
        """

        # Build the Fokker-Planck training set Z_fp and predict the values of p and its derivatives on Z_fp.
        self.kde = p
        self.prepare_data(T_tr, X_tr)

        # Concatenate the probability density predictions.
        self.P_tr = self.P_tr.reshape((-1, 1))
        self.d_P_tr = self.d_P_tr.reshape((self.n, -1, 1))
        self.d_2_P_tr = self.d_2_P_tr.reshape((self.n, self.n, -1, 1))

        # KRR fitting.
        if nystrom == -1:
            # KRR standard fitting.
            self.fit_standard_krr()
        else:
            # KRR Nyström fitting.
            self.fit_nystrom_krr(nystrom)

    def fit_standard_krr(self):
        """
        Perform standard kernel ridge regression (KRR) fitting with positivity constraints.

        Computes and inverts the Gram matrices to obtain the KRR weights. It incorporates positivity constraints for
        the diffusion coefficient by solving an additional quadratic programming (QP) problem.
        """

        # Compute the Gram matrices.
        t0 = time.time()
        d_t_P_tr = self.d_t_P_tr.reshape((-1, 1))
        K_tr, d_K_tr, d_2_K_tr, d_3_K_tr, d_4_K_tr = self.compute_grams_K(
            self.Z_tr, full=True
        )
        K_tilde_tr = self.compute_K_tilde(
            K_tr,
            d_K_tr,
            d_2_K_tr,
            d_3_K_tr,
            d_4_K_tr,
            self.P_tr,
            self.d_P_tr,
            self.d_2_P_tr,
        )

        # Inversion.
        K_tilde_inv = np.linalg.inv(
            K_tilde_tr + self.n_z * self.la * np.eye(K_tilde_tr.shape[0])
        )
        print(f"Inversion time: {time.time() - t0}", flush=True)

        # KRR weights.
        self.alpha = d_t_P_tr.T.dot(K_tilde_inv)

        # Positivity constraints additional computations.
        t0 = time.time()
        self.n_pc = self.Z_tr.shape[0]
        self.idx_pc = np.random.choice(
            self.Z_tr.shape[0], size=self.n_pc, replace=False
        )

        K_tr_pc = K_tr[:, self.idx_pc]
        d_K_tr_pc = d_K_tr[:, :, self.idx_pc]
        d_2_K_tr_pc = d_2_K_tr[:, :, :, self.idx_pc]
        R_b_tr_pc, R_s_tr_pc = self.compute_R(
            K_tr_pc, d_K_tr_pc, d_2_K_tr_pc, self.P_tr, self.d_P_tr, self.d_2_P_tr
        )
        K_pc_pc = K_tr_pc[self.idx_pc, :]
        A0 = R_s_tr_pc.T @ K_tilde_inv @ R_s_tr_pc
        A = (self.la**-1) * (K_pc_pc - A0)
        b = 2 * R_s_tr_pc.T @ K_tilde_inv @ d_t_P_tr
        self.eps = (self.la**-1) * qp_solver(A, b)
        print(f"QP solving time: {time.time() - t0}", flush=True)
        self.alpha_pc = self.eps.T.dot(R_s_tr_pc.T).dot(K_tilde_inv)

    def fit_nystrom_krr(self, nystrom):
        """
        Perform kernel ridge regression (KRR) fitting with positivity constraints using the Nyström approximation.

        Parameters
        ----------
        nystrom : int
            Number of data points to use in the Nyström approximation. This parameter controls the size of the subset
            of training data used as landmarks for the approximation.
        """

        # Select a random subset of training data.
        t0 = time.time()
        self.idx_ny = np.random.choice(self.Z_tr.shape[0], size=nystrom, replace=False)

        # Compute the Gram matrices.
        (
            K_tr_ny,
            d_K_tr_ny,
            d_2_K_tr_ny,
            d_3_K_tr_ny,
            d_4_K_tr_ny,
        ) = self.compute_grams_K(self.Z_tr, self.Z_tr[self.idx_ny], full=True)
        K_tilde_tr_ny = self.compute_K_tilde(
            K_tr_ny,
            d_K_tr_ny,
            d_2_K_tr_ny,
            d_3_K_tr_ny,
            d_4_K_tr_ny,
            self.P_tr,
            self.d_P_tr,
            self.d_2_P_tr,
            self.P_tr[self.idx_ny],
            self.d_P_tr[:, self.idx_ny],
            self.d_2_P_tr[:, :, self.idx_ny],
        )
        K_tilde_ny_ny = K_tilde_tr_ny[self.idx_ny]

        # Inversion.
        M = np.linalg.inv(
            K_tilde_tr_ny.T.dot(K_tilde_tr_ny)
            + self.n_z * self.la * np.eye(K_tilde_ny_ny.shape[0])
        )
        print(f"Inversion time: {time.time() - t0}", flush=True)

        # Positivity constraints additional computations.
        t0 = time.time()
        self.n_pc = nystrom
        self.idx_pc = np.random.choice(nystrom, size=self.n_pc, replace=False)
        K_ny_pc = K_tr_ny[self.idx_ny, :][:, self.idx_pc]
        d_K_ny_pc = d_K_tr_ny[:, self.idx_ny, :][:, :, self.idx_pc]
        d_2_K_ny_pc = d_2_K_tr_ny[:, :, self.idx_ny, :][:, :, :, self.idx_pc]

        _, R_s_ny_pc = self.compute_R(
            K_ny_pc,
            d_K_ny_pc,
            d_2_K_ny_pc,
            self.P_tr[self.idx_pc],
            self.d_P_tr[:, self.idx_pc],
            self.d_2_P_tr[:, :, self.idx_pc],
        )
        A = R_s_ny_pc.T.dot(M).dot(R_s_ny_pc)
        d_t_P_tr = self.d_t_P_tr.reshape((-1, 1))
        b = 2 * R_s_ny_pc.T @ M @ K_tilde_tr_ny.T @ d_t_P_tr
        self.eps = qp_solver(A, b)
        print(f"QP solving time: {time.time() - t0}", flush=True)

        K_tilde_pc_ny = K_tilde_ny_ny[self.idx_pc, :]
        self.alpha = (
            d_t_P_tr.T.dot(K_tilde_tr_ny) + self.eps.T.dot(K_tilde_pc_ny)
        ).dot(M)

    def prepare_data(self, T_tr, X_tr):
        """
        Prepares the training data for the Fokker-Planck equation fitting process.

        This method constructs the training set `Z_tr` by taking the cartesian product of the time
        and spatial components of the training data. It then evaluates the probability density function `p`
        and its derivatives at these points.

        Parameters
        ----------
        T_tr : numpy.ndarray
            The time component of the training data, shaped as (n_t, ), where `n_t` is the number of time points.
        X_tr : numpy.ndarray
            The spatial component of the training data. Can be either 2D shaped as (n_x, n), where `n_x` is the
            number of spatial points and `n` is the dimension of each point, or 3D shaped as (n_x, n, n_t) for
            time-varying spatial data.

        Raises
        ------
        ValueError
            If `X_tr` has an unsupported dimensionality, indicating it is neither 2D nor 3D.

        """

        # Build the Fokker-Planck training set Z_fp and predict the values of p and its derivatives on Z_fp.
        # Reshape data and predict p on it.
        t0 = time.time()
        if X_tr.ndim == 2:
            self.n_x, self.n = X_tr.shape
            self.n_t = T_tr.shape[0]
            self.n_z = self.n_t * self.n_x
            self.Z_tr = cartesian_products_of_rows(T_tr, X_tr)
            self.P_tr, self.d_P_tr, self.d_2_P_tr, self.d_t_P_tr = self.kde(
                T_tr, X_tr, partial=True
            )
        elif X_tr.ndim == 3:
            self.n_x, self.n_t, self.n = X_tr.shape[0], X_tr.shape[1], X_tr.shape[2]
            self.n_z = self.n_t * self.n_x
            self.Z_tr = np.zeros((self.n_z, self.n + 1))
            self.P_tr = np.zeros((self.n_t, self.n_x))
            self.d_P_tr = np.zeros((self.n, self.n_t, self.n_x))
            self.d_2_P_tr = np.zeros((self.n, self.n, self.n_t, self.n_x))
            self.d_t_P_tr = np.zeros((self.n_t, self.n_x))
            for i in range(self.n_t):
                P_ti, d_P_ti, d_2_P_ti, d_t_ti = self.kde(
                    T_tr[i : i + 1], X_tr[:, i, :], partial=True
                )
                (
                    self.P_tr[i : i + 1, :],
                    self.d_P_tr[:, i : i + 1, :],
                    self.d_2_P_tr[:, :, i : i + 1, :],
                    self.d_t_P_tr[i, :],
                ) = (P_ti, d_P_ti, d_2_P_ti, d_t_ti)
                self.Z_tr[
                    i * self.n_x : (i + 1) * self.n_x
                ] = cartesian_products_of_rows(T_tr[i : i + 1], X_tr[:, i, :])
        else:
            raise ValueError("Unsupported dimension for X_tr.")
        print(f"Building Z and predicting p(Z), time: {time.time() - t0}", flush=True)

    def predict(self, T_te, X_te, thresholding=False):
        """
        Predict drift and diffusion coefficients for a given test dataset using the fitted model.

        Parameters
        ----------
        T_te : numpy.ndarray
            Test set of time points, with shape (n_te_t,).
        X_te : numpy.ndarray
            Test set of spatial points, with shape (n_te_x, n), where `n` is the spatial dimension.
        thresholding : bool, optional
            If True, applies non-negativity thresholding on the predicted diffusion coefficients.
            Default is False.

        Returns
        -------
        B_pc : numpy.ndarray
            Predicted drift coefficients with positivity constraints applied, shape (n_te,).
        S_pc : numpy.ndarray
            Predicted diffusion coefficients with positivity constraints applied, shape (n_te,).
            Non-negativity is enforced if `thresholding` is True.
        B : numpy.ndarray
            Predicted drift coefficients without positivity constraints, shape (n_te,).
        S : numpy.ndarray
            Predicted diffusion coefficients without positivity constraints, shape (n_te,).
            Non-negativity is enforced if `thresholding` is True.
        """

        Z_te = cartesian_products_of_rows(T_te, X_te)
        if self.idx_ny is None:
            # Compute train/test Gram matrices.
            K_tr_te, d_K_tr_te, d_2_K_tr_te = self.compute_grams_K(
                self.Z_tr, Z_te, full=False
            )
            R_b_tr_te, R_s_tr_te = self.compute_R(
                K_tr_te, d_K_tr_te, d_2_K_tr_te, self.P_tr, self.d_P_tr, self.d_2_P_tr
            )
            # Compute predictions of KRR (without positivity constraints).
            B = self.alpha.dot(R_b_tr_te)
            S = self.alpha.dot(R_s_tr_te)

            # Compute predictions for KRR with positivity constraints.
            add_B = -self.alpha_pc.dot(R_b_tr_te)

            add_S = self.eps.T.dot(K_tr_te[self.idx_pc, :]) - self.alpha_pc.dot(
                R_s_tr_te
            )
            B_pc = B + add_B
            S_pc = S + add_S

        else:
            # Compute train/test Gram matrices.
            K_ny_te, d_K_ny_te, d_2_K_ny_te = self.compute_grams_K(
                self.Z_tr[self.idx_ny], Z_te, full=False
            )
            R_b_ny_te, R_s_ny_te = self.compute_R(
                K_ny_te,
                d_K_ny_te,
                d_2_K_ny_te,
                self.P_tr[self.idx_ny],
                self.d_P_tr[:, self.idx_ny],
                self.d_2_P_tr[:, :, self.idx_ny],
            )
            # Compute predictions of standard KRR (without positivity constraints).
            B_pc = self.alpha.dot(R_b_ny_te)
            S_pc = self.alpha.dot(R_s_ny_te)
            B, S = B_pc, S_pc

        # Ensure S_pc are positives.
        if thresholding:
            S_pc[S_pc < 0] = 0
            S[S < 0] = 0
        # Return standard deviations.
        S[S > 0] = S[S > 0] ** (1 / 2)
        S_pc[S_pc > 0] = S_pc[S_pc > 0] ** (1 / 2)

        return B_pc, S_pc, B, S

    def predict_kolmogorov(self, Z_te, P_te, d_P_te, d_2_P_te):
        """
        Predicts values of the time derivative of the probability density, on the given test set, as the probability
        density values through the Kolmogorov operator associated to the estimated coefficients.

        Parameters
        ----------
        Z_te : numpy.ndarray
            Test set points, specified as either a 2D array with shape (n_te, n + 1).
        P_te : numpy.ndarray
            Probability density at test points, with shape (n_te,).
        d_P_te : numpy.ndarray
            First derivative of the probability density at test points, with shape (n_te,).
        d_2_P_te : numpy.ndarray
            Second derivative of the probability density at test points, with shape (n_te,).

        Returns
        -------
        numpy.ndarray
            Predicted time derivatives of the probability density at the test points, with shape (n_te, 1).
        """

        # Compute train/test Gram matrices.
        (
            K_tr_te,
            d_K_tr_te,
            d_2_K_tr_te,
            d_3_K_tr_te,
            d_4_K_tr_te,
        ) = self.compute_grams_K(self.Z_tr, Z_te, full=True)

        K_tilde_te = self.compute_K_tilde(
            K_tr_te,
            d_K_tr_te,
            d_2_K_tr_te,
            d_3_K_tr_te,
            d_4_K_tr_te,
            self.P_tr,
            self.d_P_tr,
            self.d_2_P_tr,
            P_te,
            d_P_te,
            d_2_P_te,
        )

        # Compute predictions of standard KRR (without positivity constraints).
        d_t_p_kolmogorov_te = K_tilde_te.T.dot(self.alpha.T)

        # Compute predictions for KRR with positivity constraints.
        (
            K_te_pc,
            d_K_te_pc,
            d_2_K_te_pc,
        ) = self.compute_grams_K(Z_te, self.Z_tr[self.idx_pc], full=False)

        _, R_s_te_pc = self.compute_R(
            K_te_pc,
            d_K_te_pc,
            d_2_K_te_pc,
            P_te,
            d_P_te,
            d_2_P_te,
        )

        d_t_p_kolmogorov_add = self.eps.T.dot(R_s_te_pc.T) - self.alpha_pc.dot(
            K_tilde_te
        )
        d_t_p_kolmogorov_te += d_t_p_kolmogorov_add.reshape((-1, 1))

        return d_t_p_kolmogorov_te

    def predict_kolmogorov_nystrom(self, Z_te, P_te, d_P_te, d_2_P_te):
        """
        Predicts values of the time derivative of the probability density, on the given test set, as the probability
        density values through the Kolmogorov operator associated to the estimated coefficients, when using Nyström
        approximation.

        Parameters
        ----------
        Z_te : numpy.ndarray
            Test set points, with shape (n_te, n + 1).
        P_te : numpy.ndarray
            Probability density values at test points, with shape (n_te,).
        d_P_te : numpy.ndarray
            First derivatives of the probability density at test points, with shape (n_te,).
        d_2_P_te : numpy.ndarray
            Second derivatives of the probability density at test points, with shape (n_te,).

        Returns
        -------
        numpy.ndarray
            Predicted time derivatives of the probability density function at test points, with shape (n_te, 1).
        """

        # Compute ny/test Gram matrices.
        (
            K_te_ny,
            d_K_te_ny,
            d_2_K_te_ny,
            d_3_K_te_ny,
            d_4_K_te_ny,
        ) = self.compute_grams_K(Z_te, self.Z_tr[self.idx_ny], full=True)

        K_tilde_te_ny = self.compute_K_tilde(
            K_te_ny,
            d_K_te_ny,
            d_2_K_te_ny,
            d_3_K_te_ny,
            d_4_K_te_ny,
            P_te,
            d_P_te,
            d_2_P_te,
            self.P_tr[self.idx_ny],
            self.d_P_tr[:, self.idx_ny],
            self.d_2_P_tr[:, :, self.idx_ny],
        )
        # Compute predictions of KRR with Nyström approximation and positivity constraints.
        d_t_p_kolmogorov_te = K_tilde_te_ny.dot(self.alpha.T)
        d_t_p_kolmogorov_te = d_t_p_kolmogorov_te.reshape((-1, 1))

        return d_t_p_kolmogorov_te

    def compute_mse(self, T_te=None, X_te=None):
        """
        Compute the mean squared error (MSE) between the predicted and true time derivatives of the probability
        density.

        Parameters
        ----------
        T_te : numpy.ndarray, optional
            Test set time points. If not provided, uses the training set time points.
        X_te : numpy.ndarray, optional
            Test set spatial points. If not provided, uses the training set spatial points.

        Returns
        -------
        d_t_p_kolmogorov_te : numpy.ndarray
            Predicted time derivative of the probability density function.
        d_t_P_te : numpy.ndarray
            True time derivative of the probability density function.
        MSE_te : float
            Mean squared error between the predicted and actual time derivatives.
        norms_te : tuple
            Tuple containing the mean squared norms of the predicted and true time derivatives, respectively.
        """

        if T_te is None:
            Z_te = self.Z_tr
            P_te, d_P_te, d_2_P_te, d_t_p_te = (
                self.P_tr,
                self.d_P_tr,
                self.d_2_P_tr,
                self.d_t_P_tr,
            )
            P_te = P_te.reshape((-1, 1))
            d_P_te = d_P_te.reshape((self.n, -1, 1))
            d_2_P_te = d_2_P_te.reshape((self.n, self.n, -1, 1))
            n_t = self.n_t
        else:
            Z_te = cartesian_products_of_rows(T_te, X_te)
            P_te, d_P_te, d_2_P_te, d_t_p_te = self.kde(T_te, X_te, partial=True)
            P_te = P_te.reshape((-1, 1))
            d_P_te = d_P_te.reshape((self.n, -1, 1))
            d_2_P_te = d_2_P_te.reshape((self.n, self.n, -1, 1))
            n_t = T_te.shape[0]

        if self.idx_ny is None:
            d_t_p_kolmogorov_te = self.predict_kolmogorov(Z_te, P_te, d_P_te, d_2_P_te)
        else:
            d_t_p_kolmogorov_te = self.predict_kolmogorov_nystrom(
                Z_te, P_te, d_P_te, d_2_P_te
            )

        # Compute test MSE.
        n_te = Z_te.shape[0]
        L = d_t_p_te.reshape((-1, 1)) - d_t_p_kolmogorov_te
        MSE_te = 1 / n_te * np.linalg.norm(L) ** 2

        # Compute mean squared norm of predictions and true targets.
        d_t_p_kolmogorov_te = d_t_p_kolmogorov_te.reshape((n_t, -1))
        norms_te = (
            1 / n_te * np.linalg.norm(d_t_p_kolmogorov_te) ** 2,
            1 / n_te * np.linalg.norm(d_t_p_te.reshape((-1, 1))) ** 2,
        )

        return d_t_p_kolmogorov_te, d_t_p_te, MSE_te, norms_te

    def compute_grams_K(self, Z_1, Z_2=None, full=False):
        """
        Computes the Gram matrix and its partial derivatives up to the fourth order for two sets of points, Z_1 and Z_2,
        using a Radial Basis Function (RBF) kernel with optional constant term.

        Parameters
        ----------
        Z_1 : numpy.ndarray
            The first set of points, shape (n_samples_1, n_features).
        Z_2 : numpy.ndarray, optional
            The second set of points, shape (n_samples_2, n_features). If not provided, it defaults to Z_1.
        full : bool, optional
            If True, computes and returns derivatives up to the fourth order. If False, only the Gram matrix and
            its first and second derivatives are computed. Default is False.

        Returns
        -------
        K_1_2 : numpy.ndarray
            The Gram matrix computed using the RBF kernel, shape (n_samples_1, n_samples_2).
        d_K_1_2 : numpy.ndarray
            The first derivative of the Gram matrix.
        d_2_K_1_2 : numpy.ndarray
            The second derivative of the Gram matrix.
        d_3_K_1_2 : numpy.ndarray, optional
            The third derivative of the Gram matrix, returned only if `full` is True.
        d_4_K_1_2 : numpy.ndarray, optional
            The fourth derivative of the Gram matrix, returned only if `full` is True.
        """

        Z_1 = Z_1.copy()
        if Z_2 is None:
            Z_2 = Z_1.copy()
        else:
            Z_2 = Z_2.copy()

        K_1_2 = rbf_kernel(Z_1, Z_2, self.gamma_z) + self.c_kernel

        # Calculate difference between each pair of elements.
        X_1 = Z_1[:, 1:]
        X_2 = Z_2[:, 1:]
        D_x = -(X_2[np.newaxis, :, :] - X_1[:, np.newaxis, :])

        # Apply recursive formulas to compute partial derivatives Gram matrices.
        n_1 = Z_1.shape[0]
        n_2 = Z_2.shape[0]
        d_K_1_2 = np.zeros((self.n, n_1, n_2))
        d_2_K_1_2 = np.zeros((self.n, self.n, n_1, n_2))
        for i in range(self.n):
            d_K_1_2[i] = -2 * self.gamma_z * D_x[:, :, i] * K_1_2
            for j in range(self.n):
                d_2_K_1_2[i, j] = (
                    -2 * self.gamma_z * (D_x[:, :, j] * d_K_1_2[i] + (i == j) * K_1_2)
                )

        if not full:
            return K_1_2, d_K_1_2, d_2_K_1_2
        else:
            d_3_K_1_2 = np.zeros((self.n, self.n, self.n, n_1, n_2))
            d_4_K_1_2 = np.zeros((self.n, self.n, self.n, self.n, n_1, n_2))

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        K_k_ij = (
                            -2
                            * self.gamma_z
                            * (
                                D_x[:, :, j] * d_2_K_1_2[i, k]
                                + (k == j) * d_K_1_2[i]
                                + (i == j) * d_K_1_2[k]
                            )
                        )
                        K_ij_k = -K_k_ij
                        d_3_K_1_2[i, j, k] = K_ij_k

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        for q in range(self.n):
                            K_kl_ij = (
                                -2
                                * self.gamma_z
                                * (
                                    -D_x[:, :, j] * d_3_K_1_2[k, q, i]
                                    + (q == j) * d_2_K_1_2[i, k]
                                    + (k == j) * d_2_K_1_2[i, q]
                                    + (i == j) * d_2_K_1_2[k, q]
                                )
                            )
                            d_4_K_1_2[k, q, i, j] = K_kl_ij

            return K_1_2, d_K_1_2, d_2_K_1_2, d_3_K_1_2, d_4_K_1_2

    def compute_K_tilde(
        self,
        K,
        d_K,
        d_2_K,
        d_3_K,
        d_4_K,
        P_1,
        d_P_1,
        d_2_P_1,
        P_2=None,
        d_P_2=None,
        d_2_P_2=None,
    ):
        """
        Compute the Gram matrix tilde K associated to the kernel tilde k(z_1, z_2) := <tilde phi (z_1), tilde phi(z_2)>
        (See, Lemma 4.3).

        Parameters
        ----------
        K : numpy.ndarray
            Gram matrix between two sets of points Z_1 and Z_2.
        d_K, d_2_K, d_3_K, d_4_K : numpy.ndarray
            The first, second, third, and fourth derivatives of the Gram matrix.
        P_1, d_P_1, d_2_P_1 : numpy.ndarray
            The probability density and its first and second derivatives evaluated at Z_1.
        P_2, d_P_2, d_2_P_2 : numpy.ndarray, optional
            The probability density and its first and second derivatives evaluated at Z_2. If not provided, P_1,
            d_P_1, and d_2_P_1 are used for both sets of points.

        Returns
        -------
        numpy.ndarray
            Gram matrix for the tilde kernel between the two sets Z_1 and Z_2.
        """

        if P_2 is None:
            P_2 = P_1
            d_P_2 = d_P_1
            d_2_P_2 = d_2_P_1
        else:
            P_2 = P_2
            d_P_2 = d_P_2
            d_2_P_2 = d_2_P_2

        tilde_K = np.zeros(K.shape)

        # Add first order partial derivative terms
        for i in range(self.n):
            P_i = d_P_1[i]
            d_i_K, d_ii_K, d_i_P_2 = d_K[i], d_2_K[i, i], d_P_2[i]
            tilde_K_i = (
                K * np.outer(P_i, d_i_P_2)
                + d_i_K * np.outer(P_1, d_i_P_2)
                + (-d_i_K) * np.outer(P_i, P_2)
                + (-d_ii_K) * np.outer(P_1, P_2)
            )
            tilde_K += tilde_K_i

        # # Add second order partial derivative terms
        for i in range(self.n):
            j = i
            for k in range(self.n):
                q = k
                d_i_K, d_j_K = d_K[i], d_K[j]
                d_k_K, d_l_K = d_K[k], d_K[q]
                d_ij_K, d_il_K = d_2_K[i, j], d_2_K[i, q]
                d_kl_K, d_ki_K = d_2_K[k, q], d_2_K[k, i]
                d_kj_K, d_jl_K = d_2_K[k, j], d_2_K[j, q]
                d_ki_j_K, d_kl_j_K = d_3_K[k, i, j], d_3_K[k, q, j]
                d_kl_i_K, d_li_j_K = d_3_K[k, q, i], d_3_K[q, i, j]
                d_kl_ij_K = d_4_K[k, q, i, j]
                d_k_P_1, d_l_P_1, d_kl_P_1 = d_P_1[k], d_P_1[q], d_2_P_1[k, q]
                d_i_P_2, d_j_P_2, d_ij_P_2 = d_P_2[i], d_P_2[j], d_2_P_2[i, j]

                tilde_K_kl_ij = (
                    K * np.outer(d_kl_P_1, d_ij_P_2)
                    + d_k_K * np.outer(d_l_P_1, d_ij_P_2)
                    + d_l_K * np.outer(d_k_P_1, d_ij_P_2)
                    + d_kl_K * np.outer(P_1, d_ij_P_2)
                    + (-d_i_K) * np.outer(d_kl_P_1, d_j_P_2)
                    + (-d_ki_K) * np.outer(d_l_P_1, d_j_P_2)
                    + (-d_il_K) * np.outer(d_k_P_1, d_j_P_2)
                    + d_kl_i_K * np.outer(P_1, d_j_P_2)
                    + (-d_j_K) * np.outer(d_kl_P_1, d_i_P_2)
                    + (-d_kj_K) * np.outer(d_l_P_1, d_i_P_2)
                    + (-d_jl_K) * np.outer(d_k_P_1, d_i_P_2)
                    + d_kl_j_K * np.outer(P_1, d_i_P_2)
                    + d_ij_K * np.outer(d_kl_P_1, P_2)
                    + (-d_ki_j_K) * np.outer(d_l_P_1, P_2)
                    + (-d_li_j_K) * np.outer(d_k_P_1, P_2)
                    + d_kl_ij_K * np.outer(P_1, P_2)
                )

                tilde_K += (1 / 4) * tilde_K_kl_ij

        return tilde_K

    def compute_R(self, K, d_K, d_2_K, P_1, d_P_1, d_2_P_1):
        """
        Compute the Gram matrix R associated to the kernels r^b_i(z_1, z_2) := <tilde phi_i(z_1), phi(z_2)>, and
        r^s_ij(z_1, z_2) := <tilde phi_ij(z_1), phi(z_2)>.

        Parameters
        ----------
        K : numpy.ndarray
            Gram matrix between the two sets Z_1, Z_2.
        d_K, d_2_K : numpy.ndarray
            The first and second derivatives of the Gram matrix with respect to the features.
        P_1 : numpy.ndarray
            The probability density evaluated at Z_1.
        d_P_1 : numpy.ndarray
            The first derivative of the probability density evaluated at Z_1.
        d_2_P_1 : numpy.ndarray
            The second derivative of the probability density evaluated at Z_1.

        Returns
        -------
        R_b : numpy.ndarray
            Gram matrix R^b between the two sets Z_1, Z_2, for the drift component of the model.
        R_s : numpy.ndarray
            Gram matrix R^s between the two sets Z_1, Z_2, for the diffusion component of the model.
        """

        n_1, n_2 = K.shape
        R_b = np.zeros((self.n, n_1, n_2))
        R_s = np.zeros((n_1, n_2))

        for i in range(self.n):
            d_i_P_1 = d_P_1[i]
            R_te_b_i = K * np.outer(d_i_P_1, np.ones(n_2).T) + d_K[i] * np.outer(
                P_1, np.ones(n_2).T
            )
            R_b[i] = -R_te_b_i
            j = i
            d_j_P_1, d_ij_P_1 = d_P_1[j], d_2_P_1[i, j]
            R_te_s_ij = K * np.outer(d_ij_P_1, np.ones(n_2).T)
            R_te_s_ij += d_K[i] * np.outer(d_j_P_1, np.ones(n_2).T)
            R_te_s_ij += d_K[j] * np.outer(d_i_P_1, np.ones(n_2).T)
            R_te_s_ij += d_2_K[i, j] * np.outer(P_1, np.ones(n_2).T)
            R_s += 1 / 2 * R_te_s_ij

        return R_b, R_s
