import pytest
import numpy as np
from methods.kde import ProbaDensityEstimator
from utils.data import (
    euler_maruyama,
    ornstein_uhlenbeck,
)


def test_initialization():
    """
    Test the initialization of the ProbaDensityEstimator class.

    This test ensures that the probability density estimator is correctly initialized with the specified
     hyperparameters.
    It checks if each parameter is correctly assigned based on the intended initial values.

    Assertions:
        - Verifies that each hyperparameter (gamma_t, mu_x, L_t, c_kernel, T) is initialized to the expected value.
    """

    # Initialize the probability density estimator.
    estimator = ProbaDensityEstimator()

    # Choose the hyperparameters.
    T = 10  # Ending time
    gamma_t = 1
    L_t = 1e-6
    mu_x = 7
    c_kernel = 1e-3
    estimator.gamma_t = gamma_t
    estimator.mu_x = mu_x
    estimator.L_t = L_t
    estimator.c_kernel = c_kernel
    estimator.T = T

    # Assert to check if hyperparameters are initialized correctly
    assert estimator.gamma_t == gamma_t, "Gamma_t initialization error"
    assert estimator.mu_x == mu_x, "Mu_x initialization error"
    assert estimator.L_t == L_t, "L_t initialization error"
    assert estimator.c_kernel == c_kernel, "C_kernel initialization error"
    assert estimator.T == T, "T initialization error"


def test_fit():
    """
    Test the fit method of the ProbaDensityEstimator class.

    This function tests the fitting process of the ProbaDensityEstimator by providing a small set of training data.
    It checks if the method properly initializes the model parameters necessary for later computations, such as
    the kernel matrix and dimensions of training data.

    Assertions:
        - Ensures that the number of training samples, dimensionality of data, and kernel matrix are correctly
        initialized.
    """

    estimator = ProbaDensityEstimator()
    T_tr = np.array([[0], [1], [2]])
    X_tr = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # Set necessary parameters for kernel calculation
    estimator.gamma_t = 0.1
    estimator.c_kernel = 1e-3
    estimator.L_t = 1e-6

    estimator.fit(T_tr, X_tr)
    assert estimator.n_tr == 3, "Incorrect number of training samples"
    assert estimator.x_dim == 2, "Incorrect dimensionality of training data"
    assert estimator.T_tr is not None, "Training times not set"
    assert estimator.X_tr is not None, "Training data not set"
    assert estimator.K_t_inv is not None, "Inverse kernel matrix not initialized"


def test_predict_without_fit():
    """
    Test the predict method of the ProbaDensityEstimator without prior fitting.

    This test checks the behavior of the predict method when called without fitting the estimator first.
    It ensures that the appropriate error is raised to prevent predictions from unfitted models.

    Assertions:
        - Asserts that a ValueError is raised, indicating the model must be fitted before making predictions.
    """

    estimator = ProbaDensityEstimator()
    T_te = np.array([[1]])
    X_te = np.array([[0.5, 0.6]])

    with pytest.raises(ValueError):
        estimator.predict(T_te, X_te)


def test_predict_with_fit():
    """
    Test the predict method of the ProbaDensityEstimator after fitting.

    This test evaluates the functionality of the predict method following the model's fitting with training data.
    It verifies that the estimator can make predictions on new data and that these predictions are returned properly.

    Assertions:
        - Ensures that predictions are successfully made on test data and results are returned.
    """

    estimator = ProbaDensityEstimator()
    # Simulation parameters.
    n = 2  # Dimension of the state variable
    mu = np.array([1.0] * n)  # Mean
    theta = 0.6  # Rate of mean reversion
    sigma = 0.5  # Volatility
    eps = 0.1  # Starting time
    T = 10.0  # Ending time
    mu_0, sigma_0 = np.array([0.5] * n), sigma / (2 * theta) ** (
        1 / 2
    )  # Mean and std at t=0
    n_steps = 100  # Number of time steps
    dt = T / n_steps  # Time step
    T_tr = (np.arange(1, n_steps + 1) * dt + eps).reshape(
        -1, 1
    )  # Temporal discretization
    n_paths = 30  # Number of paths to draw
    b, sigma_func = ornstein_uhlenbeck(mu=mu, theta=theta, sigma=sigma)
    X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0)

    # Choose the hyperparameters.
    gamma_t = 1
    L_t = 1e-6
    mu_x = 7
    c_kernel = 1e-3
    estimator.gamma_t = gamma_t
    estimator.mu_x = mu_x
    estimator.L_t = L_t
    estimator.c_kernel = c_kernel
    estimator.T = T

    estimator.fit(T_tr, X_tr)

    # Generate n_x points x in [-R, R]^n.
    Q = 20
    X_te = np.random.uniform(-0.2, 0.9, size=(Q, n)).reshape((-1, 1, n))

    result = estimator.predict(T_tr, X_te)
    assert result is not None, "Prediction failed to return any output"


def test_log_likelihood():
    """
    Test the log likelihood computation in ProbaDensityEstimator.

    This test checks the log likelihood function of the estimator. It uses a sample probability array to calculate
    the log likelihood and verifies that the function returns a negative value, consistent with the properties of log
     likelihoods.

    Assertions:
        - Confirms that the log likelihood of a given probability distribution is negative.
    """

    estimator = ProbaDensityEstimator()
    P = np.array([0.1, 0.01, 0.001])
    ll = estimator.log_likelihood(P)
    assert ll < 0, "Log likelihood should be negative"
