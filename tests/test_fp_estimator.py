import pytest
import numpy as np
from methods.fp_estimator import FPEstimator


def test_fpestimator_initialization():
    """
    Test the initialization of FPEstimator class.

    This test verifies that all parameters of an FPEstimator instance are correctly initialized to None.
    It ensures that the estimator starts with no pre-defined or residual states from prior computations.

    Assertions:
        - Checks that all parameter attributes and training-related attributes of the estimator are None upon
         initialization.
    """

    # Create an instance of FPEstimator
    estimator = FPEstimator()

    # Check that all parameters are initialized to None
    assert estimator.la is None, "Expected la to be initialized to None"
    assert estimator.be is None, "Expected be to be initialized to None"
    assert estimator.gamma_z is None, "Expected gamma_z to be initialized to None"
    assert estimator.kde is None, "Expected kde to be initialized to None"
    assert estimator.c_kernel is None, "Expected c_kernel to be initialized to None"
    assert estimator.mu is None, "Expected mu to be initialized to None"

    # Check that all attributes set during training are None
    assert estimator.T is None, "Expected T to be None"
    assert estimator.n is None, "Expected n to be None"
    assert estimator.n_t is None, "Expected n_t to be None"
    assert estimator.n_x is None, "Expected n_x to be None"
    assert estimator.n_z is None, "Expected n_z to be None"
    assert estimator.n_pc is None, "Expected n_pc to be None"
    assert estimator.Z_tr is None, "Expected Z_tr to be None"
    assert estimator.P_tr is None, "Expected P_tr to be None"
    assert estimator.d_P_tr is None, "Expected d_P_tr to be None"
    assert estimator.d_2_P_tr is None, "Expected d_2_P_tr to be None"
    assert estimator.d_t_P_tr is None, "Expected d_t_P_tr to be None"
    assert estimator.idx_pc is None, "Expected idx_pc to be None"
    assert estimator.eps is None, "Expected eps to be None"
    assert estimator.alpha is None, "Expected alpha to be None"
    assert estimator.alpha_pc is None, "Expected alpha_pc to be None"
    assert estimator.idx_ny is None, "Expected idx_ny to be None"


def test_fit():
    """
    Test the fitting process of the FPEstimator.

    This test verifies the functionality of the `fit` method of the FPEstimator class. It checks the handling
    and setup of data, ensuring that model coefficients are calculated and stored as expected when fitting
    the model using mock data.

    Assertions:
        - Ensures that either 'alpha' or 'alpha_pc' coefficients are calculated and stored, indicating successful model
         fitting.
    """

    estimator = FPEstimator()
    estimator.la = 100  # Regularization parameter
    estimator.gamma_z = 1  # Kernel parameter
    estimator.be = 1  # Scaling/bandwidth parameter
    estimator.T = 1  # Scaling/bandwidth parameter
    estimator.c_kernel = 1

    # Mock data
    T_tr = np.random.uniform(0, 10, size=(50, 1))
    X_tr = np.random.uniform(-1, 1, size=(50, 1))

    # Mock probability density function
    def mock_pdf(Ts, X, partial):
        Ts_grid, X_grid = np.meshgrid(Ts, X, indexing="ij")
        P = np.exp(-(X_grid**2) / (1 + Ts_grid))
        P = P.reshape((-1, 1))
        if partial:
            return P, P[np.newaxis, :, :], P[np.newaxis, np.newaxis, :, :], P
        else:
            return P

    estimator.fit(T_tr=T_tr, X_tr=X_tr, p=mock_pdf)
    assert (
        estimator.alpha is not None or estimator.alpha_pc is not None
    ), "Model coefficients not calculated"


@pytest.fixture
def fitted_estimator():
    """
    Fixture for setting up a mock fitted FPEstimator.

    This fixture creates and returns a mock FPEstimator instance assumed to be already fitted.
    It initializes various parameters and sets mock regression coefficients and other attributes typically determined
     during fitting.
    This setup is used to facilitate testing of methods that require a pre-fitted model.

    Returns:
        FPEstimator: A mock fitted instance of FPEstimator.
    """

    # Create a mock FPEstimator assumed to be fitted
    est = FPEstimator()
    est.n = 1  # 1-dimensional SDE
    est.mu = 0.1
    est.idx_ny = None  # Assume no Nyström approximation used
    est.c_kernel = 1
    est.la = 100  # Regularization parameter
    est.gamma_z = 1  # Kernel parameter
    est.be = 1  # Scaling/bandwidth parameter
    est.T = 1  # Scaling/bandwidth parameter
    est.c_kernel = 1

    # Mock attributes that would be set during fitting
    est.alpha = np.random.rand(1, 100)  # Mock regression coefficients
    est.alpha_pc = np.random.rand(1, 100)  # Mock regression coefficients
    est.eps = np.random.rand(100, 1)
    est.Z_tr = np.random.rand(100, 2)  # Mock training data points
    est.n_z = 10  # Number of training points

    # Mock probability density function
    def mock_pdf(Ts, X, partial):
        Ts_grid, X_grid = np.meshgrid(Ts, X, indexing="ij")
        P = np.exp(-(X_grid**2) / (1 + Ts_grid))
        P = P.reshape((-1, 1))
        if partial:
            return P, P[np.newaxis, :, :], P[np.newaxis, np.newaxis, :, :], P
        else:
            return P

    T_tr = np.random.uniform(0, 10, size=(10, 1))
    X_tr = np.random.uniform(-1, 1, size=(10, 1))
    est.P_tr, est.d_P_tr, est.d_2_P_tr, est.d_t = mock_pdf(T_tr, X_tr, partial=True)

    return est


def test_predict(fitted_estimator):
    """
    Test the prediction functionality of the FPEstimator class.

    Using a mock fitted FPEstimator, this test verifies the `predict` method's ability to correctly compute
    drift and diffusion coefficients from provided input data. It ensures that the outputs have correct shapes
    and types, and checks the application of thresholding to ensure no negative values in the diffusion coefficients.

    Assertions:
        - Checks the types and shapes of the output arrays to confirm they match expected dimensions.
        - Verifies that thresholding is correctly applied to the diffusion coefficients.
    """

    # Test inputs
    T_te = np.random.rand(10, 1)
    X_te = np.random.rand(10, 1)

    # Execute predict method
    B_pc, S_pc, B, S = fitted_estimator.predict(T_te, X_te, thresholding=True)

    # Verify output shapes and types
    assert isinstance(B_pc, np.ndarray), "Expected B_pc to be a numpy array"
    assert isinstance(S_pc, np.ndarray), "Expected S_pc to be a numpy array"
    assert isinstance(B, np.ndarray), "Expected B to be a numpy array"
    assert isinstance(S, np.ndarray), "Expected S to be a numpy array"

    # Assuming each prediction corresponds to each input
    assert B_pc.shape == (1, 1, 100), "Incorrect shape for B_pc"
    assert S_pc.shape == (1, 1, 100), "Incorrect shape for S_pc"
    assert B.shape == (1, 1, 100), "Incorrect shape for B"
    assert S.shape == (1, 100), "Incorrect shape for S"

    # Check thresholding is applied
    assert np.all(S_pc >= 0), "Thresholding failed, S_pc contains negative values"
