import numpy as np
from utils.utils import (
    rho,
    rho_partial_derivative,
    rho_second_order_partial_derivative,
    mse,
    cartesian_products_of_rows,
    metropolis_hastings,
)


def test_rho():
    """
    Test the rho function which computes a kernel or distance metric.

    This test verifies that the rho function returns correct output shapes when computing distances or kernel values
    between sets of points X and Y using a specified parameter mu.

    Assertions:
        - Ensures that the output of the rho function has the expected shape based on the input dimensions.
    """

    # Setup some test data
    X = np.array([[0, 0], [0, 0]])
    Y = np.array([[0, 0], [0, 0]])
    mu = 1.0

    # Call the function
    result = rho(X, Y, mu)

    # Check shape
    assert result.shape == (2, 2), "Output shape is incorrect."


def test_rho_partial_derivative():
    """
    Test the first-order partial derivative of the rho function.

    This function evaluates the partial derivative of the rho kernel with respect to one of its parameters.
    It checks if the rho_partial_derivative function returns the correct output shape when provided with
    input arrays X and Y, a scale parameter mu, and the index k of the parameter with respect to which the derivative is
    taken.

    Assertions:
        - Ensures that the output shape of the partial derivative matches the expected dimensionality.
    """

    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[1, 2], [5, 6]])
    mu = 1.0
    k = 1

    result = rho_partial_derivative(X, Y, mu, k)

    assert result.shape == (2, 2), "Output shape is incorrect."


def test_rho_second_order_partial_derivative():
    """
    Test the second-order partial derivative of the rho function.

    This test checks the computation of the second-order partial derivative of the rho kernel with respect to
    its parameters. It verifies that the function outputs a result with the correct shape when provided with
    inputs X, Y, mu, and the indices k and q for the parameters involved in the derivative.

    Assertions:
        - Asserts that the resulting matrix from the computation matches the expected shape.
    """

    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[1, 2], [5, 6]])
    mu = 1.0
    k = 0
    q = 1

    result = rho_second_order_partial_derivative(X, Y, mu, k, q)

    assert result.shape == (2, 2), "Output shape is incorrect."


def test_mse():
    """
    Test the mean squared error (MSE) function for statistical comparison.

    This test evaluates the mse function, which calculates the mean squared error between two probabilistic models
    over a given domain. It uses mock model functions p1 and p2, and checks if the result is a floating-point number,
    confirming that mse operates correctly under the given parameters.

    Assertions:
        - Verifies that the output of the mse function is a float, indicating proper calculation.
    """

    def p1(t, x):
        return t + np.sum(x, axis=0)

    def p2(t, x):
        return t

    T = 1.0
    x_dim = 2
    R = 1.0
    n_t = 10
    n_x = 10

    result = mse(p1, p2, T, x_dim, R, n_t, n_x)
    assert isinstance(result, float), "Output should be a float."


def test_cartesian_products_of_rows():
    """
    Test the Cartesian product function between rows of two matrices.

    This test ensures that the cartesian_products_of_rows function correctly computes the Cartesian product of each
    row of matrix A with each row of matrix B. It checks if the resulting matrix has the expected shape, based on the
    input dimensions and Cartesian product rules.

    Assertions:
        - Confirms that the output matrix has the correct shape resulting from the Cartesian product of the input
         matrices.
    """

    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 2], [3, 4]])

    result = cartesian_products_of_rows(A, B)

    assert result.shape == (4, 5), "Output shape is incorrect."


def test_metropolis_hastings():
    """
    Test the Metropolis-Hastings algorithm for sampling.

    This function tests the metropolis_hastings function to ensure it correctly samples from a given probability
    distribution p using the Metropolis-Hastings algorithm. It checks if the sampled data has the expected dimensions,
    given the number of samples and the dimensionality of the sample space.

    Assertions:
        - Verifies that the output from the Metropolis-Hastings sampling has the correct shape.
    """

    def p(x):
        return np.exp(-np.sum(x**2))

    n_sample = 10
    n_iter = 100
    n = 2

    result = metropolis_hastings(p, n_sample, n_iter, n)

    assert result.shape == (n_sample, n), "Output shape is incorrect."
