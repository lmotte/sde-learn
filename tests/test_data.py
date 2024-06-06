import numpy as np
from utils.data import (
    euler_maruyama,
    ornstein_uhlenbeck,
    dubins,
    gaussian,
    van_der_pol,
    plot_paths_1d,
    plot_paths_2d,
    plot_coefficients,
    ornstein_uhlenbeck_pdf,
    plot_map_1d,
    plot_map_2d,
    line_plot,
)


def test_euler_maruyama():
    """
    Test the Euler-Maruyama simulation function for stochastic differential equations.

    This test verifies that the `euler_maruyama` function correctly simulates paths
    with expected dimensions. It uses a simple linear drift and constant diffusion
    model for testing.

    Assertions:
        - Ensures the output shape of simulated paths matches the expected dimensions.
    """

    def b(t, x):
        return np.array([-0.5 * x])

    def sigma(t, x):
        return np.array([1.0])

    paths = euler_maruyama(b, sigma, 100, 10, 1.0, 1, np.array([0]), 0.1)
    assert paths.shape == (10, 100, 1), "Output shape is incorrect."


def test_ornstein_uhlenbeck():
    """
    Test the Ornstein-Uhlenbeck process coefficient functions.

    This test ensures that the drift and diffusion functions returned by the
    `ornstein_uhlenbeck` function are callable.

    Assertions:
        - Confirms that the drift and diffusion functions are callable.
    """

    b, sigma = ornstein_uhlenbeck(np.array([0]), 1.0, 0.1)
    assert callable(b), "Drift function should be callable."
    assert callable(sigma), "Diffusion function should be callable."


def test_dubins():
    """
    Test the Dubins car model coefficient functions.

    This test verifies that the `dubins` function returns callable drift and diffusion
    functions based on provided parameters.

    Assertions:
        - Confirms that the drift and diffusion functions are callable.
    """

    b, sigma = dubins(1.0, 1.0, 0.1)
    assert callable(b), "Drift function should be callable."
    assert callable(sigma), "Diffusion function should be callable."


def test_gaussian():
    """
    Test Gaussian coefficient functions for drift and diffusion.

    This test checks that the `gaussian` function returns callable drift and diffusion
    functions based on specified Gaussian means and multiplicative constants.

    Assertions:
        - Ensures both returned functions are callable.
    """

    b, sigma = gaussian(
        0, np.array([1, 1]), 0.5, np.array([1, 1]), 1.0, np.array([1, 1]), 1.0
    )
    assert callable(b), "Drift function should be callable."
    assert callable(sigma), "Diffusion function should be callable."


def test_van_der_pol():
    """
    Test the Van der Pol oscillator model's coefficient functions.

    This test ensures that the `van_der_pol` function returns callable drift and
    diffusion functions suitable for simulating the Van der Pol oscillator.

    Assertions:
        - Confirms that the drift and diffusion functions are callable.
    """

    b, sigma = van_der_pol(2.0, 0.1)
    assert callable(b), "Drift function should be callable."
    assert callable(sigma), "Diffusion function should be callable."


def test_plot_paths_1d(tmp_path):
    """
    Test the 1-dimensional path plotting functionality.

    This function tests if `plot_paths_1d` generates a plot file at the specified
    location. It uses random paths for the test plot.

    Assertions:
        - Checks if the plot file is created in the specified directory.
    """

    # This tests the plotting function by checking if a file is created.
    T = np.linspace(0, 1, 10)
    paths = np.random.rand(5, 10, 1)
    save_path = tmp_path / "test_plot.pdf"
    plot_paths_1d(T, paths, str(save_path))
    assert save_path.exists(), "Plot file should exist."


def test_plot_paths_2d(tmp_path):
    """
    Test the 2-dimensional path plotting functionality.

    This function tests if `plot_paths_2d` successfully creates a plot file in the
    designated path. It uses random paths for testing.

    Assertions:
        - Ensures the creation of a plot file in the expected location.
    """

    paths = np.random.rand(5, 10, 2)
    save_path = tmp_path / "test_plot.pdf"
    plot_paths_2d(paths, str(save_path))
    assert save_path.exists(), "Plot file should exist."


def test_plot_coefficients(tmp_path):
    """
    Test the plotting of coefficients along the sample paths.

    This test checks if `plot_coefficients` successfully generates a plot of
    coefficient values along simulated paths and saves it to a specified path.

    Assertions:
        - Confirms that the plot file is created as expected.
    """

    paths = np.random.rand(5, 10, 1)
    coeffs = np.random.rand(5, 10, 1)
    save_path = tmp_path / "test_plot.pdf"
    plot_coefficients(paths, coeffs, str(save_path))
    assert save_path.exists(), "Plot file should exist."


def test_ornstein_uhlenbeck_pdf():
    """
    Test the PDF calculation for the Ornstein-Uhlenbeck process.

    This function tests if `ornstein_uhlenbeck_pdf` correctly computes the probability
    density function values for given parameters and path data.

    Assertions:
        - Verifies that the output dimensions match the expected shape.
    """

    T_te = np.array([0, 0.5, 1.0]).reshape((-1, 1))
    X_te = np.random.rand(1, 3, 1)
    mu = np.array([0])
    theta = 1.0
    sigma = 0.1
    mu_0 = np.array([0])
    sigma_0 = 1.0
    result = ornstein_uhlenbeck_pdf(T_te, X_te, mu, theta, sigma, mu_0, sigma_0)
    assert result.shape == (3, 3), "Output shape is incorrect."


def test_plot_map_1d(tmp_path):
    """
    Test the plotting of 1D map values over time.

    This function checks whether `plot_map_1d` successfully creates a plot of 1D map
    values as a function of time and saves the plot to the specified location.

    Assertions:
        - Ensures the plot file exists after execution.
    """

    Ts = np.array([0, 1]).reshape((-1, 1))
    X = np.array([[0, 1]])
    map1 = np.random.rand(2, 2)
    save_name = "test_map"
    save_path = tmp_path / f"{save_name}.pdf"

    plot_map_1d(
        Ts,
        X,
        save_name,
        "Test Map",
        "Time",
        "Position",
        "Density",
        map1,
        save_path=str(tmp_path),
    )

    # Check if the file with the expected full path exists
    assert save_path.exists(), "Plot file should exist."


def test_plot_map_2d(tmp_path):
    """
    Test the plotting of 2D map values over time.

    This test verifies if `plot_map_2d` correctly generates a plot file of 2D map
    values based on given temporal and spatial discretizations, and saves it to
    the intended directory.

    Assertions:
        - Checks for the existence of the expected plot file.
    """

    Ts = np.array([0, 1]).reshape((-1, 1))
    X1s = np.array([0, 1])
    X2s = np.array([0, 1])
    map_v = np.random.rand(2, 2, 2)
    save_name = "test_plot"  # Just the name without the extension
    expected_file_path = (
        tmp_path / f"{save_name}.pdf"
    )  # Construct the full expected filename

    # Call your plotting function
    plot_map_2d(
        Ts,
        X1s,
        X2s,
        save_name,
        "Test Map",
        "X1",
        "X2",
        "Density",
        map_v,
        save_path=str(tmp_path),  # Provide the directory part here
    )

    # Check if the file with the expected full path exists
    assert (
        expected_file_path.exists()
    ), f"Expected plot file does not exist: {expected_file_path}"


def test_line_plot(tmp_path):
    """
    Test the line plot functionality.

    This test checks if `line_plot` can create a line plot based on given data points
    and save it to the specified path. It uses a sine wave as the test data.

    Assertions:
        - Ensures the creation of the plot file at the specified location.
    """

    Ts = np.linspace(0, 10, 100)
    values = np.sin(Ts)
    save_path = tmp_path / "test_plot.pdf"
    line_plot(Ts, values, str(save_path), "Sine Wave", "Time", "Amplitude")
    assert save_path.exists(), "Plot file should exist."
