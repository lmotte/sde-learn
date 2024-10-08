import numpy as np
from utils.data import euler_maruyama, ornstein_uhlenbeck, plot_paths_1d

# Simulation parameters.
n = 1  # Dimension of the state variable
mu = np.array([2.5] * n)  # Mean
theta = 0.5  # Rate of mean reversion
sigma = 0.5 * theta ** (1 / 2)  # Volatility
mu_0, sigma_0 = np.array([0.5] * n), sigma / (2 * theta) ** (
    1 / 2
)  # Mean and std at t=0
T = 10  # Ending time
n_steps = 100  # Number of time steps
dt = T / n_steps  # Time step
Ts = (np.arange(0, n_steps) * dt).reshape(-1, 1)  # Temporal discretization
n_paths = 10  # Number of paths to draw

# Drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
b, sigma_func = ornstein_uhlenbeck(mu=mu, theta=theta, sigma=sigma)

# Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
paths = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0)

# Plot the training data set.
save_path = "../plots/test_ornstein_uhlenbeck_paths_plot/ornstein_uhlenbeck_samples.pdf"
plot_paths_1d(Ts, paths, save_path=save_path)
