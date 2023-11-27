import numpy as np
from utils.data import euler_maruyama, ornstein_uhlenbeck, plot_ornstein_uhlenbeck_paths


# Simulation parameters.
n = 1  # Dimension of the state variable
mu = np.array([1.0] * n)  # Mean
theta = 1.  # Rate of mean reversion
sigma = 0.1  # Volatility
x0 = np.array([0.0] * n)  # x(t=0)
eps = 0.1  # Starting time
T = 10.0  # Ending time
n_steps = 100   # Number of time steps
dt = T / n_steps  # Time step
Ts = (np.arange(1, n_steps + 1) * dt + eps).reshape(-1, 1)   # Temporal discretization
n_paths = 10  # Number of paths to draw

# Drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
b, sigma_func = ornstein_uhlenbeck(mu=mu, theta=theta, sigma=sigma)

# Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
paths = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n)

# Plot the training data set.
plot_ornstein_uhlenbeck_paths(Ts, paths)
