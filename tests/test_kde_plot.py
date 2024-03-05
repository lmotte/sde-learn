import numpy as np
from methods.kde import ProbaDensityEstimator
from utils.data import (
    euler_maruyama,
    ornstein_uhlenbeck,
    ornstein_uhlenbeck_pdf,
    plot_map_1d,
)

# Simulation parameters.
n = 2  # Dimension of the state variable
mu = np.array([1.0] * n)  # Mean
theta = 0.6  # Rate of mean reversion
sigma = 0.5  # Volatility
x0 = np.array([0.0] * n)  # x(t=0)
eps = 0.1  # Starting time
T = 10.0  # Ending time
mu_0, sigma_0 = np.array([0.5] * n), sigma / (2 * theta) ** (
    1 / 2
)  # Mean and std at t=0
n_steps = 100  # Number of time steps
dt = T / n_steps  # Time step
T_tr = (np.arange(1, n_steps + 1) * dt + eps).reshape(-1, 1)  # Temporal discretization
n_paths = 3000  # Number of paths to draw

# Drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
b, sigma_func = ornstein_uhlenbeck(mu=mu, theta=theta, sigma=sigma)

# Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0)

# Initialize the probability density estimator.
estimator = ProbaDensityEstimator()

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

# Fit the probability density estimator to the sample paths.
estimator.fit(T_tr=T_tr, X_tr=X_tr)

# Predict the values of the probability density on a test set.
t_interpolation = dt / 2  # Time offset between test and train times
T_te = np.array([eps + i * dt + t_interpolation for i in range(n_steps)]).reshape(-1, 1)

# Generate n_x points x in [-R, R]^n.
Q = 200
X_te = np.random.uniform(-0.2, 0.9, size=(Q, n)).reshape((-1, 1, n))

# Build a test set on a 1D subspace by projecting the n_x points.
vec = mu - x0
norm = np.linalg.norm(vec)
vec_norm = vec / norm
X_te_prof_coef = X_te.dot(vec)
X_te_proj = X_te_prof_coef.dot(vec.reshape(1, -1))
X_te_proj = X_te_proj.reshape((-1, 1, n))

# Predict the probability density on the test set.
p_pred = estimator.predict(T_te=T_te, X_te=X_te_proj)
p_true = ornstein_uhlenbeck_pdf(T_te, X_te_proj, mu, theta, sigma, mu_0, sigma_0)

# Plot the predictions.
save_name = f"p_true_fixed_x"
title = r"True density - $p$"
save_path = "../plots/test_kde_plot/"

plot_map_1d(
    T_te,
    X_te_prof_coef,
    f"p_pred",
    r"$\hat{p}$",
    xlabel="t",
    ylabel="",
    alt_label=r"$x$",
    map1=p_pred,
    save_path=save_path,
)

# Plot the true values of the density.
plot_map_1d(
    T_te,
    X_te_prof_coef,
    f"p_true",
    r"$p$",
    xlabel="t",
    ylabel="",
    alt_label=r"$x$",
    map1=p_true,
    save_path=save_path,
)

# Plot both.
plot_map_1d(
    T_te,
    X_te_prof_coef,
    f"p_pred_p_true",
    r"$p$ and $\hat{p}$",
    xlabel="t",
    ylabel="",
    alt_label=r"$x$",
    map1=p_true,
    map2=p_pred,
    save_path=save_path,
    legend1=r"$p$",
    legend2=r"$\hat{p}$",
)
