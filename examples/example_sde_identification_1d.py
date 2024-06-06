import time
import numpy as np
from methods.kde import ProbaDensityEstimator
from methods.fp_estimator import FPEstimator
from utils.data import (
    euler_maruyama,
    ornstein_uhlenbeck,
    ornstein_uhlenbeck_pdf,
    plot_map_1d,
    plot_paths_1d,
    line_plot,
)

# Simulation parameters.
n = 1  # Dimension of the state variable
mu = np.array([2.5] * n)  # Mean
theta = 0.5  # Rate of mean reversion
sigma = 0.5 * theta ** (1 / 2)  # Volatility
mean_0, std_0 = np.array([0.5] * n), sigma / (2 * theta) ** (
    1 / 2
)  # Mean and std at t=0
T = 10  # Ending time
n_steps = 400  # Number of time steps
dt = T / n_steps  # Time step
T_tr = (np.arange(0, n_steps) * dt).reshape(-1, 1)  # Temporal discretization
n_paths = 2000  # Number of paths to draw

# Get drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
b, sigma_func = ornstein_uhlenbeck(mu=mu, theta=theta, sigma=sigma)

# Generate a training data set of sample paths from the SDE associated to the provided coefficients.
X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mean_0, std_0)

# Plot the training data set.
n_plot = 100
save_path = "../plots/test_sde_identification_1d/true_samples.pdf"
plot_paths_1d(T_tr, X_tr[:n_plot], save_path=save_path)

# Initialize the probability density estimator.
kde = ProbaDensityEstimator()

# Choose the hyperparameters for the probability density estimator.
gamma_t = 1e-0
L_t = 1e-6
mu_x = 10
c_kernel = 1e-5
kde.gamma_t = gamma_t
kde.mu_x = mu_x
kde.L_t = L_t
kde.c_kernel = c_kernel

# Fit the probability density estimator to the sample paths.
kde.fit(T_tr=T_tr, X_tr=X_tr)

# Generate uniform grid of test points (t,x) in [0,T] x [-beta/2, beta/2]^n for beta > 0.
n_t_te = 50
n_x_te = 200
dt_te = T / n_t_te
t_interpolation = dt_te / 2  # Time offset between test and train times
T_te = np.array([i * dt_te + t_interpolation for i in range(n_t_te)]).reshape(-1, 1)
x_min, x_max = X_tr[:, :, 0].min() - 1, X_tr[:, :, 0].max() + 1
X_te = np.linspace(x_min, x_max, n_x_te).reshape(-1, 1)

# Predict the probability density on the test set.
p_pred = kde.predict(T_te=T_te, X_te=X_te)
p_true = ornstein_uhlenbeck_pdf(
    T_te, X_te.reshape((-1, 1, 1)), mu, theta, sigma, mean_0, std_0
)

# Initialize the Fokker-Planck matching estimator.
estimator = FPEstimator()

# Choose the hyperparameters for the Fokker-Planck matching estimator.
gamma_z = 1e-2
c_kernel_z = 1e-2
la = 1e-1
estimator.gamma_z = gamma_z
estimator.la = la
estimator.be = x_max - x_min
estimator.T = T
estimator.c_kernel = c_kernel

# Generate training points (t,x) uniformly in [0,T] x [-beta/2, beta/2]^n.
n_t_fp = 50
n_fp = 50
T_fp = np.random.uniform(0, T, size=(n_t_fp, 1))
X_fp = np.random.uniform(x_min, x_max, size=(n_fp, 1))


# Fit the  Fokker-Planck matching estimator with the training samples.
def p(Ts, X, partial):
    return kde.predict(Ts, X, partial=partial)


estimator.fit(T_tr=T_fp, X_tr=X_fp, p=p)

# Compute MSE of the FP fitting on the train and test sets.
_, _, MSE_tr, _ = estimator.compute_mse()
d_p_te_kolmogorov, d_p_te_direct, MSE_te, norms = estimator.compute_mse(T_te, X_te)
print(f"Fokker-Planck train MSE : {MSE_tr}")
print(f"Fokker-Planck test MSE : {MSE_te}")
print(f"Target mean squared norm: {norms[1]}")
print(f"Prediction mean squared norm: {norms[0]}")

# Plot the true probability density values.
xlabel = "$x$"
ylabel = ""
save_path = "../plots/test_sde_identification_1d/"
save_name = "p_true"
title = r"True density - $p$"
alt_label = r"$t$"
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=p_true,
    save_path=save_path,
)

# Plot the estimated probability density values.
xlabel = "$x$"
ylabel = ""
save_name = f"p_pred"
title = r"Estimated density - $\hat{p}$"
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=p_pred,
    save_path=save_path,
)

# Plot both.
xlabel = "$x$"
ylabel = ""
save_name = f"p_pred_p_true"
title = r"True and estimated densities - $p$ and $\hat{p}$"
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=r"Time  $t$",
    map1=p_true,
    map2=p_pred,
    save_path=save_path,
    legend1="p(t,x)",
    legend2=r"$\hat{p}(t,x)$",
)

# Plot both (with varying t on the x-axis, for several fixed x).
xlabel = "$t$"
ylabel = ""
save_name = f"p_true_fixed_x"
title = r"True density - $p$"
alt_label = r"$x$"
plot_map_1d(
    X_te.reshape((-1, 1, 1)),
    T_te,
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=p_true.T,
    save_path=save_path,
)
save_name = f"p_pred_fixed_x"
title = r"Estimated density -$\hat p$"
plot_map_1d(
    X_te.reshape((-1, 1, 1)),
    T_te,
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=p_pred.T,
    save_path=save_path,
)

# Plot both.
save_name = f"p_both_fixed_x"
title = r"True and estimated densities"
plot_map_1d(
    X_te.reshape((-1, 1, 1)),
    T_te,
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=p_true.T,
    map2=p_pred.T,
    save_path=save_path,
    legend1="p(t,x)",
    legend2=r"$\hat{p}(t,x)$",
)

# Plot the values of the time derivative of the estimated probability density values
# (for several fixed x, with varying t on the x-axis).
xlabel = "$t$"
ylabel = ""
save_name = f"d_t_p_direct_fixed_x"
title = r"$\frac{\partial \hat{p}}{\partial t}$"
plot_map_1d(
    X_te.reshape((-1, 1, 1)),
    T_te,
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=d_p_te_direct.T,
    save_path=save_path,
)

# Plot the values of the time derivative of the estimated probability density values.
xlabel = "$x$"
ylabel = ""
save_name = f"d_t_p_direct"
title = r"$\frac{\partial \hat{p}}{\partial t}$"
alt_label = r"$t$"
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=d_p_te_direct,
    save_path=save_path,
)

# Plot the values of the estimated probability density values through the Kolmogorov operator.
title = r"$(\mathcal{L}^{(\hat b, \hat \sigma)})^* \hat p$"
save_name = f"d_t_p_kolmogorov"
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=d_p_te_kolmogorov,
    save_path=save_path,
)

# Plot both.
save_name = f"d_t_p_kolmogorov_direct"
title = r"$\frac{\partial \hat{p}}{\partial t}$ and $(\mathcal{L}^{(\hat b, \hat \sigma)})^* \hat p$"
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    save_name,
    title,
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=d_p_te_kolmogorov,
    map2=d_p_te_direct,
    save_path=save_path,
    legend1=r"$(\mathcal{L}^{(\hat b, \hat \sigma)})^* \hat p$",
    legend2=r"$\frac{\partial \hat{p}}{\partial t}$",
)

# Predict the values of the SDE's coefficients on the test set.
B_pred_pos, S_pred_pos, B_pred, S_pred = estimator.predict(T_te=T_te, X_te=X_te)
S_pred = S_pred.reshape((n_t_te, n_x_te))
B_pred = B_pred.reshape((n_t_te, n_x_te))
S_pred_pos = S_pred_pos.reshape((n_t_te, n_x_te))
B_pred_pos = B_pred_pos.reshape((n_t_te, n_x_te))

# Plot these values.
xlabel = "$x$"
ylabel = ""
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    f"sigma_both",
    r"$\sigma^2$",
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=S_pred,
    map2=S_pred_pos,
    save_path=save_path,
    legend1="No cons.",
    legend2=r"Pos. cons.",
)
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    f"b_both",
    r"$b$",
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=B_pred,
    map2=B_pred_pos,
    save_path=save_path,
    legend1="No cons.",
    legend2=r"Pos. cons.",
)
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    f"sigma_pos",
    r"$\sigma^2$",
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=r"$t$",
    map1=S_pred_pos,
    save_path=save_path,
)
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    f"b_pos",
    r"$b$",
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=B_pred_pos,
    save_path=save_path,
)
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    f"sigma",
    r"$\sigma^2$",
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=S_pred,
    save_path=save_path,
)
plot_map_1d(
    T_te,
    X_te.reshape((-1, 1, 1)),
    f"b",
    r"$b$",
    xlabel=xlabel,
    ylabel=ylabel,
    alt_label=alt_label,
    map1=B_pred,
    save_path=save_path,
)

# Generate a set of sample paths from the SDE associated to the estimated coefficients (b, sigma).
print("Start sampling")
n_paths = 100
n_steps = 100


def b(t, x):
    b_pred = estimator.predict(
        T_te=np.array(t).reshape(1, 1), X_te=np.array(x).reshape(1, 1)
    )[0]
    return b_pred


def sigma_func(t, x):
    s_pred = estimator.predict(
        T_te=np.array(t).reshape(1, 1),
        X_te=np.array(x).reshape(1, 1),
        thresholding=True,
    )[1]
    return s_pred


t0 = time.time()
paths_pos = euler_maruyama(
    b, sigma_func, n_steps, n_paths, T, n, mean_0, std_0, time=True
)
print(f"End sampling")
print(f"Sampling computation time: {time.time() - t0}")

# Plot the set.
dt = T / n_steps
T_samp = (np.arange(0, n_steps) * dt).reshape(-1, 1)
save_path = "../plots/test_sde_identification_1d/samples_pos.pdf"
plot_paths_1d(T_samp, paths_pos, save_path=save_path)

# Compute and plot mean and variance of estimated SDE w.r.t. time.
mean_pos = np.mean(paths_pos, axis=0)
std_pos = np.std(paths_pos, axis=0)
save_path = "../plots/test_sde_identification_1d/mean_pos.pdf"
line_plot(T_samp, mean_pos, save_path, title=r"$\hat \mu(t)$")
save_path = "../plots/test_sde_identification_1d/std_pos.pdf"
line_plot(T_samp, std_pos, save_path, title=r"$\hat \sigma(t)$")

# Compute and plot mean and variance of true SDE w.r.t. time.
mean_true = np.exp(-theta * T_samp) * (mean_0 - mu) + mu
var_true = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * T_samp)) + np.exp(
    -2 * theta * T_samp
) * std_0**2
std_true = var_true ** (1 / 2)
save_path = "../plots/test_sde_identification_1d/mean_true.pdf"
title = r"$\mu(t)$"
line_plot(T_samp, mean_true, save_path, title)
save_path = "../plots/test_sde_identification_1d/std_true.pdf"
title = r"$\sigma(t)$"
line_plot(T_samp, std_true, save_path, title)
