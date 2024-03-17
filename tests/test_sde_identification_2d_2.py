import time
import numpy as np
from utils.utils import cartesian_products_of_rows
from methods.fp_estimator import FPEstimator
from methods.kde import ProbaDensityEstimator
from utils.data import (
    euler_maruyama,
    dubins,
    plot_paths_2d,
    plot_map_2d,
    line_plot,
)


# Simulation parameters.
n = 2  # Dimension of the state variable
v = 2.0
theta = 3.0  # Small for smoother
sigma = 0.3
mean_0, std_0 = np.array([0] * n), 0.5
T = 10  # Ending time
n_steps = 100  # Number of time steps
dt = T / n_steps  # Time step
T_tr = (np.arange(0, n_steps) * dt).reshape(-1, 1)  # Temporal discretization
n_paths_tr = 2000  # Number of train paths to draw
n_paths_te = 100  # Number of test paths to draw


# Drift and diffusion coefficients for the Dubins process.
b, sigma_func = dubins(v=v, theta=theta, sigma=sigma)


# Generate train and test data sets of sample paths from the SDE associated to the provided coefficients.
X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths_tr, T, n, mean_0, std_0)


# Plot the training data set.
n_plot = 100
plot_paths_2d(
    X_tr[:n_plot],
    save_path="../plots/test_sde_identification_2d_2/true_samples.pdf",
)


# Initialize the probability density estimator.
kde = ProbaDensityEstimator()
kde.T = T


# Choose the hyperparameters for the probability density estimator.
gamma_t = 1e-0
mu_x = 10
L_t = 1e-5
c_kernel = 1e-6
kde.gamma_t = gamma_t
kde.mu_x = mu_x
kde.L_t = L_t
kde.c_kernel = c_kernel


# Fit the probability density estimator using the training paths.
t0 = time.time()
kde.fit(T_tr=T_tr, X_tr=X_tr)
print(f"KDE fitting time: {time.time() - t0}")

# Generate a uniform grid on [0,T] x [-beta/2, beta/2]^n.
n_t_grid = 50
n_x_grid = 1000
n_x_grid = int(n_x_grid ** (1 / 2))
dt_grid = T / n_t_grid
t_interpolation = dt / 2  # Time offset between test and train times
T_grid = np.array([i * dt_grid + t_interpolation for i in range(n_t_grid)]).reshape(
    -1, 1
)
x_1_min, x_1_max = X_tr[:, :, 0].min() - 1, X_tr[:, :, 0].max() + 1
x_2_min, x_2_max = X_tr[:, :, 1].min() - 1, X_tr[:, :, 1].max() + 1
X_1_grid = np.linspace(x_1_min, x_1_max, n_x_grid)
X_2_grid = np.linspace(x_2_min, x_2_max, n_x_grid)
X_grid = cartesian_products_of_rows(X_1_grid.reshape(-1, 1), X_2_grid.reshape(-1, 1))


# Predict the probability density on the uniform grid.
t0 = time.time()
p_pred_grid = kde.predict(T_te=T_grid, X_te=X_grid)
print(f"KDE prediction time: {time.time() - t0}")


# Plot the predicted probability density values.
plot_map_2d(
    T_grid,
    X_1_grid,
    X_2_grid,
    f"p_pred",
    r"Estimated density - $\hat{p}$",
    xlabel="$x_1$",
    ylabel="$x_2$",
    alt_label=r"$t$",
    map_v=p_pred_grid,
    save_path="../plots/test_sde_identification_2d_2/",
    T_plot=T_tr,
    X_plot=X_tr,
)


# Initialize the Fokker-Planck matching estimator.
estimator = FPEstimator()


# Choose the hyperparameters for the Fokker-Planck matching estimator.
gamma_z = 1e-2
c_kernel_z = 1e-6
estimator.la = 1e-3
estimator.be = (x_2_min - x_2_max) * (x_1_min - x_1_max)
estimator.gamma_z = gamma_z
estimator.T = T
estimator.c_kernel = c_kernel


def p(Ts, X, partial):
    return kde.predict(Ts, X, partial=partial)


# Generate training points (t,x) for Fokker-Planck using training sample paths
X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths_tr, T, n, mean_0, 30 * std_0)
n_t_fp = 30
n_x_fp = 100
sub_idx_t = np.linspace(0, len(T_tr) - 1, n_t_fp, dtype=int)
T_fp = T_tr[sub_idx_t]
X_fp = np.zeros((n_x_fp, n_t_fp, n))
for i in range(n_t_fp):
    sub_idx_x = np.random.choice(n_paths_tr, size=n_x_fp, replace=False)
    X_fp_i = X_tr[sub_idx_x, sub_idx_t[i], :]
    X_fp[:, i, :] = X_fp_i

# Plot the predicted probability density values with selected subset of the training set for Fokker-Planck matching.
plot_map_2d(
    T_grid,
    X_1_grid,
    X_2_grid,
    f"FP_training_set",
    r"Fokker-Planck training set",
    xlabel="$x_1$",
    ylabel="$x_2$",
    alt_label=r"$t$",
    map_v=p_pred_grid,
    save_path="../plots/test_sde_identification_2d_2/",
    T_plot=T_fp,
    X_plot=X_fp,
)

T_fp = np.random.uniform(0, T, size=(n_t_fp, 1))
X_fp = np.zeros((n_x_fp, n_t_fp, n))
for i in range(n_t_fp):
    X_fp_i = np.random.uniform([x_1_min, x_2_min], [x_1_max, x_2_max], size=(n_x_fp, n))
    X_fp[:, i, :] = X_fp_i

# Plot the predicted probability density values with selected subset of the training set for Fokker-Planck matching.
plot_map_2d(
    T_grid,
    X_1_grid,
    X_2_grid,
    f"uniform_training_set",
    r"Uniform training set",
    xlabel="$x_1$",
    ylabel="$x_2$",
    alt_label=r"$t$",
    map_v=p_pred_grid,
    save_path="../plots/test_sde_identification_2d_2/",
    T_plot=T_fp,
    X_plot=X_fp,
)


# Fit the  Fokker-Planck matching estimator with the training points.
t0 = time.time()
estimator.fit(T_tr=T_fp, X_tr=X_fp, p=p)
print(f"Fokker-Planck matching fitting time: {time.time() - t0}", flush=True)


# Compute MSE of the FP fitting on the train and test sets.
compute_mse = True
if compute_mse:
    n_grid_fp = 100
    n_grid_fp = int(n_grid_fp ** (1 / 2))
    X_1_grid_fp = np.linspace(x_1_min, x_1_max, n_grid_fp)
    X_2_grid_fp = np.linspace(x_2_min, x_2_max, n_grid_fp)
    X_grid_fp = cartesian_products_of_rows(
        X_1_grid_fp.reshape(-1, 1), X_2_grid_fp.reshape(-1, 1)
    )
    t0 = time.time()
    _, _, MSE_tr, norms_tr = estimator.compute_mse()
    d_p_te_kolmogorov, d_p_te_direct, MSE_te, norms_te = estimator.compute_mse(
        T_grid, X_grid_fp
    )
    print(f"Fokker-Planck MSE computation time: {time.time() - t0}", flush=True)
    print(f"Fokker-Planck train MSE : {MSE_tr}")
    print(f"Fokker-Planck test MSE : {MSE_te}")
    print(f"Train target mean squared norm: {norms_tr[1]}")
    print(f"Train prediction mean squared norm: {norms_tr[0]}")
    print(f"Test target mean squared norm: {norms_te[1]}")
    print(f"Test prediction mean squared norm: {norms_te[0]}")

    # Plot to visualize the accuracy of the Fokker-Planck matching.
    plot_levels = None
    norm = None
    idx_grid = list(np.linspace(0, len(T_grid) - 1, 10, dtype=int))
    for j, idx in enumerate(idx_grid):
        # Plot the values of the time derivative of the estimated probability density values.
        plot_levels_0, norm_0 = plot_map_2d(
            T_grid[idx : idx + 1],
            X_1_grid_fp,
            X_2_grid_fp,
            f"d_t_p_direct_te" + f"(t_{j})",
            r"$\frac{\partial \hat{p}}{\partial t}$" + f"$(t_{j})$",
            xlabel="$x_1$",
            ylabel="$x_2$",
            alt_label=r"$t$",
            map_v=d_p_te_direct[idx : idx + 1],
            save_path="../plots/test_sde_identification_2d_2/",
            fixed_t=True,
            plot_levels=plot_levels,
            norm_bar=norm,
        )

        # Save the levels of the first plot to be able then to use it for all plots.
        if j == 0:
            plot_levels = plot_levels_0
            norm = norm_0

        # Plot the values of the estimated probability density values through the Kolmogorov operator.
        plot_map_2d(
            T_grid[idx : idx + 1],
            X_1_grid_fp,
            X_2_grid_fp,
            f"d_t_p_kolmogorov_te" + f"(t_{j})",
            r"$(\mathcal{L}^{(\hat b, \hat \sigma)})^* \hat p$" + f"$(t_{j})$",
            xlabel="$x_1$",
            ylabel="$x_2$",
            alt_label=r"$t$",
            map_v=d_p_te_kolmogorov[idx : idx + 1],
            save_path="../plots/test_sde_identification_2d_2/",
            fixed_t=True,
            plot_levels=plot_levels,
            norm_bar=norm,
        )


# Generate a set of sample paths from the SDE associated to the estimated coefficients (b, sigma).
print("Start sampling")


def b_pos(t, x):
    b_v = estimator.predict(
        T_te=np.array(t).reshape(1, 1),
        X_te=np.array(x).reshape(1, n),
        thresholding=True,
    )[0].reshape((2,))
    return b_v


def sigma_func_pos(t, x):
    s = estimator.predict(
        T_te=np.array(t).reshape(1, 1),
        X_te=np.array(x).reshape(1, n),
        thresholding=True,
    )[1][0, 0]
    return s


n_paths_samp = 100
n_steps_samp = 200
T_samp = T
dt_samp = T_samp / n_steps_samp  # Time step
t0 = time.time()
std_0 = 0.1
paths_samp = euler_maruyama(
    b_pos,
    sigma_func_pos,
    n_steps_samp,
    n_paths_samp,
    T_samp,
    n,
    mean_0,
    std_0,
    time=True,
)
print(f"End sampling")
print(f"Sampling computation time: {time.time() - t0}")


# Plot the set.
T_samp = (np.arange(0, n_steps_samp) * dt_samp).reshape(-1, 1)
plot_paths_2d(paths_samp, save_path="../plots/test_sde_identification_2d_2/samples.pdf")


# Compute and plot mean and variance of true SDE w.r.t. time.
mean = np.mean(X_tr, axis=0)
std = np.std(X_tr, axis=0)
save_path = "../plots/test_sde_identification_2d_2/mean_sample_true.pdf"
line_plot(
    mean[:, 0], mean[:, 1], save_path, title=r"$\mu(t)$", xlabel="$x_1$", ylabel="$x_2$"
)
save_path = "../plots/test_sde_identification_2d_2/std_sample_true.pdf"
line_plot(
    std[:, 0],
    std[:, 1],
    save_path,
    title=r"$\sigma(t)$",
    xlabel="$x_1$",
    ylabel="$x_2$",
)


# Compute and plot mean and variance of estimated SDE w.r.t. time.
mean = np.mean(paths_samp, axis=0)
std = np.std(paths_samp, axis=0)
save_path = "../plots/test_sde_identification_2d_2/mean_sample_es.pdf"
line_plot(
    mean[:, 0],
    mean[:, 1],
    save_path,
    title=r"$\hat \mu(t)$",
    xlabel="$x_1$",
    ylabel="$x_2$",
)
save_path = "../plots/test_sde_identification_2d_2/std_sample_es.pdf"
line_plot(
    std[:, 0],
    std[:, 1],
    save_path,
    title=r"$\hat \sigma(t)$",
    xlabel="$x_1$",
    ylabel="$x_2$",
)

# Plot the predicted probability density values and the points sampled from the estimated SDE.
plot_map_2d(
    T_grid,
    X_1_grid,
    X_2_grid,
    f"p_pred_and_fp_samples",
    title=r"$\hat p$ and estimated SDE sample paths",
    xlabel="$x_1$",
    ylabel="$x_2$",
    alt_label=r"$t$",
    map_v=p_pred_grid,
    save_path="../plots/test_sde_identification_2d_2/",
    T_plot=T_samp,
    X_plot=paths_samp,
)
