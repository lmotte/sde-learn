import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap


def euler_maruyama(
    b, sigma, n_steps, n_paths, T, n, mu_0, sigma_0, coef_save=False, time=False
):
    """
    Simulate sample paths of a multidimensional stochastic differential equation (SDE)
    using the Euler-Maruyama method.

    Parameters:
        b (callable): The drift coefficient function.
        sigma (callable): The diffusion coefficient function.
        n_steps (int): The number of time steps for discretization of the total duration T.
        n_paths (int): The number of sample paths to simulate.
        T (float): The total duration of the simulation.
        n (int): The dimension of the state variable.
        mu_0 (numpy.ndarray): Array of shape (n, 1) mean of the initial distribution
        sigma_0 (float): Standard deviation of the initial distribution
        coef_save (bool, optional): Whether to save coefficient values along with paths. Defaults to False.
        time (bool, optional): Whether to print the simulation progress as a percentage. Defaults to False.
    Returns:
        numpy.ndarray or tuple: If coef_save is False, returns an array of shape (n_paths, n_steps, n)
            containing the simulated paths. If coef_save is True, returns a tuple containing three arrays:
            paths, B_save, S_save, where paths is the array of simulated paths, B_save is an array of shape
            (n_paths, n_steps, n) containing the drift coefficient values for each path and time step, and
            S_save is an array of shape (n_paths, n_steps, 1) containing the diffusion coefficient values
            for each path and time step. If time is True, progress is printed to the console during simulation.
    """

    dt = T / n_steps  # Time step
    sqrt_dt = np.sqrt(dt)

    # Initialization of paths and coefficient arrays
    paths = np.zeros((n_paths, n_steps, n))
    B_save = np.zeros((n_paths, n_steps, n)) if coef_save else None
    S_save = np.zeros((n_paths, n_steps, 1)) if coef_save else None

    # Simulation loop
    for i in range(n_paths):
        x = np.zeros((n_steps, n))
        cov_0 = sigma_0**2 * np.eye(n)
        x[0] = np.random.multivariate_normal(mu_0, cov_0)
        b_save = np.zeros((n_steps, n)) if coef_save else None
        s_save = np.zeros((n_steps, 1)) if coef_save else None

        for j in range(n_steps - 1):
            dW = np.random.normal(0, 1, size=n) * sqrt_dt
            t = dt * j
            b_t = b(t, x[j])
            s_t = sigma(t, x[j])
            x[j + 1] = x[j] + b_t * dt + s_t * dW

            if coef_save:
                b_save[j + 1] = b_t
                s_save[j + 1] = s_t

        paths[i, :, :] = x

        # Save the coefficients' values
        if coef_save:
            B_save[i, :, :] = b_save
            S_save[i, :, :] = s_save

        # Progress reporting
        if time:
            print(f"{int(i / n_paths * 100)}%", end=" ", flush=True)

    if coef_save:
        return paths, B_save, S_save

    return paths


def ornstein_uhlenbeck(mu, theta, sigma):
    """
    Returns the drift and diffusion coefficients for the Ornstein–Uhlenbeck process.

    Parameters:
        mu (numpy.ndarray): Mean towards which the process reverts.
        theta (float): The rate of mean reversion.
        sigma (float): The volatility parameter.

    Returns:
        tuple: A tuple containing the drift and diffusion coefficients.
    """

    def b(t, x):
        return theta * (mu - x)

    def sigma_func(t, x):
        return sigma

    return b, sigma_func


def dubins(v, theta, sigma):
    """
    Returns the drift and diffusion coefficients for the Dubins process.

    Parameters:
        v (float): Speed of the process.
        theta (float): Influences turning rate.
        sigma (float): Diffusion coefficient.

    Returns:
        tuple: A tuple containing the drift and diffusion coefficients.
    """

    def b(t, x):
        def u(z):
            return theta * np.sin(z / 10 * np.pi)

        return v * np.array([np.cos(u(t)), np.sin(u(t))])

    def sigma_func(t, x):
        return sigma

    return b, sigma_func


def gaussian(t1, x1, t2, x2, t3, x3, g):
    """
    Returns functions for Gaussian drift and diffusion coefficients based on the provided mean times, positions, and
     multiplicative constant.

    Parameters:
        t1, t2, t3 (float): Mean times for the Gaussian components.
        x1, x2, x3 (ndarray): Mean positions for the Gaussian components.
        g (float): Multiplicative constant.

    Returns:
        tuple: A tuple containing the drift and diffusion coefficient functions.
    """

    def b(t, x):
        b1 = np.exp(-g * np.linalg.norm(x - x1) * np.linalg.norm(t - t1))
        b2 = np.exp(-g * np.linalg.norm(x - x2) * np.linalg.norm(t - t2))
        b3 = np.exp(-g * np.linalg.norm(x - x3) * np.linalg.norm(t - t3))
        bs = b1 * np.array([1, 0]) + b2 * np.array([0, 1]) + b3 * np.array([-1, 1])
        return 10 * bs

    def sigma_func(t, x):
        s1 = np.exp(-g * np.linalg.norm(x - x1) * np.linalg.norm(t - t1))
        s2 = np.exp(-g * np.linalg.norm(x - x2) * np.linalg.norm(t - t2))
        s3 = np.exp(-g * np.linalg.norm(x - x3) * np.linalg.norm(t - t3))
        ss = s1 + s2 + s3
        return 0.2 * ss

    return b, sigma_func


def van_der_pol(mu, sigma):
    """
    Returns the drift and diffusion coefficients for the Van der Pol oscillator.

    Parameters:
        mu (float): Parameter for the Van der Pol oscillator.
        sigma (float): Volatility parameter.

    Returns:
        tuple: A tuple containing the drift and diffusion coefficients.
    """

    def b(x):
        return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])

    def sigma_func(_):
        return sigma

    return b, sigma_func


def plot_paths_1d(T, paths, save_path):
    """
    Plot 1-dimensional sample paths.

    This function generates a plot of 1-dimensional sample paths over a specified
    time interval. Each path is plotted with a slight transparency to visualize overlap
    and divergence among paths. The plot is then saved to a specified file path.

    Parameters:
        T (numpy.ndarray): Array of shape (n_steps,) containing the times of discretization.
                           This represents the time points at which the paths are evaluated.
        paths (numpy.ndarray): Array of shape (n_paths, n_steps, n) containing the simulated paths.
                               Each path represents a possible evolution over time, where `n_paths`
                               is the number of paths, `n_steps` is the number of time steps, and `n`
                               should be 1 for 1D paths.
        save_path (str): File path where the plot will be saved.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each path
    for idx, path in enumerate(paths, start=1):
        ax.plot(
            T, path[:, 0], label=f"Path {idx}", color="black", alpha=0.5, linewidth=1
        )

    ax.set_xlabel("Time t")
    ax.set_ylabel("X(t)")
    ax.set_title("Sample Paths")
    plt.savefig(save_path)
    plt.close(fig)


def plot_paths_2d(paths, save_path):
    """
    Plot 2-dimensional sample paths.

    Parameters:
        paths (numpy.ndarray): Array of shape (n_paths, n_steps, 2) containing the simulated paths.
        save_path (str): File path where the plot will be saved.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each path
    for idx, path in enumerate(paths, start=1):
        ax.plot(
            path[:, 0],
            path[:, 1],
            label=f"Path {idx}",
            color="black",
            alpha=0.5,
            linewidth=1,
        )

    ax.set_xlabel("$X_1(t)$")
    ax.set_ylabel("$X_2(t)$")
    ax.set_title("2D Sample Paths")
    ax.set_aspect("equal", adjustable="box")  # Ensure equal aspect ratio
    plt.savefig(save_path)
    plt.close(fig)


def plot_coefficients(paths, coeffs, save_path):
    """
    Plot the coefficients' values along the sample paths.

    Parameters:
        paths (numpy.ndarray): Array of shape (n_paths, n_steps, 1) containing the simulated paths.
        coeffs (numpy.ndarray): Array of shape (n_paths, n_steps, 1) containing the coefficients' values.
        save_path (str): File path where the plot will be saved.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (path, coeff) in enumerate(zip(paths, coeffs), start=1):
        ax.plot(
            path[:, 0],
            coeff[:, 0],
            label=f"Path {idx}",
            color="black",
            alpha=0.5,
            linewidth=1,
        )

    ax.set_xlabel("X(t)")
    ax.set_ylabel(r"b/$\sigma^2$(t, x(t))")
    ax.set_title("Coefficient values along paths")
    plt.savefig(save_path)
    plt.close(fig)


def ornstein_uhlenbeck_pdf(T_te, X_te, mu, theta, sigma, mu_0, sigma_0):
    """
    Calculate the probability density function of the Ornstein-Uhlenbeck process at x and time t.

    Parameters:
        T_te (numpy.ndarray): Array of shape (M_te,) containing the times.
        X_te (numpy.ndarray): Array of shape (n_te, M, n) containing the simulated paths.
        mu (numpy.ndarray): Mean towards which the process reverts.
        theta (float): Mean reversion strength or speed of mean reversion.
        sigma (float): Volatility of the process.
        mu_0 (numpy.ndarray): Array of shape (n, 1) mean of the initial distribution
        sigma_0 (float): Standard deviation of the initial distribution

    Returns:
        numpy.ndarray: Array of shape (M_te, M * n_te) containing the probability density of the Ornstein-Uhlenbeck
         process evaluated at all (t,x) in (T_grid x X_te).
    """

    M_te = len(T_te)
    n_te, M, n = X_te.shape
    p = np.zeros((M_te, n_te * M))

    for i in range(M_te):
        t = T_te[i]
        mu_t = np.exp(-theta * t) * mu_0 + (1 - np.exp(-theta * t)) * mu
        var_t = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t)) + np.exp(
            -2 * theta * t
        ) * sigma_0**2
        for j in range(n_te):
            for k in range(M):
                x = X_te[j][k]
                norm = np.linalg.norm(x - mu_t)
                pdf_value = (
                    var_t ** (-n / 2)
                    * (2 * np.pi) ** (-n / 2)
                    * np.exp(-0.5 * norm**2 / var_t)
                )
                p[i, j * M + k] = pdf_value

    return p


def plot_map_1d(
    Ts,
    X,
    save_name,
    title,
    xlabel,
    ylabel,
    alt_label,
    map1,
    map2=None,
    legend1=None,
    legend2=None,
    save_path=None,
):
    """
    Plots the 1d values of maps f(t, x) for given data.

    Parameters:
        Ts (numpy.ndarray): Array of shape (M, ) containing the time discretization.
        X (numpy.ndarray): Array of shape (n_x, 1) containing the spatial discretization.
        save_name (str): Name used for saving the plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        alt_label (str): Label for the color bar.
        map1 (numpy.ndarray): Probability density values for the first distribution.
        map2 (numpy.ndarray, optional): Probability density values for the second distribution.
        legend1 (str, optional): Legend label for the first distribution. Required if `map2` is provided.
        legend2 (str, optional): Legend label for the second distribution. Required if `map2` is provided.
        save_path (str): File path where the plot will be saved.
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("viridis")
    norm = colors.Normalize(min(Ts), max(Ts))

    # Select 20 time points for plotting
    L = np.linspace(0, len(Ts) - 1, 20, dtype=int)
    X_flat = X.flatten()  # Flatten X for scatter plot compatibility

    for i in L:
        t = round(float(Ts[i]), 1)
        col = float((t - Ts[0]) / (Ts[-1] - Ts[0]))

        ax.scatter(X_flat, map1[i], marker="x", s=20, c=[cmap(col)], label=f"t={t}")

        if map2 is not None:
            ax.scatter(X_flat, map2[i], s=20, c=[cmap(col)], label=f"t={t}")

    # Set labels, title, and color bar
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=alt_label)

    # Handle legend for p1 and p2
    if map2 is not None:
        ax.legend(
            [
                plt.Line2D([0], [0], marker="x", color="black", lw=0, markersize=5),
                plt.Line2D([0], [0], marker="o", color="black", lw=0, markersize=5),
            ],
            [legend1, legend2],
        )

    plt.savefig(os.path.join(save_path, f"{save_name}.pdf"))
    plt.close(fig)


def plot_map_2d(
    Ts,
    X1s,
    X2s,
    name,
    title,
    xlabel,
    ylabel,
    alt_label,
    map_v,
    save_path=None,
    fixed_t=False,
    plot_levels=None,
    norm_bar=None,
    T_plot=None,
    X_plot=None,
):
    """
    Plots 2D values of maps f(t, x1, x2) for given data.

    Parameters:
        Ts (numpy.ndarray): Temporal discretization.
        X1s (numpy.ndarray): Spatial discretization of the first dimension.
        X2s (numpy.ndarray): Spatial discretization of the second dimension.
        name (str): Name used for saving the plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        alt_label (str): Label for the color bar.
        map_v (numpy.ndarray): Map's values, should match the shape of Ts, X1s, and X2s meshgrid.
        save_path (str, optional): File path where the plot will be saved.
        fixed_t (bool, optional): If True, use a fixed number of contour levels; otherwise, adapt. Defaults to False.
        plot_levels (list, optional): Specific contour levels to plot. If None, automatically determined.
        Defaults to None.
        norm_bar (matplotlib.colors.Normalize, optional): Normalization for the color bar.
        If None, it is calculated. Defaults to None.
        T_plot (numpy.ndarray, optional): Times of points to scatter.
        X_plot (numpy.ndarray, optional): Positions of points to scatter.

    Returns:
        plot_levels (list): The contour levels used in the plot.
        norm (matplotlib.colors.Normalize): The normalization used for the color bar.
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("viridis")

    # Compute time indices to plot for the map. Handle single or multiple time points.
    time_indices = [0] if len(Ts) == 1 else np.linspace(0, len(Ts) - 1, 10, dtype=int)

    # Compute minimum and maximum time
    t_min, t_max = Ts[0][0], Ts[-1][0]

    # Compute time indices to plot for the set (T_plot, X_plot).
    idx_plot = None
    if T_plot is not None:
        idx_plot = np.linspace(0, len(T_plot) - 1, 10, dtype=int)
        idx_plot = (idx_plot + idx_plot[1] // 2)[:-1]
        t_min = np.min([Ts[0][0], T_plot[0][0]])
        t_max = np.max([Ts[-1][0], T_plot[-1][0]])

    for i in range(len(time_indices)):
        # Plot the map x1, x2 -> map(T_plot(idx_plot(i), x1, x2).
        idx_t = time_indices[i]
        t = Ts[idx_t][0]
        x, y = np.meshgrid(X1s, X2s)
        z = map_v[idx_t].reshape(x.shape).T

        # Determine contour levels and colormap based on 'fixed_t' flag.
        num_levels = 14 if not fixed_t else 30
        levels = np.linspace(z.min(), z.max(), num_levels)

        if not fixed_t:
            plot_levels = sorted(levels)[-8:]
            col_ratio = (t - t_min) / (t_max - t_min)
            cmap_i = LinearSegmentedColormap.from_list(
                "custom_cmap", [(1, 1, 1, 1), cmap(col_ratio)], N=num_levels
            )
        else:
            plot_levels = plot_levels if plot_levels is not None else sorted(levels)
            norm_bar = (
                norm_bar if norm_bar is not None else colors.Normalize(z.min(), z.max())
            )
            cmap_i = cmap

        # Contour plotting.
        plt.contourf(x, y, z, levels=plot_levels, cmap=cmap_i)
        plt.contour(
            x,
            y,
            z,
            levels=[plot_levels[0], plot_levels[-1]],
            colors="black",
            linewidths=0.1,
            linestyles="dashed",
        )
        plt.plot(x, y, ".", ms=0.1, color="black")

        # Optionally plot scatter points for specific time indices if T_plot and X_plot are provided.
        if idx_plot is not None and i < len(idx_plot):
            idx_t_plot = idx_plot[i]
            t_plot = T_plot[idx_t_plot][0]
            col_plot = (t_plot - t_min) / (t_max - t_min)
            plt.scatter(
                X_plot[:, idx_t_plot, 0],
                X_plot[:, idx_t_plot, 1],
                c=[cmap(col_plot)],
                s=1,
            )

    # Set labels, title, color bar, and save.
    norm = colors.Normalize(t_min, t_max) if norm_bar is None else norm_bar
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, aspect="equal")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=alt_label)
    plt.savefig(os.path.join(save_path, f"{name}.pdf"))
    plt.close(fig)

    return plot_levels, norm


def line_plot(Ts, values, save_path, title, xlabel="t", ylabel=""):
    """
    Creates and saves a line plot.

    Parameters:
    Ts : array-like
        X-values for the plot.
    values : array-like
        Y-values for each x-value in `Ts`.
    save_path : str
        Path including filename and extension where the plot is saved.
    title : str
        Plot title.
    xlabel : str, optional
        X-axis label (default is "t").
    ylabel : str, optional
        Y-axis label (default is empty).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ts, values)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.savefig(save_path)
    plt.close(fig)
