import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors


def euler_maruyama(b, sigma, n_steps, n_paths, T, n):
    """
    Simulate sample paths of a multidimensional stochastic differential equation (SDE)
    using the Euler-Maruyama method.

    Parameters:
        b (callable): The drift coefficient function.
        sigma (callable): The diffusion coefficient function.
        n_steps (int): The number of time steps for discretization.
        n_paths (int): The number of sample paths to simulate.
        T (float): The total duration of the simulation.
        n (int): The dimension of the state variable.

    Returns:
        numpy.ndarray: Array of shape (n_paths, n_steps, n) containing the simulated paths.
    """

    dt = T / n_steps  # Time step
    sqrt_dt = np.sqrt(dt)

    # Initialization of paths
    paths = np.zeros((n_paths, n_steps, n))

    for i in range(n_paths):
        # Initialization of each path
        x = np.zeros((n_steps, n))

        for t in range(n_steps - 1):
            # Generation of the Gaussian random variable
            dW = np.random.normal(0, 1, size=n) * sqrt_dt

            # Update the path using the Euler-Maruyama method
            x[t + 1] = x[t] + b(x[t]) * dt + sigma(x[t]) * dW

        # Record the path
        paths[i, :, :] = x

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

    def b(x):
        return theta * (mu - x)

    def sigma_func(_):
        return sigma

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
        return np.array([x[1], mu * (1 - x[0]**2) * x[1] - x[0]])

    def sigma_func(_):
        return sigma

    return b, sigma_func


def plot_ornstein_uhlenbeck_paths(T, paths):
    """
    Plot sample paths of the Ornstein-Uhlenbeck process.

    Parameters:
        T (numpy.ndarray): Array of shape (n_steps,) containing the times of discretization.
        paths (numpy.ndarray): Array of shape (n_paths, n_steps, n) containing the simulated paths.
    """

    # Plot the paths
    n_paths = paths.shape[0]
    for i in range(n_paths):
        plt.plot(T, paths[i, :, 0], label=f'Path {i + 1}', color='black')

    plt.xlabel('Time t')
    plt.ylabel('X(t)')
    plt.title(f'Ornstein Uhlenbeck - Sample Paths')
    plt.text(7, 0.5, r"$\mu=1$", va='center', fontsize=12)
    plt.text(7, 0.4, r"$\sigma=0.1$", va='center', fontsize=12)
    plt.text(7, 0.3, r"$\\theta=1$", va='center', fontsize=12)
    plt.text(7, 0.2, r"$x(t=0)=0$", va='center', fontsize=12)
    plt.savefig('../plots/test_ornstein_uhlenbeck_paths_plot/ornstein_uhlenbeck_samples.pdf')
    plt.close()


def ornstein_uhlenbeck_pdf(T_te, X_te, mu, theta, sigma, x0):
    """
    Calculate the probability density function of the Ornstein-Uhlenbeck process at x and time t.

    Parameters:
        T_te (numpy.ndarray): Array of shape (M_te,) containing the times.
        X_te (numpy.ndarray): Array of shape (n_te, M, n) containing the simulated paths.
        mu (numpy.ndarray): Mean towards which the process reverts.
        theta (float): Mean reversion strength or speed of mean reversion.
        sigma (float): Volatility of the process.
        x0 (numpy.ndarray): Initial condition or starting value of the process.

    Returns:
        numpy.ndarray: Array of shape (M_te, M * n_te) containing the probability density of the Ornstein-Uhlenbeck
         process evaluated at all (t,x) in (T_te x X_te).
    """

    M_te = len(T_te)
    n_te, M, n = X_te.shape
    p_true = np.zeros((M_te, n_te * M))

    for i in range(M_te):
        t = T_te[i]
        mu_t = np.exp(-theta * t) * x0 + (1 - np.exp(-theta * t)) * mu
        var_t = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))
        for j in range(n_te):
            for k in range(M):
                x = X_te[j][k]
                norm = np.linalg.norm(x - mu_t)
                pdf_value = var_t**(-n/2) * (2 * np.pi)**(-n/2) * np.exp(- 0.5 * norm ** 2 / var_t)
                p_true[i, j * M + k] = pdf_value

    return p_true


def plot_proba_density_predictions(T, Ts, X, name, title, p1, p2=None):
    """
    Plots the values of probability density functions for given data.

    Parameters:
    - T (float): Total time.
    - Ts (dict): # Temporal discretization.
    - X (numpy.ndarray): Array of shape (N, M, n) containing the simulated paths.
    - name (str): Name used for saving the plot.
    - title (str): Title of the plot.
    - p1 (dict): Dictionary of probability density values for the first distribution.
    - p2 (dict, optional): Dictionary of probability density values for the second distribution.
    """

    fig, ax = plt.subplots(figsize=(8, 5))   # Set up figure
    cmap = plt.cm.get_cmap('viridis')
    norm = colors.Normalize(min(Ts), max(Ts))

    # Plot densities' values for selected times point.
    L = [5, 7, 10, 15, 25, 90]
    for i in L:
        t = round(float(Ts[i]), 1)
        plt.scatter(list(X), list(p1[i]), marker='x', label=f"t={t}", s=20, c=[cmap(t / T)])
        if p2 is not None:
            t = round(float(Ts[i]), 1)
            plt.scatter(list(X), list(p2[i]), s=20, label=f"t={t}", c=[cmap(t / T)])

    # Set labels, title, and color bar
    plt.xlabel('$x$')
    plt.ylabel('$p(t,x)$')
    plt.title(f'Probability density - ' + title)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'Time  $t$')

    # Add legends if the second distribution is provided
    if p2 is not None:
        legends = {
            r'$p(t,x)$':  plt.Line2D([0], [0], marker='o', color='black', lw=0, markersize=5),
            r'$\hat{p}(t,x)$': plt.Line2D([0], [0], marker='x', color='black', lw=0, markersize=5)
        }
        plt.legend(legends.values(), legends.keys())

    plt.setp(ax, xticks=[0.0, 0.5, 1.0, 1.5], xticklabels=[0.0, 0.5, 1.0, 1.5], yticks=[0, 0.4, 0.8, 1.2])
    plt.savefig(f'../plots/test_kde_plot/{name}.pdf')
