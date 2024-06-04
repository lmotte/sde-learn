# sde-learn

This repository provides a Python implementation of the method proposed in [Bonalli et al., 2023](#references) for
estimating coefficients of multidimensional nonlinear stochastic differential equations (SDEs). Specifically, the method
divides the estimation process into two more manageable problems:

1. **Density estimation.** Estimating the probability density associated with the unknown SDE, and its derivatives,
   thanks to a data set of sample paths.
2. **Coefficients estimation with Fokker-Planck matching.** Estimating SDE coefficients (drift and diffusion) that match
   the Fokker-Planck dynamics described by the estimated density.

Please refer to [Bonalli et al., 2023](#references) for a detailed description of the method.

**Features.** This code implements the following features.

- **Kernel-based modeling.** Both the density estimation and Fokker-Planck matching steps employ kernel-based models for
  the probability density and the SDE coefficients, respectively.
- **Positivity constraints.** Enforcing positive-definiteness of the estimated diffusion coefficient is crucial for
  physical and mathematical validity. For this purpose, we consider uniform diffusion and add positivity constraints on
  a finite set of predetermined points.
- **Fast computations.** Our implementation includes the Nyström approximation to reduce computational complexity when
  handling large datasets. Additionally, for the Gaussian kernel, we implement fast computation of all partial
  derivatives and Gram matrices, employing vectorized computations in Python with NumPy.

## Examples

You can find several example scripts in the `examples` folder, each demonstrating different use cases of the `sde-learn`
library.

**Example descriptions.**

Below is a brief description of each example.

1. **example_ornstein_uhlenbeck_paths_plot.py**.
   Shows how to generate and plot sample paths from an Ornstein-Uhlenbeck process, illustrating the simulation of SDE
   paths.
2. **example_kde_plot.py**.
   Demonstrates how to use the `ProbaDensityEstimator` to estimate and plot the probability density of sample paths
   generated from an SDE.

3. **example_sde_identification_1d.py**.
   Provides a complete example for simulating a one-dimensional SDE, estimating its density with
   the `ProbaDensityEstimator` class, and then estimating its coefficients with Fokker-Planck matching using
   the `FPEstimator` class.

4. **example_sde_identification_2d_1.py**.
   Provides a complete example for simulating a two-dimensional nonlinear SDE, estimating its density with
   the `ProbaDensityEstimator` class, and then estimating its coefficients with Fokker-Planck matching using
   the `FPEstimator` class.

5. **example_sde_identification_2d_2.py**.
   Similar to `example_sde_identification_2d_1.py` but with a different two-dimensional nonlinear SDE.

**Running the examples.**

To run any of these examples, navigate to the `examples` folder and execute the desired script using Python. For
instance, to run `example_kde_plot.py`, use the following command.

```bash
python examples/example_kde_plot.py
```

## Installation

Clone this repository to your local machine.

```bash
git clone https://github.com/lmotte/sde-learn.git
```

Navigate into the project directory and install required dependencies. Ensure you have Python 3.x installed.

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Bonalli, Riccardo, and Alessandro Rudi. "Non-Parametric Learning of Stochastic Differential Equations with Fast Rates
  of Convergence." arXiv preprint [arXiv:2305.15557](https://arxiv.org/abs/2305.15557) (2023).