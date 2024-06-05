# sde-learn

## Overview

This repository provides a Python implementation of the method proposed by [Bonalli et al., 2023](#references), for
estimating the coefficients of multidimensional nonlinear stochastic differential equations (SDEs). This method divides
the estimation into two steps:

1. **Density estimation:** Estimation of the probability density associated with the unknown SDE and its derivatives
   using a dataset of sample paths.
2. **Coefficients estimation with Fokker-Planck matching:** Estimation of the SDE coefficients (drift and diffusion) by
   matching the Fokker-Planck dynamics described by the estimated density.

For a detailed description of the method, please refer to [Bonalli et al., 2023](#references).

**Features** This code implements the following features.

- **Kernel-based modeling:** Both steps employ kernel-based models for estimating the probability density and the SDE
  coefficients.
- **Positivity constraints:** To ensure physical and mathematical validity, it is crucial to enforce the
  positive-definiteness of the estimated diffusion coefficient. This is achieved by considering uniform diffusion and
  applying positivity constraints at a finite set of points.
- **Fast computations:** The code implements Nyström approximation to reduce computational complexity with large
  datasets. Additionally, for the Gaussian kernel, it enables fast computation of all partial derivatives and Gram
  matrices through vectorized computations in Python using NumPy.

## Examples

The `examples` folder contains several scripts demonstrating different applications of the `sde-learn` library.

1. **example_ornstein_uhlenbeck_paths_plot.py**: Illustrates the generation and plotting of sample paths from an
   Ornstein-Uhlenbeck process.
2. **example_kde_plot.py**: Demonstrates the use of the `ProbaDensityEstimator` for estimating and visualizing the
   probability density of sample paths from an SDE.
3. **example_sde_identification_1d.py**: Offers a complete example for simulating a one-dimensional SDE, estimating its
   density, and subsequently estimating its coefficients using Fokker-Planck matching.
4. **example_sde_identification_2d_1.py**: Provides a complete example for a two-dimensional nonlinear SDE, covering the
   estimation of its density and coefficients.
5. **example_sde_identification_2d_2.py**: Similar to the previous example, focusing on a different two-dimensional
   nonlinear SDE.

**Running the examples.** To run an example, navigate to the `examples` folder and execute the desired script. For
instance:

```bash
python examples/example_kde_plot.py
```


## Installation

To install:

1. Clone the repository.
   ```bash
   git clone https://github.com/lmotte/sde-learn.git
   ```
2. Navigate to the project directory and install the required dependencies (Python 3.x required).
   ```bash
   pip install -r requirements.txt
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Bonalli, Riccardo, and Alessandro Rudi. "Non-Parametric Learning of Stochastic Differential Equations with Fast Rates
  of Convergence." arXiv preprint [arXiv:2305.15557](https://arxiv.org/abs/2305.15557) (2023).