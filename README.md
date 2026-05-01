# Functional-Liu
Generalized Liu-type estimator, specialized for functional data analysis

After discretizing, we assume a model $y = X\beta + \text{noise}$ and the following estimators

- Ordinary least squares
$$\hat{\beta}_\text{OLS} = (X^\top X)^{-1} X^\top y$$

- Ridge regression (for $\lambda \ge 0$)
$$\hat{\beta}_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$$

- generalized ridge regression: create the spsd roughness matrix $R \succeq 0$ to penalize large values of curvature (i.e., specialized for data arising from **functional data analysis**), 
create the matrix $Q = \alpha I + (1-\alpha)R$
and then do generalized ridge regression
$$\hat{\beta}_{\lambda,\alpha} = (X^\top X + \lambda Q)^{-1} X^\top y$$

- Liu's biased estimator, for $0\le d \le 1$,
$$\hat{\beta}_{\lambda,d} = (X^\top X + \lambda I)^{-1}(X^\top y + d\lambda \hat{\beta}_0)$$
where $\hat{\beta}_0$ is the OLS estimator.

- and we introduce the **functional Liu estimator**, which combines the generalized ridge estimator with Liu's estimator,
$$\hat{\beta}_{\lambda,\alpha, d} = (X^\top X + \lambda Q)^{-1}(X^\top y + d\lambda Q\hat{\beta}_0)$$
where $\hat{\beta}_0$ is the OLS estimator.

This code repository consists of:
- A Python package, `fliu.py`, which implements these estimators as well as an efficient method to find the optimal parameter values
  - Specifically, it searches for parameter values that optimize the **generalized cross validation** (GCV) criterion. It does this using a gradient-based optimization method from `scipy.optimize.minimize`, and uses the automatic differentiation capabilities of `jax` to find the gradient
- Scripts in the R language to setup relevant statistical examples


## Authors
Shaista Ashraf and Stephen Becker
