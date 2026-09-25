"""Inference and plotting helpers for the combined lectures 4-5 tutorial only.

The demo uses the exact Gaussian posterior and supplied plotting functions.
Other tutorials have their own lecture-specific helper modules.

The optional sampler is a preconditioned random-walk Metropolis--Hastings chain
for smooth Gaussian posteriors. It is not a general-purpose MCMC package:
its fixed curvature estimate assumes one approximately Gaussian mode.
Enable JAX 64-bit arithmetic in the notebook setup for the analytic comparisons.
"""

import operator

import jax
import jax.numpy as jnp
from jax import random


def _positive_definite(matrix, name):
    """Validate a covariance or precision matrix before sampling or solving."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError(f"{name} must have finite entries.")
    if not bool(jnp.allclose(matrix, matrix.T)):
        raise ValueError(f"{name} must be symmetric.")
    matrix = (matrix + matrix.T) / 2
    if not bool(jnp.all(jnp.linalg.eigvalsh(matrix) > 0)):
        raise ValueError(f"{name} must be positive definite.")
    return matrix


def gaussian_posterior(data_mean, n, Sigma, prior_mean, prior_cov):
    """Return the exact (mean, covariance) for a Gaussian mean and prior.

    Observations are independent N(mu, Sigma); the prior is
    N(prior_mean, prior_cov). All vectors have shape (dimension,) and all
    matrices (dimension, dimension). Sigma is the covariance of ONE observation,
    so the likelihood precision for the sample mean is n * inverse(Sigma).
    """
    data_mean = jnp.asarray(data_mean, dtype=float)
    prior_mean = jnp.asarray(prior_mean, dtype=float)
    if data_mean.ndim != 1 or prior_mean.shape != data_mean.shape:
        raise ValueError("data_mean and prior_mean must be equal-length vectors.")
    if not bool(jnp.all(jnp.isfinite(data_mean)) &
                jnp.all(jnp.isfinite(prior_mean))):
        raise ValueError("The data and prior means must have finite entries.")
    if not bool(jnp.isfinite(n)) or n <= 0:
        raise ValueError("n must be positive and finite.")
    Sigma = _positive_definite(jnp.asarray(Sigma, dtype=float), "Sigma")
    prior_cov = _positive_definite(jnp.asarray(prior_cov, dtype=float), "prior_cov")
    identity = jnp.eye(data_mean.size)
    if Sigma.shape != identity.shape or prior_cov.shape != identity.shape:
        raise ValueError("Covariance shapes must match the mean vectors.")
    data_precision = n * jnp.linalg.solve(Sigma, identity)
    prior_precision = jnp.linalg.solve(prior_cov, identity)
    precision = data_precision + prior_precision
    covariance = jnp.linalg.solve(precision, identity)
    mean = jnp.linalg.solve(
        precision, data_precision @ data_mean + prior_precision @ prior_mean)
    return mean, (covariance + covariance.T) / 2


def sample_posterior(draw_key, log_posterior, initial, *, num_samples=20_000,
                     warmup=1_000, return_diagnostics=False):
    """Draw a Metropolis--Hastings chain for the tutorial's Gaussian posterior.

    ``log_posterior(theta)`` must be a JAX-differentiable scalar function; an
    additive normalisation constant is unnecessary. ``initial`` is a finite
    vector. The returned JAX array has shape ``(num_samples, dimension)``: rows
    are retained, serially correlated parameter draws after warm-up. Reusing
    the same key, target and settings reproduces exactly the same chain.

    A symmetric Gaussian random-walk proposal uses the inverse negative Hessian
    at ``initial`` as its covariance, scaled by ``2.38**2 / dimension``. This
    curvature is constant for all Gaussian priors used in the tutorial, so no
    student tuning is needed. The Hessian must be finite and positive definite.
    Warm-up discards initial transients; it does not adapt the proposal. This
    choice is intended for these Gaussian targets, not arbitrary multimodal,
    bounded or heavy-tailed posteriors. Initial states extremely far into the
    tails can require more warm-up.

    With ``return_diagnostics=True``, return ``(draws, diagnostics)`` instead;
    diagnostics contains the retained-chain acceptance rate and proposal scale.
    Acceptance alone is not a convergence check. Instructor validation compares
    repeated chains with the exact posterior, accounting for autocorrelation.
    """
    num_samples, warmup = operator.index(num_samples), operator.index(warmup)
    if num_samples < 1 or warmup < 0:
        raise ValueError("num_samples must be positive and warmup nonnegative.")
    initial = jnp.asarray(initial, dtype=float)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("initial must be a nonempty parameter vector.")
    if not bool(jnp.all(jnp.isfinite(initial))):
        raise ValueError("initial must have finite entries.")
    initial_log_density = jnp.asarray(log_posterior(initial))
    if initial_log_density.ndim != 0 or not bool(jnp.isfinite(initial_log_density)):
        raise ValueError("log_posterior(initial) must be a finite scalar.")

    precision = _positive_definite(
        -jax.hessian(log_posterior)(initial), "Negative log-posterior Hessian")
    # If H = L L.T, then L^{-T} z has covariance H^{-1} for z ~ N(0, I).
    proposal_scale = 2.38 / jnp.sqrt(initial.size)
    proposal_root = proposal_scale * jnp.linalg.solve(
        jnp.linalg.cholesky(precision).T, jnp.eye(initial.size))

    def step(state, _):
        position, log_density, key = state
        key, proposal_key, accept_key = random.split(key, 3)
        proposal = position + proposal_root @ random.normal(
            proposal_key, position.shape, dtype=position.dtype)
        proposed_log_density = log_posterior(proposal)
        log_ratio = proposed_log_density - log_density
        accept = (jnp.isfinite(proposed_log_density) &
                  (jnp.log(random.uniform(accept_key)) < log_ratio))
        position = jnp.where(accept, proposal, position)
        log_density = jnp.where(accept, proposed_log_density, log_density)
        return (position, log_density, key), (position, accept)

    @jax.jit
    def run(key):
        state = (initial, initial_log_density, key)
        state, _ = jax.lax.scan(step, state, xs=None, length=warmup)
        _, (draws, accepted) = jax.lax.scan(
            step, state, xs=None, length=num_samples)
        return draws, jnp.mean(accepted)

    draws, acceptance_rate = run(draw_key)
    if return_diagnostics:
        return draws, {"acceptance_rate": acceptance_rate,
                       "proposal_scale": proposal_scale}
    return draws


# Supplied plots: keep the exercises focused on statistical calculations.
import matplotlib.pyplot as plt


def plot_scalar_observations(values, estimate, interval):
    """Distinguish individual observations from uncertainty on their mean."""
    fig, ax = plt.subplots(figsize=(7, 2.5), layout="constrained")
    ax.scatter(values, jnp.zeros_like(values))
    errors = jnp.array([[estimate - interval[0]], [interval[1] - estimate]])
    ax.errorbar([estimate], [1], xerr=errors, fmt="o", capsize=4)
    ax.set(xlabel=r"$x_1$ or $\mu_1$", yticks=[0, 1],
           yticklabels=["Observations", "Mean and 95% CI"], ylim=(-0.5, 1.5))
    plt.show()


def plot_confidence_intervals(estimates, lower, upper, truth):
    """Show a small subset of repeated confidence intervals."""
    hits = (lower <= truth) & (truth <= upper)
    rows = jnp.arange(len(estimates))
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for mask, label in ((hits, "Contains truth"), (~hits, "Misses truth")):
        ax.errorbar(estimates[mask], rows[mask],
                    xerr=jnp.stack((estimates[mask] - lower[mask], upper[mask] - estimates[mask])),
                    fmt="o", markersize=3, label=label)
    ax.axvline(float(truth), linestyle="--", label="Fixed truth")
    ax.set(xlabel=r"$\mu_1$", ylabel="Repeated experiment")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def _ellipse(mean, covariance, level):
    angle = jnp.linspace(0, 2 * jnp.pi, 300)
    circle = jnp.stack((jnp.cos(angle), jnp.sin(angle)))
    return mean[:, None] + jnp.sqrt(-2 * jnp.log1p(-level)) * jnp.linalg.cholesky(covariance) @ circle


def plot_regions(regions, *, truth=None, samples=None, level=0.95):
    """Compare supplied two-dimensional Gaussian region boundaries."""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.2, label="Posterior draws")
    for label, (mean, covariance) in regions.items():
        boundary = _ellipse(mean, covariance, level)
        ax.plot(boundary[0], boundary[1], label=label)
    if truth is not None:
        ax.scatter(truth[0], truth[1], marker="*", s=100, label="Generating truth")
    ax.set(xlabel=r"$\mu_1$", ylabel=r"$\mu_2$", aspect="equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def plot_chi_squared(statistics, *, level=0.95):
    """Compare repeated 2D mean-distance statistics with their exact reference."""
    cutoff = -2 * jnp.log1p(-level)
    grid = jnp.linspace(0, max(15., float(jnp.max(statistics))), 400)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.hist(statistics, bins=40, density=True, alpha=0.5, label="Repeated experiments")
    ax.plot(grid, 0.5 * jnp.exp(-grid / 2), label=r"$\chi^2_2$ density")
    ax.axvline(float(cutoff), linestyle="--", label="95% cutoff")
    ax.set(xlabel=r"$\Delta\chi^2$ at the true mean", ylabel="Density")
    ax.legend()
    plt.show()


def plot_coverage_comparison(results, *, level=0.95):
    """Coverage fractions with two Monte Carlo standard errors."""
    labels = list(results)
    fractions = jnp.array([float(value[0]) for value in results.values()])
    errors = jnp.array([2 * float(value[1]) for value in results.values()])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.errorbar(fractions, jnp.arange(len(labels)), xerr=errors, fmt="o", capsize=3)
    ax.axvline(level, linestyle="--", label="Nominal 95%")
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlim=(-0.03, 1.03),
           xlabel="Coverage (bars: two Monte Carlo SE)")
    ax.invert_yaxis()
    ax.legend(loc="lower center")
    plt.show()
