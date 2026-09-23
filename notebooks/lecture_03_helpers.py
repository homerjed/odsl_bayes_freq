"""Supplied posterior integration, HPD and plotting helpers for lecture three.

The numerical work is supplied so the exercises can focus on the models and
probability statements. Inference grids depend on the observed data, never on
the generating truth. Enable JAX 64-bit arithmetic in the notebook setup.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def posterior_grid(data, log_likelihood, prior_bounds=(-10., 10.), grid_size=4001):
    """Normalise a likelihood under a uniform prior using adaptive quadrature.

    ``log_likelihood(alpha, data)`` must support broadcasting: alpha has shape
    (..., grid_size), and data has shape (..., 1, 2). Search the whole prior
    first, retaining all modes within 40 log units of the maximum, then resolve
    that span finely. The omitted tails are negligible for the lecture models.
    Data may be a single pair or a batch of pairs.
    """
    data = jnp.asarray(data)
    coarse = jnp.linspace(*prior_bounds, 5001)
    log_density = log_likelihood(coarse, data[..., None, :])
    keep = log_density >= jnp.max(log_density, axis=-1, keepdims=True) - 40
    padding = 2 * (coarse[1] - coarse[0])
    lower = jnp.maximum(jnp.min(jnp.where(keep, coarse, jnp.inf), axis=-1)
                        - padding, prior_bounds[0])
    upper = jnp.minimum(jnp.max(jnp.where(keep, coarse, -jnp.inf), axis=-1)
                        + padding, prior_bounds[1])
    grid = lower[..., None] + (upper - lower)[..., None] * jnp.linspace(0, 1, grid_size)
    log_density = log_likelihood(grid, data[..., None, :])
    density = jnp.exp(log_density - jnp.max(log_density, axis=-1, keepdims=True))
    density /= jnp.trapezoid(density, grid, axis=-1)[..., None]
    return grid, density


def _mass_above(grid, density, threshold):
    """Integrate the piecewise-linear density wherever it exceeds threshold."""
    left, right = density[..., :-1], density[..., 1:]
    low, high = jnp.minimum(left, right), jnp.maximum(left, right)
    threshold = jnp.asarray(threshold)[..., None]
    # A crossing cuts a trapezoid down to a smaller trapezoid at its high end.
    fraction = jnp.clip((high - threshold) / jnp.where(high > low, high - low, 1), 0, 1)
    area = jnp.where(threshold <= low, (low + high) / 2,
                     fraction * (high + threshold) / 2)
    return jnp.sum(jnp.diff(grid, axis=-1) * area, axis=-1)


def hpd_region(grid, density, level=0.683):
    """Return (grid mask, density threshold, enclosed mass), keeping all pieces."""
    def bisect(_, bounds):
        lower, upper = bounds
        middle = (lower + upper) / 2
        too_much_mass = _mass_above(grid, density, middle) > level
        return (jnp.where(too_much_mass, middle, lower),
                jnp.where(too_much_mass, upper, middle))

    lower, upper = jax.lax.fori_loop(0, 40, bisect, (jnp.array(0.), jnp.max(density)))
    threshold = (lower + upper) / 2
    return density >= threshold, threshold, _mass_above(grid, density, threshold)


def hpd_contains(data, truths, log_likelihood, level=0.683,
                 prior_bounds=(-10., 10.), grid_size=4001, batch_size=128):
    """Test one truth per dataset without constructing or joining HPD intervals.

    A truth belongs to the HPD region when the posterior mass at higher density
    is at most ``level``. Truths are used only for this final membership query.
    Batches keep memory use modest when repeating thousands of experiments.
    """
    data = jnp.asarray(data)
    truths = jnp.broadcast_to(jnp.asarray(truths), data.shape[:-1])

    @jax.jit
    def contains_batch(batch, targets):
        grid, density = posterior_grid(batch, log_likelihood, prior_bounds, grid_size)
        target_density = jax.vmap(jnp.interp)(targets, grid, density)
        in_grid = (targets >= grid[:, 0]) & (targets <= grid[:, -1])
        return in_grid & (_mass_above(grid, density, target_density) <= level)

    return jnp.concatenate([
        contains_batch(data[start:start + batch_size], truths[start:start + batch_size])
        for start in range(0, len(data), batch_size)
    ])


def plot_analysis(data, mean_model, grid, density, mask, truth, level=0.683):
    """Show the mean relation, observation and posterior with its HPD region."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    curve = mean_model(jnp.linspace(-1.5, 1.5, 300))
    axes[0].plot(curve[:, 0], curve[:, 1], label="Mean model")
    axes[0].scatter(data[0], data[1], label="Observed data")
    axes[0].set(xlabel="d1", ylabel="d2")
    axes[0].legend()
    axes[1].plot(grid, density, label="Posterior")
    axes[1].fill_between(grid, 0, density, where=mask, interpolate=True,
                         alpha=0.3, label=f"{100 * level:g}% HPD region")
    axes[1].axvline(truth, linestyle="--", label="Generating truth")
    axes[1].set(xlabel="alpha", ylabel="Posterior density")
    axes[1].legend()
    plt.show()


def plot_coverage(results, level=0.683):
    """Compare coverage estimates with one Monte Carlo standard error."""
    labels = [label.replace(": ", "\n") for label in results]
    coverage, error = zip(*results.values())
    fig, ax = plt.subplots()
    ax.errorbar(range(len(labels)), coverage, yerr=error, fmt="o")
    ax.axhline(level, linestyle="--", label=f"Nominal {100 * level:g}%")
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Coverage")
    ax.legend()
    plt.show()
