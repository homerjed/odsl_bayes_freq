"""Supplied displays for lecture one; the notebooks contain the exercises."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.stats import norm, uniform


def plot_distributions(samples, mean=2.0, variance=1.0):
    """Compare sample densities with the three supplied population PDFs."""
    if any(values is None for values in samples.values()):
        print("Complete the three draws in draw_distributions, then rerun this cell.")
        return

    sd = jnp.sqrt(variance)
    lower, upper = mean - jnp.sqrt(3) * sd, mean + jnp.sqrt(3) * sd
    log_variance = jnp.log1p(variance / mean**2)
    log_sd = jnp.sqrt(log_variance)
    log_mean = jnp.log(mean) - log_variance / 2

    # Shared bins include every observation, including the lognormal tail.
    all_samples = jnp.concatenate(list(samples.values()))
    bins = jnp.histogram_bin_edges(all_samples, bins=30)
    x = jnp.linspace(bins[0], bins[-1], 500)
    positive_x = jnp.maximum(x, 1e-12)
    pdfs = {
        "Gaussian": norm.pdf(x, loc=mean, scale=sd),
        "Uniform": uniform.pdf(x, loc=lower, scale=upper - lower),
        "Lognormal": jnp.where(
            x > 0,
            norm.pdf(jnp.log(positive_x), loc=log_mean, scale=log_sd) / positive_x,
            0.0,
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (name, values) in zip(axes, samples.items()):
        ax.hist(values, bins=bins, density=True, alpha=0.5, label="Sample")
        ax.plot(x, pdfs[name], label="PDF")
        ax.set(title=name, xlabel="x")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    plt.show()


def report_moments(samples, moment_function):
    """Format the student's scalar summaries without computing them here."""
    if any(values is None for values in samples.values()):
        print("Complete draw_distributions before comparing sample moments.")
        return

    results = {name: moment_function(values) for name, values in samples.items()}

    if any(mean is None or variance is None for mean, variance in results.values()):
        print("Complete sample_moments, then rerun this cell.")
        return

    n = len(next(iter(samples.values())))
    print(f"n = {n} observations per distribution")
    print(f"{'Distribution':<14} {'Sample mean':>13} {'Sample variance':>17}")

    for name, (mean, variance) in results.items():
        print(f"{name:<14} {float(mean):13.3f} {float(variance):17.3f}")
    print("Population values: mean = 2; variance = 1.")


def show_gaussian_sample(points, population_mean, population_covariance,
                         estimated_mean, estimated_covariance, *, plot=True):
    """Display the student's 2D estimates and a simple data scatter plot."""
    if points is None:
        print("Complete draw_gaussian_2d, then rerun this cell.")
        return
    if estimated_mean is None or estimated_covariance is None:
        print("Complete sample_mean_covariance, then rerun this cell.")
        return
    rho = population_covariance[0, 1] / jnp.sqrt(
        population_covariance[0, 0] * population_covariance[1, 1]
    )
    print(f"n = {len(points)}; population correlation = {float(rho):.2f}")
    print("Population mean:", population_mean)
    print("Sample mean:    ", jnp.round(estimated_mean, 3))
    print("Population covariance:\n", population_covariance)
    print("Sample covariance:\n", jnp.round(estimated_covariance, 3))
    if not plot:
        return

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1])
    # Keep the same coordinate scale for every correlation and sample size.
    radius = 4 * float(jnp.sqrt(jnp.diag(population_covariance)).max())
    ax.set(xlabel="x1", ylabel="x2", aspect="equal",
           xlim=(float(population_mean[0]) - radius, float(population_mean[0]) + radius),
           ylim=(float(population_mean[1]) - radius, float(population_mean[1]) + radius))
    plt.show()
