"""Supplied repetition and plotting helpers for lecture two."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random


def repeat_estimator(draw_key, n, repetitions, estimator):
    """Repeat the student's estimator with an independent key for each sample."""
    keys = random.split(draw_key, repetitions)
    return jax.vmap(lambda key: estimator(key, n)[2])(keys)


def plot_circle(points, inside):
    """Show the sampled points and the unit circle inside the square."""
    fig, ax = plt.subplots()
    ax.scatter(points[inside, 0], points[inside, 1], label="Inside")
    ax.scatter(points[~inside, 0], points[~inside, 1], label="Outside")
    angle = jnp.linspace(0, 2 * jnp.pi, 200)
    ax.plot(jnp.cos(angle), jnp.sin(angle), label="Unit circle")
    ax.set(xlabel="x", ylabel="y", xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
    ax.legend()
    plt.show()


def plot_sampling_distributions(samples, truth, xlabel):
    """Compare repeated estimates using the same histogram bins."""
    bins = jnp.histogram_bin_edges(jnp.concatenate(list(samples.values())), bins=30)
    fig, ax = plt.subplots()
    for label, values in samples.items():
        ax.hist(values, bins=bins, density=True, histtype="step", label=label)
    ax.axvline(truth, label="True value")
    ax.set(xlabel=xlabel, ylabel="Density")
    ax.legend()
    plt.show()


def plot_likelihoods(observation, sigma, rho=0.0):
    """Compare relative likelihood contours from x1 alone and both measurements."""
    x1, x2 = observation
    centre_theta, centre_phi = x1 - x2, x2
    theta = jnp.linspace(centre_theta - 4 * sigma, centre_theta + 4 * sigma, 201)
    phi = jnp.linspace(centre_phi - 4 * sigma, centre_phi + 4 * sigma, 201)
    theta_grid, phi_grid = jnp.meshgrid(theta, phi)
    r1 = x1 - theta_grid - phi_grid
    r2 = x2 - phi_grid
    marginal = jnp.exp(-r1**2 / (2 * sigma**2))
    joint = jnp.exp(
        -(r1**2 - 2 * rho * r1 * r2 + r2**2) / (2 * sigma**2 * (1 - rho**2))
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True,
                             layout="constrained")
    for ax, values, title in zip(axes, (marginal, joint), ("x1 alone", "x1 and x2")):
        ax.contour(theta_grid, phi_grid, values, levels=[0.1, 0.4, 0.7, 0.9])
        ax.set(xlabel="theta", ylabel="phi", title=title, aspect="equal")
    plt.show()
