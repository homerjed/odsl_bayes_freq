"""Supplied posterior sampler and plots for the lecture six tutorial."""

import operator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METHOD_COLORS = {'True Gaussian': 'C0', 'Hartlap Gaussian': 'C1', 'Percival 2021': 'C2'}
PARAMETER_LABELS = (r'Intercept $\theta_1$', r'Slope $\theta_2$')


def _metropolis(log_density, initial, proposal_cov, rng, n_steps=18000, burn_in=2000):
    """Basic symmetric random-walk Metropolis; retain rejected states."""
    if not 0 <= burn_in < n_steps:
        raise ValueError('Require 0 <= burn_in < n_steps.')
    position = np.array(initial, dtype=float, copy=True)
    log_p = log_density(position)
    if not np.isfinite(log_p):
        raise ValueError('Initial log density must be finite.')
    proposal_root = np.linalg.cholesky(proposal_cov)
    chain = np.empty((n_steps, len(position)))
    accepted = 0
    for step in range(n_steps):
        proposal = position + proposal_root @ rng.standard_normal(len(position))
        proposed_log_p = log_density(proposal)
        if np.isfinite(proposed_log_p) and np.log(rng.uniform()) < proposed_log_p - log_p:
            position, log_p = proposal, proposed_log_p
            accepted += (step >= burn_in)
        chain[step] = position
    return chain[burn_in:], accepted / (n_steps - burn_in)


def sample_posterior(model, *, seed, stream=10, n_samples=16000):
    """Return parameter draws from one of this tutorial's posterior models.

    The model supplies log_density, mean and scale. The helper handles the
    initial state, proposal covariance and burn-in. The returned array has
    shape (n_samples, 2). Draws are correlated; this is a basic Metropolis
    implementation for these unimodal targets. Reusing seed and stream
    reproduces the draws, while distinct streams separate the three methods.
    """
    n_samples = operator.index(n_samples)
    if n_samples < 1:
        raise ValueError('n_samples must be positive.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    draws, _ = _metropolis(model['log_density'], model['mean'],
                           1.5**2 * model['scale'], rng,
                           n_steps=n_samples + 2000, burn_in=2000)
    return draws


def _ellipse(mean, covariance, radius_squared=1.0):
    angle = np.linspace(0, 2 * np.pi, 300)
    circle = np.array([np.cos(angle), np.sin(angle)])
    return (np.asarray(mean)[:, None]
            + np.sqrt(radius_squared) * np.linalg.cholesky(covariance) @ circle)


def _posterior_boundary(model, level):
    if not 0 < level < 1:
        raise ValueError('level must lie between zero and one.')
    nu = model['df']
    radius = (-2 * np.log1p(-level) if np.isinf(nu)
              else nu * np.expm1(-2 * np.log1p(-level) / nu))
    return _ellipse(model['mean'], model['scale'], radius)


def _limits(point_sets):
    points = np.concatenate([np.asarray(points).reshape(-1, 2) for points in point_sets])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = 0.08 * np.maximum(high - low, 0.01)
    return (low[0] - padding[0], high[0] + padding[0]), (low[1] - padding[1], high[1] + padding[1])


def _parameter_axes(ax, limits):
    # Square panels with consistent limits; the parameters can have different units.
    ax.set(xlabel=PARAMETER_LABELS[0], ylabel=PARAMETER_LABELS[1],
           xlim=limits[0], ylim=limits[1])
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=9)
    ax.locator_params(axis='both', nbins=5)


def _legend(fig, handles, labels, ncol=3):
    fig.legend(handles, labels, loc='outside lower center', ncol=ncol,
               frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.5)


def _truth(ax, truth):
    ax.scatter(*truth, marker='*', s=75, color='black', zorder=5)


def plot_covariances(C, S):
    """True and sample data covariances with one shared color scale."""
    C, S = np.asarray(C), np.asarray(S)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or S.shape != C.shape:
        raise ValueError('C and S must be square matrices of the same shape.')
    limit = max(np.abs(C).max(), np.abs(S).max())
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), layout='constrained')
    for ax, matrix, title in zip(axes, (C, S), ('True covariance', 'Estimated covariance')):
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(title=title, xlabel='Data component')
        ax.set_xticks([0, 3, 6, 9])
        ax.set_yticks([0, 3, 6, 9])
    axes[0].set_ylabel('Data component')
    fig.colorbar(image, ax=axes, label='Covariance', shrink=0.85, pad=0.03)
    return fig


def plot_mle_scatter(estimates, truth, covariances, title=''):
    """MLEs and covariance ellipses at squared Mahalanobis radius one."""
    estimates = np.asarray(estimates)
    fig, ax = plt.subplots(figsize=(5.4, 4.8), layout='constrained')
    ax.scatter(*estimates.T, s=5, color='0.55', alpha=0.20, rasterized=True)
    styles = {'Empirical covariance': ('0.15', '--'),
              'Inverse Fisher': ('C0', '-'),
              'Dodelson–Schneider': ('C1', '-'),
              'Exact covariance at fixed S': ('C2', ':')}
    handles, labels = [], []
    for label, covariance in covariances.items():
        boundary = _ellipse(truth, covariance)
        color, linestyle = styles.get(label, ('C2', '-'))
        line, = ax.plot(*boundary, color=color, ls=linestyle, lw=2, zorder=3)
        handles.append(line)
        labels.append(label.replace('Exact covariance at fixed S', 'Fixed-S prediction'))
    _truth(ax, truth)
    handles.append(Line2D([], [], color='black', marker='*', ls='', markersize=8))
    labels.append('Truth')
    _parameter_axes(ax, _limits([estimates, np.asarray(truth)[None, :]]))
    ax.set_title(title, fontsize=11)
    _legend(fig, handles, labels, ncol=2)
    return fig


def plot_variance_comparison(mock_counts, empirical_covariances, variance_se,
                            ideal_covariance, ds_factors, exact_factors=None):
    """MLE variances and theoretical predictions, with two-SE error bars."""
    mock_counts = np.asarray(mock_counts)
    covariances, errors = np.asarray(empirical_covariances), np.asarray(variance_se)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True, layout='constrained')
    for parameter, ax in enumerate(axes):
        baseline = ideal_covariance[parameter, parameter]
        ax.errorbar(mock_counts, covariances[:, parameter, parameter] / baseline,
                    yerr=2 * errors[:, parameter] / baseline, fmt='o', ms=4,
                    capsize=3, color='0.2', label='MLE variance')
        ax.axhline(1, color='C0', ls='--', lw=1.5, label='Inverse Fisher')
        ax.plot(mock_counts, ds_factors, color='C1', lw=2, label='Dodelson–Schneider')
        if exact_factors is not None:
            ax.plot(mock_counts, exact_factors, color='C2', ls=':', lw=2,
                    label='Exact linear result')
        ax.set(xlabel=r'Covariance mocks $n_s$', title=PARAMETER_LABELS[parameter], xscale='log')
        ax.set_xticks(mock_counts, [str(count) for count in mock_counts])
        ax.minorticks_off()
        ax.grid(axis='y', color='0.92', lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r'Variance / $(F^{-1})_{ii}$')
    handles, labels = axes[0].get_legend_handles_labels()
    _legend(fig, handles, labels, ncol=2)
    return fig


def plot_data(t, x, C, J, truth, posteriors):
    """Observed data and the two distinct best-fit lines."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout='constrained')
    ax.errorbar(t, x, yerr=np.sqrt(np.diag(C)), fmt='o', ms=4, color='0.5',
                elinewidth=1, capsize=2, label='Observed data')
    ax.plot(t, J @ truth, color='black', ls=':', lw=1.5, label='True mean')
    for label in ('True Gaussian', 'Hartlap Gaussian'):
        ax.plot(t, J @ posteriors[label]['mean'], color=METHOD_COLORS[label],
                lw=1.8, label=f'{label} fit')
    ax.set(xlabel='Measurement position t', ylabel='Observed value x')
    handles, labels = ax.get_legend_handles_labels()
    _legend(fig, handles, labels, ncol=2)
    return fig


def plot_posteriors(posteriors, truth, *, samples=None, level=0.95, ax=None, title=''):
    """Compare posteriors on aligned axes, separating clouds into panels."""
    boundaries = {label: _posterior_boundary(model, level)
                  for label, model in posteriors.items()}
    shown = {} if samples is None else {
        label: np.asarray(draws)[::max(1, len(draws) // 450)]
        for label, draws in samples.items()}
    limits = _limits([line.T for line in boundaries.values()]
                     + list(shown.values()) + [np.asarray(truth)[None, :]])
    if ax is not None or samples is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.4, 4.8), layout='constrained')
            own_figure = True
        else:
            fig, own_figure = ax.figure, False
        for label, boundary in boundaries.items():
            ax.plot(*boundary, color=METHOD_COLORS[label], lw=2, label=label)
        _truth(ax, truth)
        _parameter_axes(ax, limits)
        ax.set_title(title, fontsize=11)
        if own_figure:
            handles, labels = ax.get_legend_handles_labels()
            _legend(fig, handles, labels)
        return fig
    fig, axes = plt.subplots(1, len(posteriors), figsize=(9.4, 3.5),
                              sharex=True, sharey=True, layout='constrained', squeeze=False)
    for index, (label, boundary) in enumerate(boundaries.items()):
        axis = axes[0, index]
        color = METHOD_COLORS[label]
        if label in shown:
            axis.scatter(*shown[label].T, s=5, alpha=0.15, color=color, rasterized=True)
        axis.plot(*boundary, color=color, lw=2)
        _truth(axis, truth)
        _parameter_axes(axis, limits)
        axis.set_title(label, fontsize=11)
        if index:
            axis.set_ylabel('')
    handles = [Line2D([], [], color='0.5', marker='.', ls='', markersize=6),
               Line2D([], [], color='0.3', lw=2),
               Line2D([], [], color='black', marker='*', ls='', markersize=8)]
    _legend(fig, handles, ['Posterior draws', f'{100 * level:g}% joint region', 'Truth'])
    if title:
        fig.suptitle(title, fontsize=11)
    return fig


def plot_posterior_grid(cases, truth, *, seeds, mock_counts, level=0.95):
    """Align multiple observations and mock budgets with one shared legend."""
    limits = _limits([_posterior_boundary(model, level).T
                      for row in cases for models in row for model in models.values()]
                     + [np.asarray(truth)[None, :]])
    fig, axes = plt.subplots(len(mock_counts), len(seeds), figsize=(9.4, 6.5),
                              sharex=True, sharey=True, layout='constrained', squeeze=False)
    for row, count in enumerate(mock_counts):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            for label, model in cases[row][col].items():
                ax.plot(*_posterior_boundary(model, level), color=METHOD_COLORS[label], lw=1.8)
            _truth(ax, truth)
            _parameter_axes(ax, limits)
            if row == 0:
                ax.set_title(f'Data seed {seed}', fontsize=11)
            ax.set_ylabel(f'{count} mocks\n{PARAMETER_LABELS[1]}' if col == 0 else '')
            ax.set_xlabel(PARAMETER_LABELS[0] if row == len(mock_counts) - 1 else '')
    handles = [Line2D([], [], color=color, lw=2) for color in METHOD_COLORS.values()]
    handles.append(Line2D([], [], color='black', marker='*', ls='', markersize=8))
    _legend(fig, handles, [*METHOD_COLORS, 'Truth'], ncol=4)
    return fig


def plot_coverage(coverage_results, level=0.95):
    """Fixed- and fresh-covariance coverage in aligned panels, with two-SE bars."""
    groups = {'Fixed S': {}, 'Fresh S': {}}
    low, high = level, level
    for category, results in coverage_results.items():
        scheme, count = category.split(', n_s=')
        groups[scheme][int(count)] = results
        for fraction, se in results.values():
            low, high = min(low, fraction - 2 * se), max(high, fraction + 2 * se)
    padding = max(0.015, 0.12 * (high - low))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True, layout='constrained')
    titles = ('Fixed covariance estimate', 'Fresh covariance estimates')
    for ax, (scheme, counts), title in zip(axes, groups.items(), titles):
        mock_counts = sorted(counts)
        for index, (method, color) in enumerate(METHOD_COLORS.items()):
            fractions, errors = zip(*[counts[count][method] for count in mock_counts])
            ax.errorbar(np.arange(len(mock_counts)) + (index - 1) * 0.16, fractions,
                        yerr=2 * np.asarray(errors), fmt='o', ms=4, capsize=3,
                        color=color, label=method)
        ax.axhline(level, color='0.3', ls='--', lw=1.2, label=f'Nominal {100 * level:g}%')
        ax.set_xticks(np.arange(len(mock_counts)), mock_counts)
        ax.set(xlabel=r'Covariance mocks $n_s$', title=title,
               xlim=(-0.5, len(mock_counts) - 0.5),
               ylim=(max(0, low - padding), min(1, high + padding)))
        ax.grid(axis='y', color='0.92', lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Coverage')
    handles, labels = axes[0].get_legend_handles_labels()
    _legend(fig, handles, labels, ncol=2)
    return fig
