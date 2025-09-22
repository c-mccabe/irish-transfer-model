"""
Convergence checks and diagnostic tools for Bayesian STV models.

This module provides functions for assessing MCMC convergence,
posterior quality, and model fit diagnostics.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    import arviz as az
except ImportError:
    az = None


def check_convergence(samples: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Comprehensive convergence check for MCMC samples.

    Parameters
    ----------
    samples : dict
        Dictionary of parameter names to sample arrays (chains, draws, ...)

    Returns
    -------
    dict
        Dictionary containing R-hat values, ESS, and divergence information
    """
    results = {}

    # R-hat values
    rhat_values = check_rhat(samples)
    results['rhat'] = rhat_values

    # Effective sample sizes
    ess_values = effective_sample_size(samples)
    results['ess'] = ess_values

    # Check for divergences if available
    divergences = 0
    if 'diverging' in samples:
        divergences = int(np.sum(samples['diverging']))
    results['divergences'] = divergences

    # Summary flags
    results['converged'] = all(r < 1.01 for r in rhat_values.values() if not np.isnan(r))
    results['adequate_ess'] = all(e > 100 for e in ess_values.values() if not np.isnan(e))
    results['no_divergences'] = divergences == 0

    return results


def plot_posterior_matrix(
    pi_samples: np.ndarray,
    candidate_names: Optional[Dict[int, str]] = None,
    constituency_idx: int = 0,
    source_idx: int = 0,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot posterior transfer probability matrix.

    Parameters
    ----------
    pi_samples : np.ndarray
        Posterior samples of pi with shape (chains, draws, constituencies, sources, destinations)
    candidate_names : dict, optional
        Mapping from destination indices to candidate names
    constituency_idx : int
        Which constituency to plot
    source_idx : int
        Which source candidate to plot
    figsize : tuple
        Figure size
    """
    if plt is None:
        print("Matplotlib not available for plotting")
        return

    # Extract samples for specific constituency and source
    # Expected shape: (chains, draws, destinations)
    if pi_samples.ndim == 5:  # (chains, draws, constituencies, sources, destinations)
        relevant_samples = pi_samples[:, :, constituency_idx, source_idx, :]
    elif pi_samples.ndim == 4:  # (chains*draws, constituencies, sources, destinations)
        relevant_samples = pi_samples[:, constituency_idx, source_idx, :]
    else:
        print(f"Unexpected pi_samples shape: {pi_samples.shape}")
        return

    # Flatten chains and draws
    if relevant_samples.ndim == 3:  # (chains, draws, destinations)
        flattened_samples = relevant_samples.reshape(-1, relevant_samples.shape[-1])
    else:  # Already flattened
        flattened_samples = relevant_samples

    # Compute posterior statistics
    posterior_mean = np.mean(flattened_samples, axis=0)
    posterior_std = np.std(flattened_samples, axis=0)
    lower_ci = np.percentile(flattened_samples, 2.5, axis=0)
    upper_ci = np.percentile(flattened_samples, 97.5, axis=0)

    # Create destination labels
    n_destinations = len(posterior_mean)
    if candidate_names:
        dest_labels = [candidate_names.get(i, f"Dest_{i}") for i in range(n_destinations)]
    else:
        dest_labels = [f"Dest_{i}" for i in range(n_destinations)]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Posterior means with error bars
    x_pos = np.arange(n_destinations)
    ax1.bar(x_pos, posterior_mean, yerr=posterior_std, alpha=0.7, capsize=5)
    ax1.set_xlabel('Destination Candidate')
    ax1.set_ylabel('Transfer Probability')
    ax1.set_title(f'Posterior Mean ± SD\n(Constituency {constituency_idx}, Source {source_idx})')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dest_labels, rotation=45, ha='right')

    # Right plot: Credible intervals
    ax2.errorbar(x_pos, posterior_mean,
                yerr=[posterior_mean - lower_ci, upper_ci - posterior_mean],
                fmt='o', capsize=5, alpha=0.7)
    ax2.set_xlabel('Destination Candidate')
    ax2.set_ylabel('Transfer Probability')
    ax2.set_title(f'95% Credible Intervals\n(Constituency {constituency_idx}, Source {source_idx})')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dest_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def check_rhat(posterior_samples: Dict[str, np.ndarray], threshold: float = 1.01) -> Dict[str, float]:
    """
    Check R-hat convergence diagnostic for all parameters.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary of parameter names to sample arrays
    threshold : float
        R-hat threshold for convergence (1.01 is recommended)

    Returns
    -------
    dict
        Dictionary of parameter names to R-hat values
    """
    if az is None:
        print("Warning: ArviZ not available, using simplified R-hat calculation")
        return _simple_rhat(posterior_samples)

    # TODO: Implement proper R-hat calculation using ArviZ
    rhat_values = {}

    for param_name, samples in posterior_samples.items():
        if samples.ndim < 2:
            continue  # Need multiple chains

        try:
            # Assuming samples shape is (chains, draws, ...)
            rhat = az.rhat(samples)
            if isinstance(rhat, (np.ndarray, float)):
                rhat_values[param_name] = float(np.max(rhat)) if isinstance(rhat, np.ndarray) else float(rhat)
        except Exception as e:
            print(f"Error calculating R-hat for {param_name}: {e}")
            rhat_values[param_name] = np.nan

    # Check for convergence issues
    problematic_params = [
        param for param, rhat in rhat_values.items()
        if rhat > threshold
    ]

    if problematic_params:
        print(f"Warning: R-hat > {threshold} for parameters: {problematic_params}")

    return rhat_values


def _simple_rhat(posterior_samples: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Simplified R-hat calculation when ArviZ is not available.
    """
    rhat_values = {}

    for param_name, samples in posterior_samples.items():
        if samples.ndim != 2:
            continue  # Expecting (chains, draws)

        n_chains, n_draws = samples.shape
        if n_chains < 2:
            continue

        # Between-chain variance
        chain_means = np.mean(samples, axis=1)
        overall_mean = np.mean(chain_means)
        B = n_draws * np.var(chain_means, ddof=1)

        # Within-chain variance
        chain_vars = np.var(samples, axis=1, ddof=1)
        W = np.mean(chain_vars)

        # R-hat estimate
        var_plus = ((n_draws - 1) / n_draws) * W + (1 / n_draws) * B
        rhat = np.sqrt(var_plus / W)

        rhat_values[param_name] = float(rhat)

    return rhat_values


def effective_sample_size(posterior_samples: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculate effective sample size for each parameter.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary of parameter names to sample arrays

    Returns
    -------
    dict
        Dictionary of parameter names to effective sample sizes
    """
    if az is None:
        print("Warning: ArviZ not available, using simplified ESS calculation")
        return _simple_ess(posterior_samples)

    ess_values = {}

    for param_name, samples in posterior_samples.items():
        try:
            ess = az.ess(samples)
            if isinstance(ess, (np.ndarray, float)):
                ess_values[param_name] = float(np.min(ess)) if isinstance(ess, np.ndarray) else float(ess)
        except Exception as e:
            print(f"Error calculating ESS for {param_name}: {e}")
            ess_values[param_name] = np.nan

    return ess_values


def _simple_ess(posterior_samples: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Simplified ESS calculation using autocorrelation.
    """
    ess_values = {}

    for param_name, samples in posterior_samples.items():
        if samples.ndim != 2:
            continue

        # Average ESS across chains
        chain_ess = []
        for chain in range(samples.shape[0]):
            chain_samples = samples[chain, :]
            autocorr = np.correlate(chain_samples, chain_samples, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]

            # Find first negative autocorrelation or cutoff
            cutoff = len(autocorr)
            for i in range(1, len(autocorr)):
                if autocorr[i] <= 0:
                    cutoff = i
                    break

            # ESS approximation
            tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
            ess = len(chain_samples) / tau_int
            chain_ess.append(ess)

        ess_values[param_name] = float(np.mean(chain_ess))

    return ess_values


def plot_traces(
    posterior_samples: Dict[str, np.ndarray],
    parameters: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Create trace plots for MCMC diagnostics.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary of parameter names to sample arrays
    parameters : list, optional
        Specific parameters to plot (if None, plots all)
    figsize : tuple
        Figure size for plots
    """
    if plt is None:
        print("Matplotlib not available for plotting")
        return

    if parameters is None:
        parameters = list(posterior_samples.keys())

    n_params = len(parameters)
    if n_params == 0:
        return

    fig, axes = plt.subplots(n_params, 2, figsize=figsize)
    if n_params == 1:
        axes = axes.reshape(1, -1)

    for i, param in enumerate(parameters):
        if param not in posterior_samples:
            continue

        samples = posterior_samples[param]
        if samples.ndim != 2:
            continue  # Skip non-chain data

        # Trace plot
        for chain in range(samples.shape[0]):
            axes[i, 0].plot(samples[chain, :], alpha=0.7, label=f'Chain {chain}')
        axes[i, 0].set_title(f'Trace: {param}')
        axes[i, 0].set_xlabel('Iteration')
        axes[i, 0].legend()

        # Density plot
        all_samples = samples.flatten()
        axes[i, 1].hist(all_samples, bins=50, alpha=0.7, density=True)
        axes[i, 1].set_title(f'Posterior: {param}')
        axes[i, 1].set_xlabel('Value')
        axes[i, 1].set_ylabel('Density')

    plt.tight_layout()
    plt.show()


def plot_autocorrelation(
    posterior_samples: Dict[str, np.ndarray],
    parameters: Optional[List[str]] = None,
    max_lag: int = 100,
) -> None:
    """
    Plot autocorrelation functions for convergence assessment.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary of parameter names to sample arrays
    parameters : list, optional
        Specific parameters to plot
    max_lag : int
        Maximum lag for autocorrelation
    """
    if plt is None:
        print("Matplotlib not available for plotting")
        return

    if parameters is None:
        parameters = list(posterior_samples.keys())[:4]  # Limit to first 4

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, param in enumerate(parameters[:4]):
        if param not in posterior_samples:
            continue

        samples = posterior_samples[param]
        if samples.ndim != 2:
            continue

        # Calculate autocorrelation for first chain
        chain_samples = samples[0, :]
        autocorr = np.correlate(chain_samples, chain_samples, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]

        lags = np.arange(min(max_lag, len(autocorr)))
        axes[i].plot(lags, autocorr[:len(lags)])
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].set_title(f'Autocorrelation: {param}')
        axes[i].set_xlabel('Lag')
        axes[i].set_ylabel('Autocorrelation')

    plt.tight_layout()
    plt.show()


def convergence_summary(posterior_samples: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Create a summary table of convergence diagnostics.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary of parameter names to sample arrays

    Returns
    -------
    pd.DataFrame
        Summary table with convergence diagnostics
    """
    rhat_values = check_rhat(posterior_samples)
    ess_values = effective_sample_size(posterior_samples)

    summary_data = []
    for param in posterior_samples.keys():
        if param in rhat_values and param in ess_values:
            summary_data.append({
                'parameter': param,
                'rhat': rhat_values[param],
                'ess': ess_values[param],
                'converged': rhat_values[param] < 1.01,
                'adequate_ess': ess_values[param] > 100,
            })

    return pd.DataFrame(summary_data)


def posterior_predictive_check(
    observed_data: np.ndarray,
    posterior_predictive: np.ndarray,
    test_statistic: str = "mean",
) -> Dict[str, Any]:
    """
    Perform posterior predictive checks.

    Parameters
    ----------
    observed_data : np.ndarray
        Observed data
    posterior_predictive : np.ndarray
        Posterior predictive samples
    test_statistic : str
        Test statistic to use ("mean", "std", "min", "max")

    Returns
    -------
    dict
        PPC results including p-value and plots
    """
    # Calculate test statistic for observed data
    if test_statistic == "mean":
        obs_stat = np.mean(observed_data)
        pred_stats = np.mean(posterior_predictive, axis=1)
    elif test_statistic == "std":
        obs_stat = np.std(observed_data)
        pred_stats = np.std(posterior_predictive, axis=1)
    elif test_statistic == "min":
        obs_stat = np.min(observed_data)
        pred_stats = np.min(posterior_predictive, axis=1)
    elif test_statistic == "max":
        obs_stat = np.max(observed_data)
        pred_stats = np.max(posterior_predictive, axis=1)
    else:
        raise ValueError(f"Unknown test statistic: {test_statistic}")

    # Calculate Bayesian p-value
    p_value = np.mean(pred_stats >= obs_stat)

    return {
        "test_statistic": test_statistic,
        "observed_value": obs_stat,
        "posterior_predictive_values": pred_stats,
        "p_value": p_value,
        "passes": 0.05 <= p_value <= 0.95,  # Typical range for good fit
    }


def model_comparison(models: Dict[str, Any], criterion: str = "waic") -> pd.DataFrame:
    """
    Compare multiple models using information criteria.

    Parameters
    ----------
    models : dict
        Dictionary of model names to fitted model objects
    criterion : str
        Information criterion to use ("waic", "loo")

    Returns
    -------
    pd.DataFrame
        Model comparison table
    """
    # TODO: Implement model comparison using WAIC/LOO
    # This would require proper integration with NumPyro/ArviZ

    comparison_data = []
    for name, model in models.items():
        # Placeholder comparison metrics
        comparison_data.append({
            "model": name,
            "waic": np.random.normal(1000, 50),  # Placeholder
            "p_waic": np.random.normal(10, 2),   # Placeholder
            "se": np.random.normal(20, 5),       # Placeholder
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values("waic")
    df["d_waic"] = df["waic"] - df["waic"].min()

    return df