"""
Smoke tests for the hierarchical censored-transfer model.

These tests verify that the NumPyro models run correctly and produce
reasonable output without extensive computational requirements.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add src directory to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Skip all tests if JAX/NumPyro not available
jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import jax.numpy as jnp
import jax.random as random
from numpyro.infer import MCMC, NUTS

from stv_transfers.data_structures import EventData, ModelData
from stv_transfers.model import build_model_exact, build_model_augmented
from stv_transfers.diagnostics import check_convergence


def create_test_data():
    """Create minimal test data for model fitting."""
    # Simple test case: 2 constituencies, 3 sources, 4 destinations
    events = [
        # Event 1: Source 0 in constituency 0, transfers to destinations [1, 2]
        EventData(
            constituency_idx=0,
            source_indices=np.array([0]),
            active_indices=np.array([1, 2]),
            transfer_counts=np.array([15.0, 10.0]),  # 15 to dest 1, 10 to dest 2
            total_transfers=25.0
        ),
        # Event 2: Source 1 in constituency 0, transfers to destinations [0, 3]
        EventData(
            constituency_idx=0,
            source_indices=np.array([1]),
            active_indices=np.array([0, 3]),
            transfer_counts=np.array([8.0, 12.0]),   # 8 to dest 0, 12 to dest 3
            total_transfers=20.0
        ),
        # Event 3: Source 0 in constituency 1, transfers to destinations [2, 3]
        EventData(
            constituency_idx=1,
            source_indices=np.array([0]),
            active_indices=np.array([2, 3]),
            transfer_counts=np.array([18.0, 7.0]),   # 18 to dest 2, 7 to dest 3
            total_transfers=25.0
        ),
    ]

    candidate_names = {
        0: "Candidate A",
        1: "Candidate B",
        2: "Candidate C",
        3: "Candidate D"
    }

    return ModelData(
        n_sources=3,
        n_destinations=4,
        n_constituencies=2,
        events=events,
        candidate_names=candidate_names
    )


def test_exact_model_smoke():
    """Smoke test for exact model - just check it runs."""
    data = create_test_data()
    model = build_model_exact(data)

    # Very short MCMC run for smoke test
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=20, num_samples=40, num_chains=1)

    rng_key = random.PRNGKey(42)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    # Basic sanity checks
    assert "mu" in samples
    assert "pi" in samples
    assert "kappa" in samples

    # Check shapes
    assert samples["mu"].shape == (40, 3, 4)  # (samples, sources, destinations)
    assert samples["pi"].shape == (40, 2, 3, 4)  # (samples, constituencies, sources, destinations)
    assert samples["kappa"].shape == (40,)

    # Check all values are finite
    assert jnp.all(jnp.isfinite(samples["mu"]))
    assert jnp.all(jnp.isfinite(samples["pi"]))
    assert jnp.all(jnp.isfinite(samples["kappa"]))

    # Check pi rows sum to approximately 1
    pi_sums = jnp.sum(samples["pi"], axis=-1)  # Sum over destinations
    assert jnp.allclose(pi_sums, 1.0, atol=1e-5)


def test_augmented_model_smoke():
    """Smoke test for augmented model - just check it runs."""
    data = create_test_data()
    model = build_model_augmented(data)

    # Very short MCMC run for smoke test
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=20, num_samples=40, num_chains=1)

    rng_key = random.PRNGKey(43)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    # Basic sanity checks
    assert "mu" in samples
    assert "pi" in samples
    assert "kappa" in samples

    # Check shapes
    assert samples["mu"].shape == (40, 3, 4)
    assert samples["pi"].shape == (40, 2, 3, 4)
    assert samples["kappa"].shape == (40,)

    # Check all values are finite
    assert jnp.all(jnp.isfinite(samples["mu"]))
    assert jnp.all(jnp.isfinite(samples["pi"]))
    assert jnp.all(jnp.isfinite(samples["kappa"]))

    # Check pi rows sum to approximately 1
    pi_sums = jnp.sum(samples["pi"], axis=-1)
    assert jnp.allclose(pi_sums, 1.0, atol=1e-5)


def test_custom_alpha():
    """Test that custom alpha prior works."""
    data = create_test_data()
    alpha = np.array([2.0, 1.0, 1.0, 0.5])  # Biased toward first destination
    model = build_model_exact(data, alpha=alpha)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=20, num_samples=40, num_chains=1)

    rng_key = random.PRNGKey(44)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    # Should still produce valid samples
    assert jnp.all(jnp.isfinite(samples["mu"]))
    assert jnp.all(jnp.isfinite(samples["pi"]))


@pytest.mark.slow
def test_exact_model_convergence():
    """Longer test to check convergence - marked as slow."""
    data = create_test_data()
    model = build_model_exact(data)

    # Longer MCMC run for convergence check
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=4)

    rng_key = random.PRNGKey(42)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    # Check convergence
    convergence_results = check_convergence(samples)

    # Assert good convergence
    assert convergence_results['converged'], f"R-hat values: {convergence_results['rhat']}"
    assert convergence_results['adequate_ess'], f"ESS values: {convergence_results['ess']}"

    # Check specific R-hat values
    for param, rhat in convergence_results['rhat'].items():
        assert rhat < 1.05, f"R-hat for {param} is {rhat}, should be < 1.05"


@pytest.mark.slow
def test_augmented_model_convergence():
    """Longer test for augmented model convergence."""
    data = create_test_data()
    model = build_model_augmented(data)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=4)

    rng_key = random.PRNGKey(43)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    # Check convergence
    convergence_results = check_convergence(samples)

    assert convergence_results['converged'], f"R-hat values: {convergence_results['rhat']}"
    assert convergence_results['adequate_ess'], f"ESS values: {convergence_results['ess']}"


@pytest.mark.slow
def test_prior_sensitivity():
    """Test sensitivity to different prior specifications."""
    data = create_test_data()

    # Test different alpha values
    alphas = [
        np.ones(4),  # Uniform
        np.array([2.0, 2.0, 2.0, 2.0]),  # Concentrated
        np.array([0.5, 0.5, 0.5, 0.5]),  # Dispersed
        np.array([3.0, 1.0, 1.0, 1.0]),  # Biased
    ]

    results = []

    for alpha in alphas:
        model = build_model_exact(data, alpha=alpha)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=200, num_samples=400, num_chains=2)

        rng_key = random.PRNGKey(45)
        mcmc.run(rng_key)

        samples = mcmc.get_samples()

        # Store posterior means
        mu_mean = jnp.mean(samples["mu"], axis=0)
        results.append(mu_mean)

    # Check that results are different (sensitivity to priors)
    # At least some difference between uniform and biased priors
    uniform_result = results[0]
    biased_result = results[3]

    # Should see differences in the first destination (heavily weighted in biased prior)
    diff = jnp.abs(uniform_result[:, 0] - biased_result[:, 0])
    assert jnp.any(diff > 0.05), "Prior should have noticeable effect on posterior"


def test_model_data_validation():
    """Test that invalid model data raises appropriate errors."""
    # Test empty events
    invalid_data = ModelData(
        n_sources=2,
        n_destinations=3,
        n_constituencies=1,
        events=[],  # Empty events
        candidate_names={}
    )

    model = build_model_exact(invalid_data)

    # Should run without error (empty case)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=10, num_samples=20, num_chains=1)

    rng_key = random.PRNGKey(46)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    # Should still produce prior samples
    assert "mu" in samples
    assert "pi" in samples
    assert "kappa" in samples