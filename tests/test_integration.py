"""
Integration tests for the complete STV transfer analysis pipeline.

These tests verify that all components work together correctly:
scraper → data structures → Bayesian model → simulation → analysis
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stv_transfers.scraper import ElectionScraper
from stv_transfers.data_structures import ModelData, EventData
from stv_transfers.model import build_model_exact, build_model_augmented
from stv_transfers.simulator import STVSimulator, Ballot
from stv_transfers.diagnostics import compute_rhat, effective_sample_size

# Skip tests if NumPyro not available
pytest.importorskip("numpyro")
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random


@pytest.fixture
def sample_scraped_data():
    """Create sample scraped data for testing."""
    return [
        {
            'constituency_id': 'test_constituency_1',
            'candidates': ['Alice', 'Bob', 'Charlie', 'non-transferable'],
            'transfers': [
                {
                    'from_candidate': 'Charlie',
                    'to_candidate': 'Alice',
                    'transfer_count': 150,
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                },
                {
                    'from_candidate': 'Charlie',
                    'to_candidate': 'Bob',
                    'transfer_count': 100,
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                },
                {
                    'from_candidate': 'Charlie',
                    'to_candidate': 'non-transferable',
                    'transfer_count': 50,
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                },
                {
                    'from_candidate': 'Alice',
                    'to_candidate': 'Bob',
                    'transfer_count': 80,
                    'transfer_type': 'surplus',
                    'transfer_value': 0.4
                },
                {
                    'from_candidate': 'Alice',
                    'to_candidate': 'non-transferable',
                    'transfer_count': 20,
                    'transfer_type': 'surplus',
                    'transfer_value': 0.4
                }
            ],
            'quota': 1000,
            'valid_poll': 4000
        }
    ]


@pytest.fixture
def sample_model_data(sample_scraped_data):
    """Convert sample scraped data to ModelData format."""
    scraper = ElectionScraper("https://test.com")
    return scraper.convert_to_model_format(sample_scraped_data)


class TestEndToEndPipeline:
    """Test the complete end-to-end analysis pipeline."""

    def test_scraper_to_model_data_conversion(self, sample_scraped_data):
        """Test conversion from scraped data to ModelData format."""
        scraper = ElectionScraper("https://test.com")
        model_data = scraper.convert_to_model_format(sample_scraped_data)

        # Verify ModelData structure
        assert isinstance(model_data, ModelData)
        assert model_data.n_constituencies == 1
        assert model_data.n_sources >= 2  # At least Alice and Charlie as sources
        assert model_data.n_destinations >= 3  # Alice, Bob, non-transferable
        assert len(model_data.events) >= 2  # At least 2 transfer events
        assert len(model_data.candidate_names) >= 3

        # Verify non-transferable has index 0
        assert 0 in model_data.candidate_names
        assert model_data.candidate_names[0] == 'non-transferable'

    def test_model_data_to_bayesian_model(self, sample_model_data):
        """Test that ModelData can be used to build Bayesian models."""
        # Test exact model
        exact_model = build_model_exact(sample_model_data)
        assert callable(exact_model)

        # Test augmented model
        augmented_model = build_model_augmented(sample_model_data)
        assert callable(augmented_model)

    @pytest.mark.slow
    def test_tiny_mcmc_inference(self, sample_model_data):
        """Test tiny MCMC inference for integration testing."""
        model = build_model_exact(sample_model_data)

        # Very small MCMC for testing
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=20, num_warmup=10, num_chains=1)

        rng_key = random.PRNGKey(42)
        mcmc.run(rng_key)

        samples = mcmc.get_samples()

        # Verify samples structure
        assert 'mu' in samples
        assert 'kappa' in samples
        assert 'pi' in samples

        # Verify dimensions
        mu_samples = samples['mu']
        assert mu_samples.shape[1] == sample_model_data.n_sources
        assert mu_samples.shape[2] == sample_model_data.n_destinations

    @pytest.mark.slow
    def test_posterior_to_simulation(self, sample_model_data):
        """Test using posterior samples to create simulation transfer matrix."""
        model = build_model_exact(sample_model_data)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=10, num_warmup=5, num_chains=1)

        rng_key = random.PRNGKey(42)
        mcmc.run(rng_key)

        samples = mcmc.get_samples()

        # Extract posterior mean transfer matrix
        mu_samples = samples['mu']  # Shape: (samples, sources, destinations)
        posterior_mean_mu = np.mean(mu_samples, axis=0)

        # Verify it's a valid transfer matrix
        assert posterior_mean_mu.shape == (sample_model_data.n_sources, sample_model_data.n_destinations)

        # Check rows sum approximately to 1
        row_sums = np.sum(posterior_mean_mu, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.1)  # Loose tolerance for small sample

        # Use for simulation
        simulator = STVSimulator(posterior_mean_mu, seed=42)

        # Test basic simulation functionality
        transfers = simulator.simulate_transfers(1, 100)  # Transfer 100 votes from source 1
        assert len(transfers) == sample_model_data.n_destinations
        assert np.sum(transfers) == 100

    def test_complete_pipeline_flow(self, sample_scraped_data):
        """Test the complete pipeline from scraped data to simulation."""
        # Step 1: Convert scraped data to ModelData
        scraper = ElectionScraper("https://test.com")
        model_data = scraper.convert_to_model_format(sample_scraped_data)

        # Step 2: Build Bayesian model
        bayesian_model = build_model_exact(model_data)

        # Step 3: Run tiny inference
        nuts_kernel = NUTS(bayesian_model)
        mcmc = MCMC(nuts_kernel, num_samples=10, num_warmup=5, num_chains=1)
        rng_key = random.PRNGKey(42)
        mcmc.run(rng_key)

        # Step 4: Extract posterior
        samples = mcmc.get_samples()
        mu_posterior = np.mean(samples['mu'], axis=0)

        # Step 5: Create simulator
        simulator = STVSimulator(mu_posterior, seed=42)

        # Step 6: Run simulation
        ballots = [Ballot([1, 2], 1.0) for _ in range(100)]  # 100 ballots preferring candidate 1 then 2
        result = simulator.count_stv(ballots, seats=1)

        # Verify complete pipeline worked
        assert len(result['rounds']) >= 1
        assert len(result['elected']) <= 1
        assert 'events' in result


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_model_data_validation(self, sample_model_data):
        """Test that ModelData is internally consistent."""
        # Check all events reference valid candidate indices
        for event in sample_model_data.events:
            assert all(0 <= idx < sample_model_data.n_destinations for idx in event.active_indices)
            assert all(0 <= idx < sample_model_data.n_sources for idx in event.source_indices)

            # Check transfer counts are non-negative
            assert all(count >= 0 for count in event.transfer_counts)
            assert event.total_transfers >= 0

            # Check constituency index is valid
            assert 0 <= event.constituency_idx < sample_model_data.n_constituencies

    def test_transfer_matrix_properties(self, sample_model_data):
        """Test properties of inferred transfer matrices."""
        model = build_model_exact(sample_model_data)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=10, num_warmup=5, num_chains=1)
        rng_key = random.PRNGKey(42)
        mcmc.run(rng_key)

        samples = mcmc.get_samples()
        mu_samples = samples['mu']

        # Test each sample is a valid transfer matrix
        for sample_idx in range(mu_samples.shape[0]):
            transfer_matrix = mu_samples[sample_idx]

            # Rows should sum to 1
            row_sums = np.sum(transfer_matrix, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

            # All probabilities should be non-negative
            assert np.all(transfer_matrix >= 0)

    def test_simulation_vote_conservation(self):
        """Test that simulation conserves votes properly."""
        # Simple deterministic transfer matrix
        transfer_matrix = np.array([
            [0.0, 0.6, 0.4],
            [0.3, 0.0, 0.7],
            [0.5, 0.5, 0.0]
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        # Create ballots
        ballots = [Ballot([0], 1.0) for _ in range(300)]

        result = simulator.count_stv(ballots, seats=1)

        # Check vote conservation in each round (allowing for small floating point errors)
        for round_data in result['rounds']:
            total_votes = np.sum(round_data['vote_counts'])
            assert abs(total_votes - 300) <= 1.0  # Allow for rounding


class TestModelComparison:
    """Test comparison between exact and augmented models."""

    @pytest.mark.slow
    def test_model_comparison(self, sample_model_data):
        """Compare exact and augmented model results."""
        # Run both models with same data
        exact_model = build_model_exact(sample_model_data)
        augmented_model = build_model_augmented(sample_model_data)

        rng_key = random.PRNGKey(42)

        # Run exact model
        nuts_exact = NUTS(exact_model)
        mcmc_exact = MCMC(nuts_exact, num_samples=20, num_warmup=10, num_chains=1)
        mcmc_exact.run(rng_key)
        samples_exact = mcmc_exact.get_samples()

        # Run augmented model
        nuts_augmented = NUTS(augmented_model)
        mcmc_augmented = MCMC(nuts_augmented, num_samples=20, num_warmup=10, num_chains=1)
        mcmc_augmented.run(rng_key)
        samples_augmented = mcmc_augmented.get_samples()

        # Both should have similar structure
        assert 'mu' in samples_exact and 'mu' in samples_augmented
        assert 'kappa' in samples_exact and 'kappa' in samples_augmented
        assert 'pi' in samples_exact and 'pi' in samples_augmented

        # Shapes should match
        assert samples_exact['mu'].shape == samples_augmented['mu'].shape


class TestRobustness:
    """Test robustness to edge cases and data quality issues."""

    def test_small_sample_sizes(self):
        """Test behavior with very small transfer counts."""
        # Create minimal data
        scraped_data = [{
            'constituency_id': 'minimal',
            'candidates': ['A', 'B', 'non-transferable'],
            'transfers': [
                {
                    'from_candidate': 'A',
                    'to_candidate': 'B',
                    'transfer_count': 1,  # Very small transfer
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                },
                {
                    'from_candidate': 'A',
                    'to_candidate': 'non-transferable',
                    'transfer_count': 1,
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                }
            ],
            'quota': 100,
            'valid_poll': 300
        }]

        scraper = ElectionScraper("https://test.com")
        model_data = scraper.convert_to_model_format(scraped_data)

        # Should not crash with small data
        model = build_model_exact(model_data)
        assert callable(model)

    def test_single_candidate_transfers(self):
        """Test with transfers from only one candidate."""
        scraped_data = [{
            'constituency_id': 'single_source',
            'candidates': ['A', 'B', 'C', 'non-transferable'],
            'transfers': [
                {
                    'from_candidate': 'A',
                    'to_candidate': 'B',
                    'transfer_count': 100,
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                },
                {
                    'from_candidate': 'A',
                    'to_candidate': 'C',
                    'transfer_count': 50,
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                }
            ],
            'quota': 200,
            'valid_poll': 600
        }]

        scraper = ElectionScraper("https://test.com")
        model_data = scraper.convert_to_model_format(scraped_data)

        # Verify single source is handled correctly
        assert model_data.n_sources == 1  # Only candidate A transfers votes
        assert model_data.n_destinations >= 3  # B, C, non-transferable
        assert len(model_data.events) >= 1


if __name__ == "__main__":
    pytest.main([__file__])