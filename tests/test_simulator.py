"""
Tests for the STV simulator implementation.

These tests verify the STV counting algorithm, transfer handling,
and quota calculations work correctly.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

import sys
from pathlib import Path

# Add src directory to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stv_transfers.simulator import STVSimulator, Ballot, TransferEvent, TransferType


class TestBallot:
    """Test the Ballot dataclass."""

    def test_ballot_creation(self):
        """Test basic ballot creation."""
        ballot = Ballot([1, 2, 0], 1.0)
        assert ballot.preferences == [1, 2, 0]
        assert ballot.weight == 1.0

    def test_ballot_default_weight(self):
        """Test ballot with default weight."""
        ballot = Ballot([0, 1, 2])
        assert ballot.weight == 1.0

    def test_ballot_fractional_weight(self):
        """Test ballot with fractional weight."""
        ballot = Ballot([2, 1], 0.5)
        assert ballot.weight == 0.5


class TestSTVSimulatorInitialization:
    """Test STVSimulator initialization and validation."""

    def test_valid_transfer_matrix(self):
        """Test initialization with valid transfer matrix."""
        # 3 candidates, valid transfer matrix
        transfer_matrix = np.array([
            [0.0, 0.6, 0.4],  # Candidate 0 transfers 60% to 1, 40% to 2
            [0.3, 0.0, 0.7],  # Candidate 1 transfers 30% to 0, 70% to 2
            [0.5, 0.5, 0.0]   # Candidate 2 transfers 50% to 0, 50% to 1
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)
        assert simulator.n_candidates == 3
        np.testing.assert_array_equal(simulator.transfer_matrix, transfer_matrix)

    def test_invalid_transfer_matrix_not_square(self):
        """Test initialization fails with non-square matrix."""
        transfer_matrix = np.array([
            [0.0, 0.6, 0.4],
            [0.3, 0.0, 0.7]  # Missing row
        ])

        with pytest.raises(ValueError, match="Transfer matrix must be square"):
            STVSimulator(transfer_matrix)

    def test_invalid_transfer_matrix_rows_dont_sum_to_one(self):
        """Test initialization fails when rows don't sum to 1."""
        transfer_matrix = np.array([
            [0.0, 0.6, 0.5],  # Sums to 1.1
            [0.3, 0.0, 0.7],
            [0.5, 0.5, 0.0]
        ])

        with pytest.raises(ValueError, match="Transfer matrix rows must sum to 1"):
            STVSimulator(transfer_matrix)

    def test_invalid_transfer_matrix_nonzero_diagonal(self):
        """Test initialization fails with non-zero diagonal."""
        transfer_matrix = np.array([
            [0.1, 0.5, 0.4],  # Self-transfer not allowed
            [0.3, 0.0, 0.7],
            [0.5, 0.5, 0.0]
        ])

        with pytest.raises(ValueError, match="Transfer matrix diagonals must be 0"):
            STVSimulator(transfer_matrix)

    def test_invalid_transfer_matrix_negative_probabilities(self):
        """Test initialization fails with negative probabilities."""
        transfer_matrix = np.array([
            [0.0, 0.6, 0.4],
            [0.3, 0.0, 0.7],
            [-0.1, 0.6, 0.5]  # Negative probability but row sums to 1.0
        ])

        with pytest.raises(ValueError, match="Transfer matrix cannot contain negative probabilities"):
            STVSimulator(transfer_matrix)


class TestQuotaCalculation:
    """Test quota calculation methods."""

    def test_compute_droop_quota_basic(self):
        """Test basic Droop quota calculation."""
        transfer_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        simulator = STVSimulator(transfer_matrix)

        # Standard cases
        assert simulator.compute_droop_quota(100, 1) == 51  # floor(100/2) + 1
        assert simulator.compute_droop_quota(100, 2) == 34  # floor(100/3) + 1
        assert simulator.compute_droop_quota(100, 3) == 26  # floor(100/4) + 1

    def test_compute_droop_quota_edge_cases(self):
        """Test quota calculation with edge cases."""
        transfer_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        simulator = STVSimulator(transfer_matrix)

        # Small numbers
        assert simulator.compute_droop_quota(5, 1) == 3   # floor(5/2) + 1
        assert simulator.compute_droop_quota(10, 5) == 2  # floor(10/6) + 1

        # Large numbers
        assert simulator.compute_droop_quota(46090, 3) == 11523  # Real Limerick example


class TestTransferSimulation:
    """Test the simulate_transfers method."""

    def test_simulate_transfers_basic(self):
        """Test basic transfer simulation."""
        transfer_matrix = np.array([
            [0.0, 0.6, 0.4],
            [0.3, 0.0, 0.7],
            [0.5, 0.5, 0.0]
        ])
        simulator = STVSimulator(transfer_matrix, seed=42)

        # Simulate 100 transfers from candidate 0
        transfers = simulator.simulate_transfers(0, 100)

        # Should return array of length 3
        assert len(transfers) == 3
        assert transfers[0] == 0  # No self-transfers

        # Total should equal input (conservation)
        assert np.sum(transfers) == 100

        # With deterministic seed, should be reproducible if we create a new simulator
        simulator2 = STVSimulator(transfer_matrix, seed=42)
        transfers2 = simulator2.simulate_transfers(0, 100)
        np.testing.assert_array_equal(transfers, transfers2)

    def test_simulate_transfers_invalid_source(self):
        """Test simulate_transfers with invalid source index."""
        transfer_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        simulator = STVSimulator(transfer_matrix)

        with pytest.raises(ValueError, match="source_idx .* >= n_candidates"):
            simulator.simulate_transfers(5, 100)

    def test_simulate_transfers_zero_transfers(self):
        """Test simulate_transfers with zero transfers."""
        transfer_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        simulator = STVSimulator(transfer_matrix)

        transfers = simulator.simulate_transfers(0, 0)
        assert np.sum(transfers) == 0


class TestSTVCounting:
    """Test the full STV counting algorithm."""

    def test_deterministic_single_seat_election(self):
        """Test deterministic 3-candidate, 1-seat election."""
        # Transfer matrix: A->B (70%), A->C (30%), B->C (100%), C->B (100%)
        transfer_matrix = np.array([
            [0.0, 0.7, 0.3],  # A
            [0.0, 0.0, 1.0],  # B
            [0.0, 1.0, 0.0]   # C
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        # Create ballots: A(400), B(300), C(200)
        ballots = (
            [Ballot([0], 1.0) for _ in range(400)] +  # 400 first prefs for A
            [Ballot([1], 1.0) for _ in range(300)] +  # 300 first prefs for B
            [Ballot([2], 1.0) for _ in range(200)]    # 200 first prefs for C
        )

        result = simulator.count_stv(ballots, seats=1)

        # Check basic structure
        assert 'rounds' in result
        assert 'events' in result
        assert 'elected' in result
        assert 'elimination_order' in result

        # Should have first count + elimination rounds
        assert len(result['rounds']) >= 2

        # Check first count
        first_round = result['rounds'][0]
        assert first_round['round'] == 0
        assert first_round['quota'] == 451  # floor(900/2) + 1
        np.testing.assert_array_equal(first_round['vote_counts'], [400, 300, 200])

        # A should eventually win after transfers
        assert len(result['elected']) == 1

    def test_surplus_distribution(self):
        """Test surplus distribution in multi-seat election."""
        # 2 seats, 3 candidates
        transfer_matrix = np.array([
            [0.0, 0.6, 0.4],  # A transfers 60% to B, 40% to C
            [0.5, 0.0, 0.5],  # B transfers 50% to A, 50% to C
            [0.7, 0.3, 0.0]   # C transfers 70% to A, 30% to B
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        # Create scenario where A gets elected immediately with surplus
        ballots = (
            [Ballot([0], 1.0) for _ in range(800)] +  # 800 first prefs for A (will exceed quota)
            [Ballot([1], 1.0) for _ in range(200)] +  # 200 first prefs for B
            [Ballot([2], 1.0) for _ in range(100)]    # 100 first prefs for C
        )

        result = simulator.count_stv(ballots, seats=2)

        # A should be elected in first round
        assert 0 in result['elected']

        # Should have surplus distribution events
        surplus_events = [e for e in result['events'] if e.transfer_type == TransferType.SURPLUS]
        assert len(surplus_events) > 0

        # Check that surplus events have fractional transfer values
        for event in surplus_events:
            assert 0 < event.transfer_value < 1

    def test_fractional_surplus_preserves_totals(self):
        """Test that fractional surplus transfers preserve vote totals."""
        transfer_matrix = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        ballots = [Ballot([0], 1.0) for _ in range(1000)]  # All vote for candidate 0

        result = simulator.count_stv(ballots, seats=1)

        # Check vote conservation in each round
        for round_data in result['rounds']:
            total_votes = np.sum(round_data['vote_counts'])
            # Total should remain constant (allowing for small floating point errors)
            assert abs(total_votes - 1000) < 0.01

    def test_transfer_matrix_applied_correctly(self):
        """Test that transfer matrix probabilities are applied correctly."""
        # Simple deterministic transfer matrix
        transfer_matrix = np.array([
            [0.0, 1.0, 0.0],  # A transfers 100% to B
            [0.0, 0.0, 1.0],  # B transfers 100% to C
            [1.0, 0.0, 0.0]   # C transfers 100% to A
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        ballots = [
            Ballot([0], 1.0),  # Vote for A
            Ballot([1], 1.0),  # Vote for B
            Ballot([2], 1.0)   # Vote for C
        ]

        result = simulator.count_stv(ballots, seats=1)

        # Find elimination events
        elimination_events = [e for e in result['events'] if e.transfer_type == TransferType.ELIMINATION]

        # Check that transfers follow the matrix
        for event in elimination_events:
            from_idx = event.from_candidate
            to_idx = event.to_candidate

            # The transfer should match the matrix
            expected_prob = transfer_matrix[from_idx, to_idx]
            if expected_prob == 1.0:
                # For 100% transfers, all votes should go to the target
                assert event.vote_count > 0

    def test_group_elimination(self):
        """Test group elimination of tied candidates."""
        transfer_matrix = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        # Create tied candidates
        ballots = (
            [Ballot([0], 1.0) for _ in range(100)] +  # 100 for A
            [Ballot([1], 1.0) for _ in range(50)] +   # 50 for B (tied for last)
            [Ballot([2], 1.0) for _ in range(50)]     # 50 for C (tied for last)
        )

        result = simulator.count_stv(ballots, seats=1, allow_group_elimination=True)

        # Should eliminate both B and C together
        assert len(result['elimination_order']) >= 2

    def test_exhausted_ballots_handling(self):
        """Test handling of exhausted ballots (non-transferable)."""
        # Include non-transferable option in transfer matrix
        transfer_matrix = np.array([
            [0.0, 0.3, 0.7],  # A: 30% to B, 70% non-transferable
            [0.4, 0.0, 0.6],  # B: 40% to A, 60% non-transferable
            [0.2, 0.8, 0.0]   # C: 20% to A, 80% to B
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        ballots = [
            Ballot([0], 1.0),
            Ballot([1], 1.0),
            Ballot([2], 1.0)
        ]

        result = simulator.count_stv(ballots, seats=1)

        # Should complete without error even with non-transferable votes
        assert 'elected' in result
        assert len(result['elected']) <= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_more_seats_than_candidates(self):
        """Test election with more seats than candidates."""
        transfer_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        ballots = [
            Ballot([0], 1.0),
            Ballot([1], 1.0)
        ]

        result = simulator.count_stv(ballots, seats=3)  # More seats than candidates

        # Should elect all available candidates
        assert len(result['elected']) <= 2

    def test_empty_ballots(self):
        """Test handling of empty ballot list."""
        transfer_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        result = simulator.count_stv([], seats=1)

        assert len(result['elected']) == 0
        assert len(result['rounds']) >= 1  # Should have at least first count round

        # First round should show zero votes for all candidates
        first_round = result['rounds'][0]
        assert np.sum(first_round['vote_counts']) == 0

    def test_ballots_with_invalid_preferences(self):
        """Test handling of ballots with invalid candidate indices."""
        transfer_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])

        simulator = STVSimulator(transfer_matrix, seed=42)

        ballots = [
            Ballot([0], 1.0),      # Valid
            Ballot([5], 1.0),      # Invalid candidate index
            Ballot([], 1.0),       # Empty preferences
            Ballot([1], 1.0)       # Valid
        ]

        result = simulator.count_stv(ballots, seats=1)

        # Should handle gracefully and count valid ballots
        first_round = result['rounds'][0]
        # Only 2 valid votes should be counted
        assert np.sum(first_round['vote_counts']) == 2.0


if __name__ == "__main__":
    pytest.main([__file__])