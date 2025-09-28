"""
STV election simulation and validation tools.

This module provides functions for simulating STV elections with known
transfer patterns, useful for model validation and testing.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


@dataclass
class Ballot:
    """Represents a ballot with preference ordering and weight."""
    preferences: List[int]  # Candidate indices in order of preference
    weight: float = 1.0     # Fractional weight for transferred ballots


class TransferType(Enum):
    """Types of transfers in STV elections."""
    ELIMINATION = "elimination"
    SURPLUS = "surplus"


@dataclass
class TransferEvent:
    """Represents a transfer of votes between candidates."""
    from_candidate: int
    to_candidate: int
    vote_count: float
    transfer_value: float
    round_number: int
    transfer_type: TransferType


class STVSimulator:
    """
    Simulator for STV elections with configurable transfer patterns.

    This class allows researchers to simulate STV elections with known
    transfer probabilities, useful for validating Bayesian models and
    understanding the impact of different voter behaviors.
    """

    def __init__(self, transfer_matrix: np.ndarray, seed: Optional[int] = None):
        """
        Initialize the STV simulator.

        Parameters
        ----------
        transfer_matrix : np.ndarray
            Matrix where element [i,j] is probability of transferring
            from source candidate i to destination candidate j.
            Can be rectangular (n_sources × n_destinations) or square.
            Rows should sum to ≈1.
        seed : int, optional
            Random seed for reproducible simulations
        """
        self.transfer_matrix = np.array(transfer_matrix, dtype=np.float64)
        self.n_sources = self.transfer_matrix.shape[0]
        self.n_destinations = self.transfer_matrix.shape[1]
        # For backward compatibility
        self.n_candidates = max(self.n_sources, self.n_destinations)

        # Validate transfer matrix
        self.validate_transfer_matrix()

        # Set up random number generator
        self.rng = np.random.default_rng(seed)

    def validate_transfer_matrix(self) -> None:
        """Validate that transfer matrix is properly formed."""
        # Check that all probabilities are non-negative
        if np.any(self.transfer_matrix < 0):
            raise ValueError("Transfer matrix cannot contain negative probabilities")

        # Check that diagonals are zero (no self-transfers) if square matrix
        if self.n_sources == self.n_destinations:
            diagonals = np.diag(self.transfer_matrix)
            if not np.allclose(diagonals, 0.0, atol=1e-10):
                raise ValueError(f"Transfer matrix diagonals must be 0 (no self-transfers). Got: {diagonals}")

        # Check that rows sum to approximately 1
        row_sums = np.sum(self.transfer_matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(f"Transfer matrix rows must sum to 1. Got sums: {row_sums}")

    def compute_droop_quota(self, total_valid: int, seats: int) -> int:
        """
        Compute the Droop quota for STV elections.

        Parameters
        ----------
        total_valid : int
            Total number of valid votes cast
        seats : int
            Number of seats to be filled

        Returns
        -------
        int
            The Droop quota
        """
        return int(total_valid / (seats + 1)) + 1

    def simulate_transfers(self, source_idx: int, n_transfers: int) -> np.ndarray:
        """
        Simulate transfer of votes using the transfer matrix.

        Parameters
        ----------
        source_idx : int
            Index of the source candidate whose votes are being transferred
        n_transfers : int
            Number of transfers to simulate

        Returns
        -------
        np.ndarray
            Array of length n_destinations with count of transfers to each destination
        """
        if source_idx >= self.n_sources:
            raise ValueError(f"source_idx {source_idx} >= n_sources {self.n_sources}")

        # Sample from multinomial distribution using transfer probabilities
        transfer_probs = self.transfer_matrix[source_idx]
        transfers = self.rng.multinomial(n_transfers, transfer_probs)

        return transfers.astype(np.float64)

    def count_stv(self, ballots: List[Ballot], seats: int,
                  allow_group_elimination: bool = True) -> Dict[str, Any]:
        """
        Conduct a full STV count with proper transfer handling.

        Parameters
        ----------
        ballots : List[Ballot]
            List of ballots with preference orderings
        seats : int
            Number of seats to fill
        allow_group_elimination : bool
            Whether to allow group elimination of tied candidates

        Returns
        -------
        Dict[str, Any]
            Results containing:
            - rounds: List of round-by-round results
            - events: List of TransferEvent objects
            - elected: List of elected candidate indices
            - elimination_order: List of eliminated candidate indices
        """
        # Initialize vote counts
        vote_counts = np.zeros(self.n_candidates, dtype=np.float64)
        elected = []
        eliminated = []
        events = []
        rounds = []
        round_num = 0

        # Count first preferences
        for ballot in ballots:
            if ballot.preferences:
                first_choice = ballot.preferences[0]
                if 0 <= first_choice < self.n_candidates:
                    vote_counts[first_choice] += ballot.weight

        total_valid = sum(ballot.weight for ballot in ballots)
        quota = self.compute_droop_quota(int(total_valid), seats)

        # Store round 0 (first count)
        rounds.append({
            'round': 0,
            'vote_counts': vote_counts.copy(),
            'quota': quota,
            'elected': [],
            'eliminated': [],
            'action': 'First Count'
        })

        while len(elected) < seats and len(elected) + len(eliminated) < self.n_candidates:
            round_num += 1
            round_elected = []
            round_eliminated = []

            # Check for candidates reaching quota
            for cand_idx in range(self.n_candidates):
                if (cand_idx not in elected and
                    cand_idx not in eliminated and
                    vote_counts[cand_idx] >= quota):

                    elected.append(cand_idx)
                    round_elected.append(cand_idx)

                    # Handle surplus distribution (Gregory method)
                    surplus = vote_counts[cand_idx] - quota
                    if surplus > 0.001:  # Only distribute meaningful surplus
                        self._distribute_surplus(
                            cand_idx, surplus, vote_counts, ballots,
                            elected, eliminated, events, round_num
                        )

            # If no one elected and we need more seats, eliminate candidate(s)
            if not round_elected and len(elected) < seats:
                candidates_remaining = [i for i in range(self.n_candidates)
                                     if i not in elected and i not in eliminated]

                if candidates_remaining:
                    if allow_group_elimination:
                        # Find candidates with lowest vote count
                        min_votes = min(vote_counts[i] for i in candidates_remaining)
                        to_eliminate = [i for i in candidates_remaining
                                      if vote_counts[i] == min_votes]
                    else:
                        # Eliminate single candidate with lowest votes
                        min_idx = min(candidates_remaining, key=lambda i: vote_counts[i])
                        to_eliminate = [min_idx]

                    for cand_idx in to_eliminate:
                        eliminated.append(cand_idx)
                        round_eliminated.append(cand_idx)

                        # Distribute eliminated candidate's votes
                        self._distribute_elimination(
                            cand_idx, vote_counts, ballots,
                            elected, eliminated, events, round_num
                        )

            # Record this round
            rounds.append({
                'round': round_num,
                'vote_counts': vote_counts.copy(),
                'quota': quota,
                'elected': round_elected,
                'eliminated': round_eliminated,
                'action': f"{'Election' if round_elected else 'Elimination'}"
            })

            # Break if we've elected enough candidates
            if len(elected) >= seats:
                break

        return {
            'rounds': rounds,
            'events': events,
            'elected': elected,
            'elimination_order': eliminated
        }

    def _distribute_surplus(self, elected_idx: int, surplus: float,
                           vote_counts: np.ndarray, ballots: List[Ballot],
                           elected: List[int], eliminated: List[int],
                           events: List[TransferEvent], round_num: int) -> None:
        """
        Distribute surplus votes using Gregory method with fractional transfers.
        """
        # Calculate transfer value for surplus distribution
        total_votes = vote_counts[elected_idx]
        transfer_value = surplus / total_votes

        # Use transfer matrix to determine distribution
        transfers = self.simulate_transfers(elected_idx, int(surplus))

        # Apply transfers to remaining candidates
        for to_idx, transfer_count in enumerate(transfers):
            if (transfer_count > 0 and
                to_idx not in elected and
                to_idx not in eliminated and
                to_idx != elected_idx):

                vote_counts[to_idx] += transfer_count
                events.append(TransferEvent(
                    from_candidate=elected_idx,
                    to_candidate=to_idx,
                    vote_count=transfer_count,
                    transfer_value=transfer_value,
                    round_number=round_num,
                    transfer_type=TransferType.SURPLUS
                ))

        # Reduce elected candidate's votes to quota
        vote_counts[elected_idx] -= surplus

    def _distribute_elimination(self, eliminated_idx: int,
                               vote_counts: np.ndarray, ballots: List[Ballot],
                               elected: List[int], eliminated: List[int],
                               events: List[TransferEvent], round_num: int) -> None:
        """
        Distribute votes from eliminated candidate.
        """
        total_votes = vote_counts[eliminated_idx]

        if total_votes > 0:
            # Use transfer matrix to determine distribution
            transfers = self.simulate_transfers(eliminated_idx, int(total_votes))

            # Apply transfers to remaining candidates
            for to_idx, transfer_count in enumerate(transfers):
                if (transfer_count > 0 and
                    to_idx not in elected and
                    to_idx not in eliminated and
                    to_idx != eliminated_idx):

                    vote_counts[to_idx] += transfer_count
                    events.append(TransferEvent(
                        from_candidate=eliminated_idx,
                        to_candidate=to_idx,
                        vote_count=transfer_count,
                        transfer_value=1.0,  # Full value for eliminations
                        round_number=round_num,
                        transfer_type=TransferType.ELIMINATION
                    ))

        # Set eliminated candidate's votes to zero
        vote_counts[eliminated_idx] = 0