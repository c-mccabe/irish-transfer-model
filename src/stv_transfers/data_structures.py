"""
Shared data structures for STV transfer analysis.

This module defines common data formats and structures used throughout
the package for representing electoral data, candidates, and results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum


@dataclass
class EventData:
    """
    Represents a single transfer event for hierarchical modeling.

    This structure captures the essential information for modeling
    transfer patterns in a constituency-specific transfer event.
    Uses party-level aggregation for better statistical power.
    """
    constituency_idx: int
    source_indices: np.ndarray  # Indices of source parties
    active_indices: np.ndarray  # Indices of active destination parties
    transfer_counts: np.ndarray  # Observed transfer counts to active parties
    total_transfers: float  # Total number of transfers from source


@dataclass
class ModelData:
    """
    Complete dataset for hierarchical censored-transfer modeling.

    This structure organizes all transfer events across constituencies
    for efficient Bayesian modeling with NumPyro. Uses party-level
    aggregation for better statistical power and interpretability.
    """
    n_source_parties: int  # Total number of source parties (I)
    n_dest_parties: int  # Total number of destination parties (J)
    n_constituencies: int  # Total number of constituencies (C)
    events: List[EventData]  # List of all transfer events
    party_names: Dict[int, str]  # Mapping from indices to party names

    # Legacy properties for backward compatibility
    @property
    def n_sources(self) -> int:
        """Backward compatibility: n_source_parties."""
        return self.n_source_parties

    @property
    def n_destinations(self) -> int:
        """Backward compatibility: n_dest_parties."""
        return self.n_dest_parties

    @property
    def candidate_names(self) -> Dict[int, str]:
        """Backward compatibility: party_names."""
        return self.party_names


class ElectionType(Enum):
    """Types of elections using STV."""
    GENERAL = "general"
    LOCAL = "local"
    EUROPEAN = "european"
    SENATE = "senate"
    BYELECTION = "byelection"


class CandidateStatus(Enum):
    """Status of candidates during counting."""
    ACTIVE = "active"
    ELECTED = "elected"
    ELIMINATED = "eliminated"
    WITHDRAWN = "withdrawn"


@dataclass
class Candidate:
    """
    Represents a candidate in an STV election.
    """
    id: str
    name: str
    party: str
    constituency: str
    first_preferences: int = 0
    status: CandidateStatus = CandidateStatus.ACTIVE
    elected_round: Optional[int] = None
    eliminated_round: Optional[int] = None
    surplus_votes: int = 0
    final_tally: int = 0


@dataclass
class TransferRecord:
    """
    Represents a single transfer of votes between candidates.
    """
    from_candidate: str
    to_candidate: str
    vote_count: int
    transfer_value: float
    round_number: int
    transfer_type: str  # "elimination", "surplus", "distribution"
    constituency: str
    election_id: str


@dataclass
class CountRound:
    """
    Represents one round of counting in an STV election.
    """
    round_number: int
    constituency: str
    candidate_totals: Dict[str, int]
    transfers: List[TransferRecord]
    quota: int
    valid_poll: int
    elected_this_round: List[str] = field(default_factory=list)
    eliminated_this_round: List[str] = field(default_factory=list)


@dataclass
class ConstituencyResult:
    """
    Complete results for a single constituency.
    """
    constituency_id: str
    constituency_name: str
    num_seats: int
    quota: int
    valid_poll: int
    total_poll: int
    candidates: List[Candidate]
    count_rounds: List[CountRound]
    elected_candidates: List[str]
    transfer_matrix: Optional[Dict[str, Dict[str, int]]] = None


@dataclass
class ElectionData:
    """
    Complete election dataset with all constituencies and metadata.
    """
    election_id: str
    election_date: datetime
    election_type: ElectionType
    total_seats: int
    total_constituencies: int
    constituencies: List[ConstituencyResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw_data: Dict[str, Any]) -> "ElectionData":
        """
        Create ElectionData from raw scraped data.

        Parameters
        ----------
        raw_data : dict
            Raw data dictionary from scraper

        Returns
        -------
        ElectionData
            Structured election data object
        """
        # TODO: Implement conversion from raw scraped data
        # This is a placeholder implementation

        election_date = datetime.strptime(
            raw_data.get("election_date", "2020-01-01"), "%Y-%m-%d"
        )

        return cls(
            election_id=raw_data.get("election_id", "unknown"),
            election_date=election_date,
            election_type=ElectionType.GENERAL,
            total_seats=raw_data.get("total_seats", 0),
            total_constituencies=len(raw_data.get("constituencies", [])),
            constituencies=[],
            metadata=raw_data.get("metadata", {}),
        )

    def to_transfer_dataframe(self) -> pd.DataFrame:
        """
        Convert all transfer data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with all transfer records
        """
        transfers = []

        for constituency in self.constituencies:
            for count_round in constituency.count_rounds:
                for transfer in count_round.transfers:
                    transfers.append({
                        "election_id": self.election_id,
                        "constituency": transfer.constituency,
                        "round_number": transfer.round_number,
                        "from_candidate": transfer.from_candidate,
                        "to_candidate": transfer.to_candidate,
                        "vote_count": transfer.vote_count,
                        "transfer_value": transfer.transfer_value,
                        "transfer_type": transfer.transfer_type,
                    })

        return pd.DataFrame(transfers)

    def to_candidate_dataframe(self) -> pd.DataFrame:
        """
        Convert all candidate data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with all candidate information
        """
        candidates = []

        for constituency in self.constituencies:
            for candidate in constituency.candidates:
                candidates.append({
                    "election_id": self.election_id,
                    "constituency": candidate.constituency,
                    "candidate_id": candidate.id,
                    "name": candidate.name,
                    "party": candidate.party,
                    "first_preferences": candidate.first_preferences,
                    "status": candidate.status.value,
                    "elected_round": candidate.elected_round,
                    "eliminated_round": candidate.eliminated_round,
                    "final_tally": candidate.final_tally,
                })

        return pd.DataFrame(candidates)

    def get_transfer_matrix(self, constituency: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """
        Extract transfer matrix (count data) for analysis.

        Parameters
        ----------
        constituency : str, optional
            Specific constituency (if None, combines all)

        Returns
        -------
        dict
            Nested dictionary of from_candidate -> to_candidate -> total_transfers
        """
        transfer_matrix = {}
        transfers_df = self.to_transfer_dataframe()

        if constituency:
            transfers_df = transfers_df[transfers_df["constituency"] == constituency]

        # Aggregate transfers by from/to candidate
        grouped = transfers_df.groupby(["from_candidate", "to_candidate"])["vote_count"].sum()

        for (from_cand, to_cand), count in grouped.items():
            if from_cand not in transfer_matrix:
                transfer_matrix[from_cand] = {}
            transfer_matrix[from_cand][to_cand] = int(count)

        return transfer_matrix

    def summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the election.

        Returns
        -------
        dict
            Dictionary of summary statistics
        """
        candidates_df = self.to_candidate_dataframe()
        transfers_df = self.to_transfer_dataframe()

        return {
            "total_candidates": len(candidates_df),
            "total_transfers": len(transfers_df),
            "total_votes_transferred": transfers_df["vote_count"].sum(),
            "parties": candidates_df["party"].nunique(),
            "constituencies": candidates_df["constituency"].nunique(),
            "average_first_preferences": candidates_df["first_preferences"].mean(),
            "transfer_types": transfers_df["transfer_type"].value_counts().to_dict(),
        }


class PartyMapping:
    """
    Utility class for mapping candidates to party affiliations and ideological groups.
    """

    def __init__(self, party_config: Optional[Dict[str, Any]] = None):
        """
        Initialize party mapping.

        Parameters
        ----------
        party_config : dict, optional
            Configuration for party groupings and hierarchies
        """
        self.party_config = party_config or {}
        self._party_hierarchy = self._build_party_hierarchy()

    def _build_party_hierarchy(self) -> Dict[str, str]:
        """Build hierarchical party groupings for pooling."""
        # TODO: Implement party hierarchy building
        # This would group similar parties for hierarchical modeling
        return {}

    def get_party_group(self, party: str) -> str:
        """
        Get the party group for hierarchical modeling.

        Parameters
        ----------
        party : str
            Individual party name

        Returns
        -------
        str
            Party group identifier
        """
        return self._party_hierarchy.get(party, party)

    def similar_parties(self, party: str) -> List[str]:
        """
        Get list of ideologically similar parties.

        Parameters
        ----------
        party : str
            Reference party

        Returns
        -------
        list
            List of similar party names
        """
        # TODO: Implement similarity detection
        # Could be based on manifesto analysis, voting records, etc.
        return []


def validate_election_data(data: ElectionData) -> List[str]:
    """
    Validate election data for consistency and completeness.

    Parameters
    ----------
    data : ElectionData
        Election data to validate

    Returns
    -------
    list
        List of validation warnings/errors
    """
    warnings = []

    # Check basic data integrity
    if not data.constituencies:
        warnings.append("No constituency data found")

    for constituency in data.constituencies:
        # Check quota calculation
        expected_quota = int(constituency.valid_poll / (constituency.num_seats + 1)) + 1
        if abs(constituency.quota - expected_quota) > 1:
            warnings.append(
                f"Quota mismatch in {constituency.constituency_name}: "
                f"expected {expected_quota}, got {constituency.quota}"
            )

        # Check transfer consistency
        for round_data in constituency.count_rounds:
            total_in = sum(t.vote_count for t in round_data.transfers)
            if total_in == 0:
                continue

            # Check for negative transfers
            negative_transfers = [t for t in round_data.transfers if t.vote_count < 0]
            if negative_transfers:
                warnings.append(
                    f"Negative transfers found in {constituency.constituency_name} round {round_data.round_number}"
                )

    return warnings