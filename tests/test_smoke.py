"""
Smoke tests for the stv_transfers package.

These tests verify basic functionality and imports work correctly.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_package_imports():
    """Test that all main modules can be imported."""
    try:
        import stv_transfers
        assert stv_transfers.__version__ == "0.1.0"
    except ImportError as e:
        pytest.fail(f"Failed to import stv_transfers: {e}")


def test_data_structures_import():
    """Test data structures module imports."""
    try:
        from stv_transfers import data_structures

        # Test enum imports
        assert hasattr(data_structures, "ElectionType")
        assert hasattr(data_structures, "CandidateStatus")

        # Test class imports
        assert hasattr(data_structures, "Candidate")
        assert hasattr(data_structures, "ElectionData")
        assert hasattr(data_structures, "TransferRecord")

    except ImportError as e:
        pytest.fail(f"Failed to import data_structures: {e}")


def test_model_import():
    """Test model module imports (may skip if NumPyro not available)."""
    try:
        from stv_transfers import model
        assert hasattr(model, "TransferModel")

    except ImportError as e:
        if "NumPyro" in str(e) or "JAX" in str(e):
            pytest.skip("NumPyro/JAX not available, skipping model tests")
        else:
            pytest.fail(f"Failed to import model: {e}")


def test_scraper_import():
    """Test scraper module imports."""
    try:
        from stv_transfers import scraper
        assert hasattr(scraper, "ElectionScraper")
        assert hasattr(scraper, "fetch_election_results")

    except ImportError as e:
        pytest.fail(f"Failed to import scraper: {e}")


def test_simulator_import():
    """Test simulator module imports."""
    try:
        from stv_transfers import simulator
        assert hasattr(simulator, "STVSimulator")
        assert hasattr(simulator, "Candidate")
        assert hasattr(simulator, "TransferEvent")

    except ImportError as e:
        pytest.fail(f"Failed to import simulator: {e}")


def test_diagnostics_import():
    """Test diagnostics module imports."""
    try:
        from stv_transfers import diagnostics
        assert hasattr(diagnostics, "check_rhat")
        assert hasattr(diagnostics, "effective_sample_size")

    except ImportError as e:
        pytest.fail(f"Failed to import diagnostics: {e}")


def test_basic_data_structure_creation():
    """Test basic data structure instantiation."""
    from stv_transfers.data_structures import Candidate, CandidateStatus, ElectionType
    from datetime import datetime

    # Test candidate creation
    candidate = Candidate(
        id="test_001",
        name="Test Candidate",
        party="Test Party",
        constituency="Test Constituency",
        first_preferences=1000
    )

    assert candidate.id == "test_001"
    assert candidate.name == "Test Candidate"
    assert candidate.status == CandidateStatus.ACTIVE
    assert candidate.first_preferences == 1000


def test_election_data_creation():
    """Test ElectionData creation from raw data."""
    from stv_transfers.data_structures import ElectionData

    raw_data = {
        "election_id": "test_2020",
        "election_date": "2020-02-08",
        "total_seats": 160,
        "constituencies": [],
        "metadata": {"test": True}
    }

    election_data = ElectionData.from_raw(raw_data)

    assert election_data.election_id == "test_2020"
    assert election_data.total_seats == 160
    assert election_data.metadata["test"] is True


def test_simulator_basic_functionality():
    """Test basic simulator functionality without full simulation."""
    from stv_transfers.simulator import STVSimulator, Candidate

    simulator = STVSimulator(num_seats=3, random_seed=42)
    assert simulator.num_seats == 3

    # Test candidate creation
    candidate = Candidate(
        id="sim_001",
        name="Sim Candidate",
        party="Sim Party",
        first_preferences=500
    )

    assert candidate.id == "sim_001"
    assert candidate.first_preferences == 500
    assert not candidate.is_elected


@pytest.mark.parametrize("module_name", [
    "data_structures",
    "scraper",
    "simulator",
    "diagnostics"
])
def test_all_modules_importable(module_name):
    """Parametrized test for all module imports."""
    try:
        exec(f"from stv_transfers import {module_name}")
    except ImportError as e:
        if module_name == "model" and ("NumPyro" in str(e) or "JAX" in str(e)):
            pytest.skip(f"Skipping {module_name} due to missing NumPyro/JAX")
        else:
            pytest.fail(f"Failed to import {module_name}: {e}")


def test_package_metadata():
    """Test package metadata is accessible."""
    import stv_transfers

    assert hasattr(stv_transfers, "__version__")
    assert hasattr(stv_transfers, "__author__")
    assert hasattr(stv_transfers, "__email__")

    # Test version format
    version = stv_transfers.__version__
    assert isinstance(version, str)
    assert len(version.split(".")) >= 3  # Major.minor.patch at minimum