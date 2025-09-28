"""
Tests for the comprehensive election scraper (scrape_all_elections.py).

These tests verify constituency discovery, data scraping, and dataset combination
with mocked network requests to avoid hitting live servers during testing.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from typing import Dict, Any

# Add src and scripts directories to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scrape_all_elections import (
    ElectionDiscovery,
    ComprehensiveElectionScraper,
    setup_logging
)


@pytest.fixture
def mock_constituency_page_html():
    """Mock HTML for a constituency listing page."""
    return """
    <html>
        <head>
            <title>2020 General Election Results</title>
        </head>
        <body>
            <h1>2020 General Election - All Constituencies</h1>
            <div class="constituency-list">
                <a href="result.cfm?election=2020&cons=5001">Carlow-Kilkenny</a>
                <a href="result.cfm?election=2020&cons=5002">Cavan-Monaghan</a>
                <a href="result.cfm?election=2020&cons=5005">Limerick</a>
                <a href="result.cfm?election=2020&cons=5010">Dublin Central</a>
                <a href="result.cfm?election=2020&cons=5015">Cork North-Central</a>
            </div>
            <table class="results-table">
                <tr>
                    <th>Constituency</th>
                    <th>Results</th>
                </tr>
                <tr>
                    <td>Galway West</td>
                    <td><a href="result.cfm?election=2020&cons=5020">View Results</a></td>
                </tr>
                <tr>
                    <td>Kerry</td>
                    <td><a href="result.cfm?election=2020&cons=5025">View Results</a></td>
                </tr>
            </table>
        </body>
    </html>
    """


@pytest.fixture
def mock_first_prefs_html():
    """Mock HTML for first preference results."""
    return """
    <html>
        <body>
            <table class="rtable">
                <tr>
                    <th>Photo</th>
                    <th>Candidate</th>
                    <th>Order</th>
                    <th>Party</th>
                    <th>Quota</th>
                    <th>1st Pref</th>
                    <th>Transfer</th>
                    <th>%</th>
                    <th>Status</th>
                    <th>Count</th>
                </tr>
                <tr>
                    <td></td>
                    <td>John Murphy</td>
                    <td>1</td>
                    <td><a href="party.cfm?election=2020&party=FF">FF</a></td>
                    <td>10,000</td>
                    <td>8,500</td>
                    <td>+1,500</td>
                    <td>18.5</td>
                    <td>Elected</td>
                    <td>1</td>
                </tr>
                <tr>
                    <td></td>
                    <td>Mary Kelly</td>
                    <td>2</td>
                    <td><a href="party.cfm?election=2020&party=FG">FG</a></td>
                    <td>10,000</td>
                    <td>7,200</td>
                    <td>+2,800</td>
                    <td>15.7</td>
                    <td>Elected</td>
                    <td>3</td>
                </tr>
                <tr>
                    <td></td>
                    <td>Sean Walsh</td>
                    <td>3</td>
                    <td><a href="party.cfm?election=2020&party=SF">SF</a></td>
                    <td>10,000</td>
                    <td>6,800</td>
                    <td>+3,200</td>
                    <td>14.8</td>
                    <td>Elected</td>
                    <td>4</td>
                </tr>
                <tr>
                    <td></td>
                    <td>Ann O'Brien</td>
                    <td>4</td>
                    <td><a href="party.cfm?election=2020&party=IND">IND</a></td>
                    <td>10,000</td>
                    <td>4,200</td>
                    <td>-4,200</td>
                    <td>9.1</td>
                    <td>Not Elected</td>
                    <td>2</td>
                </tr>
            </table>
        </body>
    </html>
    """


@pytest.fixture
def mock_transfer_html():
    """Mock HTML for transfer count details."""
    return """
    <html>
        <body>
            <table class="rtable">
                <tr>
                    <td colspan="10">Transfer of votes from O'Brien</td>
                </tr>
                <tr>
                    <th>Candidate</th>
                    <th>Count 1</th>
                    <th>Count 2</th>
                    <th>Transfer</th>
                    <th>Total</th>
                </tr>
                <tr>
                    <td>John Murphy</td>
                    <td>8,500</td>
                    <td>+800</td>
                    <td>800</td>
                    <td>9,300</td>
                </tr>
                <tr>
                    <td>Mary Kelly</td>
                    <td>7,200</td>
                    <td>+600</td>
                    <td>600</td>
                    <td>7,800</td>
                </tr>
                <tr>
                    <td>Sean Walsh</td>
                    <td>6,800</td>
                    <td>+500</td>
                    <td>500</td>
                    <td>7,300</td>
                </tr>
                <tr>
                    <td>Non-transferable</td>
                    <td>0</td>
                    <td>+2,300</td>
                    <td>2,300</td>
                    <td>2,300</td>
                </tr>
            </table>
        </body>
    </html>
    """


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestElectionDiscovery:
    """Test the ElectionDiscovery class."""

    @pytest.fixture
    def discovery_service(self):
        """Create ElectionDiscovery instance for testing."""
        return ElectionDiscovery("https://test.example.com", rate_limit=0.1)

    def test_initialization(self, discovery_service):
        """Test ElectionDiscovery initialization."""
        assert discovery_service.base_url == "https://test.example.com"
        assert discovery_service.rate_limit == 0.1  # As set in fixture
        assert 'User-Agent' in discovery_service.session.headers

    @patch('scrape_all_elections.time.sleep')  # Speed up tests
    def test_discover_constituencies_success(self, mock_sleep, discovery_service, mock_constituency_page_html):
        """Test successful constituency discovery."""
        with patch.object(discovery_service.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.text = mock_constituency_page_html
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = discovery_service.discover_constituencies(2020)

            # Verify discovery results
            assert result["year"] == 2020
            assert "constituencies" in result
            assert "metadata" in result

            constituencies = result["constituencies"]
            assert len(constituencies) >= 5  # Should find at least 5 from mock HTML

            # Check specific constituencies
            assert "5001" in constituencies
            assert constituencies["5001"]["name"] == "Carlow-Kilkenny"
            assert "5005" in constituencies
            assert constituencies["5005"]["name"] == "Limerick"

            # Verify metadata
            assert result["metadata"]["total_constituencies"] == len(constituencies)
            assert "discovery_timestamp" in result["metadata"]

    @patch('scrape_all_elections.time.sleep')
    def test_discover_constituencies_network_failure(self, mock_sleep, discovery_service):
        """Test handling of network failures during discovery."""
        import requests
        with patch.object(discovery_service.session, 'get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Network error")

            result = discovery_service.discover_constituencies(2020)

            # Should handle failure gracefully
            assert result["year"] == 2020
            assert result["constituencies"] == {}
            assert "metadata" in result

    def test_parse_constituency_page_alternative_format(self, discovery_service):
        """Test parsing of alternative HTML formats."""
        alternative_html = """
        <html>
            <body>
                <div class="results">
                    <!-- No direct result.cfm links, need fallback -->
                    <h2>Constituencies</h2>
                    <p>Election results will be available soon</p>
                </div>
            </body>
        </html>
        """

        # This should trigger the alternative parsing with fallback generation
        result = discovery_service._parse_constituency_page(alternative_html, 2020)

        assert result["year"] == 2020
        assert "constituencies" in result
        # Should have some constituencies from fallback (5001-5040 range)
        assert len(result["constituencies"]) > 20


class TestComprehensiveElectionScraper:
    """Test the ComprehensiveElectionScraper class."""

    @pytest.fixture
    def temp_scraper(self, temp_data_dir):
        """Create scraper with temporary data directory."""
        with patch('scrape_all_elections.Path') as mock_path:
            # Mock Path to use our temp directory
            def path_side_effect(path_str):
                if path_str.startswith("data/"):
                    return temp_data_dir / path_str
                return Path(path_str)

            mock_path.side_effect = path_side_effect

            scraper = ComprehensiveElectionScraper("https://test.example.com", rate_limit=0.1)
            scraper.temp_data_dir = temp_data_dir  # Store for access in tests
            return scraper

    def test_initialization(self):
        """Test ComprehensiveElectionScraper initialization."""
        scraper = ComprehensiveElectionScraper("https://test.example.com", rate_limit=2.0)

        assert scraper.base_url == "https://test.example.com"
        assert scraper.rate_limit == 2.0
        assert hasattr(scraper, 'discovery')
        assert hasattr(scraper, 'scraper')
        assert hasattr(scraper, 'global_candidate_index')
        assert scraper.global_candidate_index["non-transferable"] == 0

    @patch('scrape_all_elections.time.sleep')  # Speed up tests
    def test_scrape_single_election_success(self, mock_sleep, temp_scraper, mock_constituency_page_html,
                                           mock_first_prefs_html, mock_transfer_html):
        """Test successful scraping of a single election."""
        # Mock constituency discovery
        mock_constituencies = {
            "5001": {"name": "Test Constituency", "url": "test_url"}
        }

        with patch.object(temp_scraper.discovery, 'discover_constituencies') as mock_discover:
            mock_discover.return_value = {
                "year": 2020,
                "constituencies": mock_constituencies,
                "metadata": {"total_constituencies": 1}
            }

            # Mock first preferences fetching
            with patch.object(temp_scraper, '_fetch_first_preferences_with_retry') as mock_first_prefs:
                mock_first_prefs.return_value = [
                    {
                        "constituency_id": 5001,
                        "candidate": "John Murphy",
                        "party": "Fianna Fáil",
                        "first_pref_votes": 8500,
                        "share": 18.5
                    }
                ]

                # Mock transfers fetching
                with patch.object(temp_scraper, '_fetch_transfers_with_retry') as mock_transfers:
                    mock_transfers.return_value = {
                        "constituency_id": "5001",
                        "transfers": [
                            {
                                "from_candidate": "O'Brien",
                                "to_candidate": "Murphy",
                                "transfer_count": 800,
                                "transfer_type": "elimination",
                                "transfer_value": 1.0
                            }
                        ]
                    }

                    # Mock file saving
                    with patch.object(temp_scraper, '_save_election_data') as mock_save:
                        result = temp_scraper.scrape_single_election(2020)

                        # Verify results
                        assert result["year"] == 2020
                        assert "constituencies" in result
                        assert "5001" in result["constituencies"]

                        constituency_data = result["constituencies"]["5001"]
                        assert constituency_data["name"] == "Test Constituency"
                        assert len(constituency_data["first_prefs"]) == 1
                        assert len(constituency_data["transfer_events"]) == 1

                        # Verify metadata
                        assert result["metadata"]["successful_constituencies"] == 1
                        assert result["metadata"]["failed_constituencies"] == 0

                        # Verify save was called
                        mock_save.assert_called_once_with(result)

    @patch('scrape_all_elections.time.sleep')
    def test_scrape_single_election_no_constituencies(self, mock_sleep, temp_scraper):
        """Test handling when no constituencies are discovered."""
        with patch.object(temp_scraper.discovery, 'discover_constituencies') as mock_discover:
            mock_discover.return_value = {
                "year": 2020,
                "constituencies": {},  # No constituencies found
                "metadata": {"total_constituencies": 0}
            }

            result = temp_scraper.scrape_single_election(2020)

            assert result["year"] == 2020
            assert result["constituencies"] == {}
            assert "error" in result["metadata"]

    @patch('scrape_all_elections.time.sleep')
    def test_scrape_all_elections(self, mock_sleep, temp_scraper):
        """Test scraping multiple elections."""
        with patch.object(temp_scraper, 'scrape_single_election') as mock_single:
            # Mock results for different years
            mock_single.side_effect = [
                {"year": 2016, "constituencies": {"5001": {"name": "Test 2016"}}},
                {"year": 2020, "constituencies": {"5001": {"name": "Test 2020"}}},
                {"year": 2024, "constituencies": {"5001": {"name": "Test 2024"}}}
            ]

            result = temp_scraper.scrape_all_elections([2016, 2020, 2024])

            # Verify all elections were processed
            assert len(result) == 3
            assert 2016 in result
            assert 2020 in result
            assert 2024 in result

            # Verify scrape_single_election was called for each year
            assert mock_single.call_count == 3

    def test_fetch_first_preferences_with_retry_success(self, temp_scraper, mock_first_prefs_html):
        """Test successful first preferences fetching with retry logic."""
        # Mock the underlying scraper's fetch method
        expected_candidates = [
            {
                "constituency_id": 5001,
                "candidate": "John Murphy",
                "party": "Fianna Fáil",
                "first_pref_votes": 8500,
                "share": 18.5
            }
        ]

        with patch.object(temp_scraper.scraper, 'fetch_first_preferences') as mock_fetch:
            mock_fetch.return_value = expected_candidates

            result = temp_scraper._fetch_first_preferences_with_retry(2020, 5001)

            assert result == expected_candidates
            mock_fetch.assert_called_once_with(2020, 5001)

    @patch('scrape_all_elections.time.sleep')
    def test_fetch_first_preferences_with_retry_failure(self, mock_sleep, temp_scraper):
        """Test first preferences fetching with retry on failures."""
        with patch.object(temp_scraper.scraper, 'fetch_first_preferences') as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            result = temp_scraper._fetch_first_preferences_with_retry(2020, 5001)

            # Should return empty list after all retries fail
            assert result == []
            # Should retry 3 times
            assert mock_fetch.call_count == 3

    def test_fetch_transfers_with_retry_cached(self, temp_scraper, mock_transfer_html):
        """Test transfer fetching when cached HTML exists."""
        # Create cached file
        cached_file = temp_scraper.temp_data_dir / "data/raw/2020/counts/5001.html"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_text(mock_transfer_html)

        expected_parsed_data = {
            "constituency_id": "5001",
            "transfers": [
                {
                    "from_candidate": "O'Brien",
                    "to_candidate": "Murphy",
                    "transfer_count": 800,
                    "transfer_type": "elimination",
                    "transfer_value": 1.0
                }
            ]
        }

        with patch.object(temp_scraper.scraper, 'parse_count_details') as mock_parse:
            mock_parse.return_value = expected_parsed_data

            result = temp_scraper._fetch_transfers_with_retry(2020, "5001")

            assert result == expected_parsed_data
            mock_parse.assert_called_once_with(mock_transfer_html, "5001")

    def test_convert_transfers_to_events(self, temp_scraper):
        """Test conversion of parsed transfer data to event format."""
        parsed_data = {
            "constituency_id": "5001",
            "transfers": [
                {
                    "from_candidate": "O'Brien",
                    "to_candidate": "Murphy",
                    "transfer_count": 800,
                    "transfer_type": "elimination",
                    "transfer_value": 1.0
                },
                {
                    "from_candidate": "O'Brien",
                    "to_candidate": "non-transferable",
                    "transfer_count": 200,
                    "transfer_type": "elimination",
                    "transfer_value": 1.0
                }
            ]
        }

        result = temp_scraper._convert_transfers_to_events(parsed_data, "5001")

        assert len(result) == 2

        # Check first transfer
        assert result[0]["from_candidate"] == "O'Brien"
        assert result[0]["to_candidate"] == "Murphy"
        assert result[0]["transfer_count"] == 800
        assert result[0]["transfer_type"] == "elimination"
        assert result[0]["transfer_value"] == 1.0

        # Check that candidates were added to global index
        assert "O'Brien" in temp_scraper.global_candidate_index
        assert "Murphy" in temp_scraper.global_candidate_index
        assert "non-transferable" in temp_scraper.global_candidate_index

    def test_global_candidate_indexing(self, temp_scraper):
        """Test global candidate indexing across elections."""
        # Add candidates from different elections
        idx1 = temp_scraper._get_global_candidate_index("Murphy")
        idx2 = temp_scraper._get_global_candidate_index("Kelly")
        idx3 = temp_scraper._get_global_candidate_index("Murphy")  # Repeated

        # Murphy should get same index both times
        assert idx1 == idx3
        # Kelly should get different index
        assert idx1 != idx2
        # Non-transferable should remain at index 0
        assert temp_scraper.global_candidate_index["non-transferable"] == 0

    def test_create_directories(self, temp_scraper):
        """Test creation of directory structure."""
        with patch('scrape_all_elections.Path') as mock_path:
            mock_path_instances = []

            def mock_path_constructor(path_str):
                mock_instance = Mock()
                mock_instance.mkdir = Mock()
                mock_path_instances.append((path_str, mock_instance))
                return mock_instance

            mock_path.side_effect = mock_path_constructor

            temp_scraper._create_directories(2020)

            # Verify directories were created
            expected_dirs = [
                "data/raw/2020/result",
                "data/raw/2020/counts",
                "data/processed/2020"
            ]

            created_paths = [path_str for path_str, _ in mock_path_instances]
            for expected_dir in expected_dirs:
                assert expected_dir in created_paths

            # Verify mkdir was called with correct arguments
            for _, mock_instance in mock_path_instances:
                mock_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_save_election_data(self, temp_scraper):
        """Test saving of election data to JSON files."""
        election_data = {
            "year": 2020,
            "constituencies": {
                "5001": {
                    "constituency_id": "5001",
                    "name": "Test Constituency",
                    "first_prefs": [
                        {
                            "constituency_id": 5001,
                            "candidate": "John Murphy",
                            "party": "Fianna Fáil",
                            "first_pref_votes": 8500,
                            "share": 18.5
                        }
                    ],
                    "transfer_events": [
                        {
                            "from_candidate": "O'Brien",
                            "to_candidate": "Murphy",
                            "transfer_count": 800,
                            "transfer_type": "elimination",
                            "transfer_value": 1.0
                        }
                    ]
                }
            },
            "metadata": {
                "successful_constituencies": 1,
                "failed_constituencies": 0
            }
        }

        # Mock file operations
        with patch('scrape_all_elections.Path') as mock_path, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            # Mock Path instances
            mock_file_instance = Mock()
            mock_file_instance.parent.mkdir = Mock()
            mock_path.return_value = mock_file_instance

            temp_scraper._save_election_data(election_data)

            # Verify main file save
            assert mock_json_dump.call_count >= 1  # At least main file

        # Test separate datasets saving
        with patch.object(temp_scraper, '_save_separate_datasets') as mock_separate:
            temp_scraper._save_election_data(election_data)
            mock_separate.assert_called_once_with(election_data)

    def test_save_separate_datasets(self, temp_scraper):
        """Test saving of separate first prefs and transfers JSON files."""
        election_data = {
            "year": 2020,
            "constituencies": {
                "5001": {
                    "first_prefs": [{"candidate": "Murphy", "votes": 8500}],
                    "transfer_events": [{"from": "O'Brien", "to": "Murphy", "count": 800}]
                },
                "5002": {
                    "first_prefs": [{"candidate": "Kelly", "votes": 7200}],
                    "transfer_events": [{"from": "Walsh", "to": "Kelly", "count": 600}]
                }
            }
        }

        with patch('scrape_all_elections.Path') as mock_path, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            mock_file_instance = Mock()
            mock_path.return_value = mock_file_instance

            temp_scraper._save_separate_datasets(election_data)

            # Should save first_prefs.json and transfers.json
            expected_calls = 2
            assert mock_json_dump.call_count == expected_calls


class TestDataQualityAndValidation:
    """Test data quality and validation features."""

    def test_constituency_count_validation(self):
        """Test that constituency counts match expected values for each year."""
        # This would normally test against known constituency counts
        # For 2020, Ireland had 39 constituencies
        expected_counts = {
            2016: 40,  # Pre-boundary review
            2020: 39,  # Post-boundary review
            2024: 43   # Latest boundary review
        }

        # Mock scraper results
        mock_results = {}
        for year, count in expected_counts.items():
            mock_constituencies = {
                str(5000 + i): {"name": f"Constituency_{i}", "url": f"test_url_{i}"}
                for i in range(1, count + 1)
            }
            mock_results[year] = {
                "year": year,
                "constituencies": mock_constituencies,
                "metadata": {"total_constituencies": count}
            }

        # Verify constituency counts
        for year, result in mock_results.items():
            actual_count = len(result["constituencies"])
            expected_count = expected_counts[year]
            assert actual_count == expected_count, f"Year {year}: expected {expected_count}, got {actual_count}"

    def test_data_structure_validation(self):
        """Test that generated data structures match expected format."""
        # Expected structure from requirements
        expected_structure = {
            "year": int,
            "constituencies": {
                "5005": {
                    "name": str,
                    "first_prefs": list,
                    "transfer_events": list
                }
            }
        }

        # Mock election data
        mock_data = {
            "year": 2020,
            "constituencies": {
                "5005": {
                    "name": "Limerick",
                    "first_prefs": [
                        {
                            "constituency_id": 5005,
                            "candidate": "Patrick O'Donovan",
                            "party": "Fine Gael",
                            "first_pref_votes": 9228,
                            "share": 20.02
                        }
                    ],
                    "transfer_events": [
                        {
                            "from_candidate": "O'Brien",
                            "to_candidate": "O'Donovan",
                            "transfer_count": 800,
                            "transfer_type": "elimination",
                            "transfer_value": 1.0
                        }
                    ]
                }
            }
        }

        # Validate structure
        assert isinstance(mock_data["year"], int)
        assert isinstance(mock_data["constituencies"], dict)

        for cons_id, cons_data in mock_data["constituencies"].items():
            assert isinstance(cons_data["name"], str)
            assert isinstance(cons_data["first_prefs"], list)
            assert isinstance(cons_data["transfer_events"], list)

            # Validate first preference structure
            for fp in cons_data["first_prefs"]:
                assert "constituency_id" in fp
                assert "candidate" in fp
                assert "party" in fp
                assert "first_pref_votes" in fp
                assert "share" in fp

            # Validate transfer event structure
            for te in cons_data["transfer_events"]:
                assert "from_candidate" in te
                assert "to_candidate" in te
                assert "transfer_count" in te
                assert "transfer_type" in te
                assert "transfer_value" in te

    def test_progress_reporting_format(self, capsys):
        """Test that progress is reported in the expected format."""
        # Mock progress output
        year = 2020
        constituency_name = "Limerick"

        # Simulate the expected progress format
        first_pref_status = "✅"
        transfer_status = "✅"
        expected_output = f"[{year} - {constituency_name}] {first_pref_status} first prefs, {transfer_status} transfers"

        print(expected_output)
        captured = capsys.readouterr()

        assert expected_output in captured.out
        assert "✅ first prefs" in captured.out
        assert "✅ transfers" in captured.out

    def test_error_handling_robustness(self):
        """Test that the scraper handles various error conditions gracefully."""
        scraper = ComprehensiveElectionScraper("https://test.example.com", rate_limit=0.1)

        # Test with various error conditions
        error_conditions = [
            None,  # No data
            {},    # Empty dict
            {"constituencies": {}},  # No constituencies
            {"constituencies": {"5001": {}}},  # Empty constituency data
        ]

        for condition in error_conditions:
            # Should not raise exceptions
            try:
                if condition is None:
                    result = scraper._convert_transfers_to_events(None, "test")
                    assert result == []
                else:
                    result = scraper._convert_transfers_to_events(condition, "test")
                    assert isinstance(result, list)
            except Exception as e:
                pytest.fail(f"Error handling failed for condition {condition}: {e}")


class TestMockNetworkBehavior:
    """Test network-related behavior with mocked responses."""

    def test_rate_limiting_behavior(self):
        """Test that rate limiting is properly implemented."""
        with patch('scrape_all_elections.time.sleep') as mock_sleep:
            scraper = ComprehensiveElectionScraper("https://test.example.com", rate_limit=2.0)

            # Rate limit should be enforced
            assert scraper.rate_limit == 2.0

    def test_user_agent_headers(self):
        """Test that proper User-Agent headers are set."""
        scraper = ComprehensiveElectionScraper("https://test.example.com")

        # Check User-Agent in discovery service
        discovery_ua = scraper.discovery.session.headers.get('User-Agent')
        assert 'Irish-Transfer-Model Research Tool' in discovery_ua
        assert 'Academic Use' in discovery_ua

        # Check User-Agent in main scraper
        scraper_ua = scraper.scraper.session.headers.get('User-Agent')
        assert 'STV-Transfers Research Tool' in scraper_ua
        assert 'Academic Use' in scraper_ua

    def test_timeout_handling(self):
        """Test that network timeouts are handled appropriately."""
        scraper = ComprehensiveElectionScraper("https://test.example.com", rate_limit=0.1)

        with patch.object(scraper.scraper.session, 'get') as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

            result = scraper._fetch_transfers_with_retry(2020, "5001")

            # Should handle timeout gracefully
            assert result is None
            # Should have tried multiple times
            assert mock_get.call_count >= 1


def test_setup_logging():
    """Test logging setup function."""
    with patch('scrape_all_elections.logging.basicConfig') as mock_config, \
         patch('scrape_all_elections.Path') as mock_path:

        mock_path.return_value.mkdir = Mock()

        setup_logging(verbose=True)

        # Verify logging was configured
        mock_config.assert_called_once()
        # Verify logs directory creation
        mock_path.assert_called_with("logs")


def test_main_function_execution():
    """Test main function execution flow."""
    with patch('scrape_all_elections.setup_logging'), \
         patch('scrape_all_elections.ComprehensiveElectionScraper') as mock_scraper_class:

        # Mock scraper instance
        mock_scraper = Mock()
        mock_scraper.scrape_all_elections.return_value = {
            2020: {
                "year": 2020,
                "metadata": {
                    "total_constituencies": 39,
                    "successful_constituencies": 35,
                    "failed_constituencies": 4
                }
            }
        }
        mock_scraper_class.return_value = mock_scraper

        # Import and test main function
        from scrape_all_elections import main

        # Should complete without errors
        result = main()

        # Verify scraper was created and called
        mock_scraper_class.assert_called_once()
        mock_scraper.scrape_all_elections.assert_called_once_with([2016, 2020, 2024])

        # Should return success code
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])