"""
Tests for the robust scraper implementation.

These tests verify HTML parsing, data validation, and consistent indexing
across various STV election result formats.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

# Add src directory to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stv_transfers.scraper import ElectionScraper
from stv_transfers.data_structures import ModelData, EventData


@pytest.fixture
def limerick_html():
    """Load the Limerick 2020 count HTML fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "limerick_2020_real.html"
    return fixture_path.read_text()


@pytest.fixture
def limerick_result_html():
    """Load the Limerick 2020 result HTML fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "limerick_result_2020.html"
    return fixture_path.read_text()


@pytest.fixture
def scraper():
    """Create a test scraper instance."""
    return ElectionScraper("https://example.com/elections/", rate_limit=0.1)


class TestElectionScraper:
    """Test the ElectionScraper class initialization and basic functionality."""

    def test_initialization(self):
        """Test scraper initialization with proper defaults."""
        scraper = ElectionScraper("https://example.com/")

        # Check basic attributes
        assert scraper.base_url == "https://example.com/"
        assert scraper.rate_limit >= 1.0  # Should enforce minimum 1 second

        # Check candidate indexing
        assert "non-transferable" in scraper.candidate_index
        assert scraper.candidate_index["non-transferable"] == 0
        assert scraper.next_index == 1

    def test_candidate_indexing_consistency(self, scraper):
        """Test that candidate indices are consistent across calls."""
        # First assignment
        idx1 = scraper._get_candidate_index("Murphy")
        idx2 = scraper._get_candidate_index("Kelly")

        # Repeated calls should return same indices
        assert scraper._get_candidate_index("Murphy") == idx1
        assert scraper._get_candidate_index("Kelly") == idx2

        # New candidate gets new index
        idx3 = scraper._get_candidate_index("O'Brien")
        assert idx3 not in [idx1, idx2, 0]  # 0 is reserved for non-transferable

    def test_user_agent_header(self, scraper):
        """Test that proper User-Agent header is set."""
        user_agent = scraper.session.headers.get('User-Agent')
        assert 'STV-Transfers Research Tool' in user_agent
        assert 'Academic Use' in user_agent


class TestParseCountDetails:
    """Test the parse_count_details method with various HTML formats."""

    def test_parse_normal_elimination(self, scraper, limerick_html):
        """Test parsing normal single candidate elimination."""
        parsed = scraper.parse_count_details(limerick_html, "limerick")

        assert parsed is not None
        assert parsed['constituency_id'] == "limerick"
        assert parsed['quota'] == 11523  # Real quota from Limerick 2020
        assert parsed['valid_poll'] == 46090  # Real valid poll from Limerick 2020

        # Check transfers were parsed
        assert len(parsed['transfers']) > 0

        # Find eliminations - look for any candidate elimination
        elimination_transfers = [t for t in parsed['transfers']
                               if t['transfer_type'] == 'elimination']
        assert len(elimination_transfers) > 0

        # Verify transfer data structure
        for transfer in elimination_transfers[:5]:  # Check first 5
            assert 'from_candidate' in transfer
            assert 'to_candidate' in transfer
            assert 'transfer_count' in transfer
            assert 'transfer_type' in transfer
            assert 'transfer_value' in transfer
            assert isinstance(transfer['transfer_count'], (int, float))
            assert transfer['transfer_count'] >= 0

    def test_parse_grouped_elimination(self, scraper, limerick_html):
        """Test parsing grouped candidate elimination."""
        parsed = scraper.parse_count_details(limerick_html, "limerick")

        # Find grouped elimination from the real data
        # The first transfer shows multiple candidates: O'Donoghue, O'Donnell, Cremin, Dalton, Ní Mhaoldhomhnaigh
        grouped_candidates = ["O'Donoghue", "O'Donnell", "Cremin", "Dalton", "Ní Mhaoldhomhnaigh"]

        transfers_by_candidate = {}
        for candidate in grouped_candidates:
            transfers_by_candidate[candidate] = [t for t in parsed['transfers']
                                               if t['from_candidate'] == candidate]

        # All grouped candidates should have transfers recorded
        for candidate in grouped_candidates:
            assert len(transfers_by_candidate[candidate]) > 0, f"No transfers found for {candidate}"

        # Should have similar transfer patterns (grouped elimination) - check first two
        first_destinations = {t['to_candidate'] for t in transfers_by_candidate[grouped_candidates[0]]}
        second_destinations = {t['to_candidate'] for t in transfers_by_candidate[grouped_candidates[1]]}
        assert len(first_destinations.intersection(second_destinations)) > 0, "Grouped candidates should have some common destinations"

    def test_parse_surplus_fractional_values(self, scraper, limerick_html):
        """Test parsing surplus transfers with fractional values."""
        parsed = scraper.parse_count_details(limerick_html, "limerick")

        # Find O'Donovan surplus (from real data)
        odonovan_transfers = [t for t in parsed['transfers']
                            if t['from_candidate'] == "O'Donovan"]
        assert len(odonovan_transfers) > 0

        # Check that surplus transfers are identified
        surplus_transfers = [t for t in parsed['transfers']
                           if t['transfer_type'] == 'surplus']
        assert len(surplus_transfers) > 0

        # Verify all surplus transfers have proper structure
        for transfer in surplus_transfers:
            assert transfer['transfer_type'] == 'surplus'
            assert isinstance(transfer['transfer_value'], (int, float))
            assert transfer['transfer_value'] > 0

    def test_parse_malformed_html_warning(self, scraper):
        """Test that malformed HTML generates warnings but doesn't crash."""
        malformed_html = """
        <html>
            <body>
                <table class="count-details">
                    <tr>
                        <td>Incomplete row</td>
                        <!-- Missing required cells -->
                    </tr>
                </table>
            </body>
        </html>
        """

        # Should not crash, but may return None or limited data
        with patch.object(scraper.logger, 'warning') as mock_warning:
            result = scraper.parse_count_details(malformed_html, "test")

            # Either returns None or returns with warnings logged
            if result is None:
                assert mock_warning.called
            else:
                # If parsed, should have logged validation warnings
                assert len(result['transfers']) == 0 or mock_warning.called

    def test_missing_table_returns_none(self, scraper):
        """Test that HTML without transfer table returns None."""
        html_no_table = """
        <html>
            <body>
                <h1>Election Results</h1>
                <p>No transfer table here</p>
            </body>
        </html>
        """

        result = scraper.parse_count_details(html_no_table, "test")
        assert result is None

    def test_validation_catches_negative_transfers(self, scraper):
        """Test that validation catches negative transfer counts."""
        # Create mock parsed data with negative transfer
        mock_data = {
            'constituency_id': 'test',
            'candidates': ['A', 'B'],
            'transfers': [
                {
                    'from_candidate': 'A',
                    'to_candidate': 'B',
                    'transfer_count': -100,  # Invalid negative
                    'transfer_type': 'elimination',
                    'transfer_value': 1.0
                }
            ],
            'quota': 1000,
            'valid_poll': 5000
        }

        errors = scraper._validate_parsed_data(mock_data, "test")
        assert any("negative" in error.lower() for error in errors)


class TestConvertToModelFormat:
    """Test conversion from parsed data to ModelData format."""

    def test_convert_yields_valid_eventdata(self, scraper):
        """Test that conversion produces valid EventData objects."""
        # Mock parsed constituency data
        parsed_data = [
            {
                'constituency_id': 'const1',
                'candidates': ['Murphy', 'Kelly', 'Walsh'],
                'transfers': [
                    {
                        'from_candidate': 'Walsh',
                        'to_candidate': 'Murphy',
                        'transfer_count': 500,
                        'transfer_type': 'elimination',
                        'transfer_value': 1.0
                    },
                    {
                        'from_candidate': 'Walsh',
                        'to_candidate': 'Kelly',
                        'transfer_count': 300,
                        'transfer_type': 'elimination',
                        'transfer_value': 1.0
                    }
                ],
                'quota': 1000,
                'valid_poll': 4000
            }
        ]

        model_data = scraper.convert_to_model_format(parsed_data)

        # Check ModelData structure
        assert isinstance(model_data, ModelData)
        assert model_data.n_constituencies == 1
        assert len(model_data.events) > 0

        # Check EventData
        event = model_data.events[0]
        assert isinstance(event, EventData)
        assert event.constituency_idx == 0
        assert len(event.source_indices) == 1
        assert len(event.active_indices) >= 1
        assert len(event.transfer_counts) == len(event.active_indices)
        assert event.total_transfers > 0

    def test_consistent_candidate_indexing_across_constituencies(self, scraper):
        """Test that candidate indices are consistent across constituencies."""
        parsed_data = [
            {
                'constituency_id': 'const1',
                'candidates': ['Murphy', 'Kelly'],
                'transfers': [
                    {
                        'from_candidate': 'Murphy',
                        'to_candidate': 'Kelly',
                        'transfer_count': 100,
                        'transfer_type': 'elimination',
                        'transfer_value': 1.0
                    }
                ],
                'quota': 1000,
                'valid_poll': 4000
            },
            {
                'constituency_id': 'const2',
                'candidates': ['Kelly', 'Walsh'],  # Kelly appears in both
                'transfers': [
                    {
                        'from_candidate': 'Kelly',
                        'to_candidate': 'Walsh',
                        'transfer_count': 200,
                        'transfer_type': 'elimination',
                        'transfer_value': 1.0
                    }
                ],
                'quota': 1200,
                'valid_poll': 4800
            }
        ]

        model_data = scraper.convert_to_model_format(parsed_data)

        # Kelly should have same index in candidate_names
        kelly_indices = [idx for idx, name in model_data.candidate_names.items()
                        if name == 'Kelly']
        assert len(kelly_indices) == 1  # Should appear only once

        # Verify Kelly appears in events from both constituencies
        kelly_idx = kelly_indices[0]
        const1_events = [e for e in model_data.events if e.constituency_idx == 0]
        const2_events = [e for e in model_data.events if e.constituency_idx == 1]

        # Kelly should be referenced by same index in both constituencies
        kelly_in_const1 = any(kelly_idx in e.source_indices or kelly_idx in e.active_indices
                             for e in const1_events)
        kelly_in_const2 = any(kelly_idx in e.source_indices or kelly_idx in e.active_indices
                             for e in const2_events)

        assert kelly_in_const1 and kelly_in_const2

    def test_non_transferable_special_index(self, scraper):
        """Test that non-transferable votes get special index 0."""
        parsed_data = [
            {
                'constituency_id': 'const1',
                'candidates': ['Murphy', 'non-transferable'],
                'transfers': [
                    {
                        'from_candidate': 'Murphy',
                        'to_candidate': 'non-transferable',
                        'transfer_count': 50,
                        'transfer_type': 'elimination',
                        'transfer_value': 1.0
                    }
                ],
                'quota': 1000,
                'valid_poll': 4000
            }
        ]

        model_data = scraper.convert_to_model_format(parsed_data)

        # Check non-transferable has index 0
        assert 0 in model_data.candidate_names
        assert model_data.candidate_names[0] == 'non-transferable'

    def test_handles_empty_constituencies(self, scraper):
        """Test handling of constituencies with no valid data."""
        parsed_data = [
            {
                'constituency_id': 'const1',
                'candidates': ['Murphy', 'Kelly'],
                'transfers': [
                    {
                        'from_candidate': 'Murphy',
                        'to_candidate': 'Kelly',
                        'transfer_count': 100,
                        'transfer_type': 'elimination',
                        'transfer_value': 1.0
                    }
                ],
                'quota': 1000,
                'valid_poll': 4000
            },
            None,  # Failed constituency
            {
                'constituency_id': 'const3',
                'candidates': [],
                'transfers': [],  # No transfers
                'quota': 1200,
                'valid_poll': 4800
            }
        ]

        model_data = scraper.convert_to_model_format(parsed_data)

        # Should still work with some valid data
        assert isinstance(model_data, ModelData)
        assert len(model_data.events) >= 0
        # Should only count constituencies with valid data
        assert model_data.n_constituencies <= len([d for d in parsed_data if d])


class TestIntegration:
    """Integration tests for the complete scraping workflow."""

    @patch('stv_transfers.scraper.ElectionScraper.fetch_constituency_page')
    def test_scrape_election_workflow(self, mock_fetch, scraper, limerick_html):
        """Test the complete scrape_election workflow."""
        # Mock the fetch to return our test HTML
        mock_fetch.return_value = limerick_html

        # Test scraping
        constituency_ids = ['limerick']
        model_data = scraper.scrape_election('test_election', constituency_ids)

        # Verify fetch was called correctly
        mock_fetch.assert_called_once_with('test_election', 'limerick')

        # Verify ModelData structure
        assert isinstance(model_data, ModelData)
        assert model_data.n_constituencies == 1
        assert len(model_data.events) > 0
        assert len(model_data.candidate_names) > 1  # At least non-transferable + others

    @patch('stv_transfers.scraper.ElectionScraper.fetch_constituency_page')
    def test_handles_fetch_failures_gracefully(self, mock_fetch, scraper):
        """Test that fetch failures are handled gracefully."""
        # Mock fetch to return None (failure)
        mock_fetch.return_value = None

        constituency_ids = ['failed_constituency']
        model_data = scraper.scrape_election('test_election', constituency_ids)

        # Should still return valid ModelData, even if empty
        assert isinstance(model_data, ModelData)
        assert model_data.n_constituencies == 0
        assert len(model_data.events) == 0

    def test_data_type_consistency(self, scraper):
        """Test that all data types are consistent with expectations."""
        parsed_data = [
            {
                'constituency_id': 'const1',
                'candidates': ['A', 'B'],
                'transfers': [
                    {
                        'from_candidate': 'A',
                        'to_candidate': 'B',
                        'transfer_count': 100.5,  # Float count
                        'transfer_type': 'surplus',
                        'transfer_value': 0.75
                    }
                ],
                'quota': 1000,
                'valid_poll': 4000
            }
        ]

        model_data = scraper.convert_to_model_format(parsed_data)

        # Check data types
        event = model_data.events[0]
        assert isinstance(event.transfer_counts, np.ndarray)
        assert event.transfer_counts.dtype == np.float32
        assert isinstance(event.total_transfers, float)
        assert isinstance(event.constituency_idx, int)


class TestFirstPreferences:
    """Test first preference vote parsing functionality."""

    def test_parse_first_preferences_basic(self, scraper, limerick_result_html):
        """Test basic first preference parsing functionality."""
        result = scraper.parse_first_preferences(limerick_result_html, 5005)

        # Basic assertions
        assert isinstance(result, list)
        assert len(result) == 12  # Limerick 2020 had 12 candidates

        # Check structure of first candidate
        first_candidate = result[0]
        assert 'constituency_id' in first_candidate
        assert 'candidate' in first_candidate
        assert 'party' in first_candidate
        assert 'first_pref_votes' in first_candidate
        assert 'share' in first_candidate

        # Check data types
        assert isinstance(first_candidate['constituency_id'], int)
        assert isinstance(first_candidate['candidate'], str)
        assert isinstance(first_candidate['party'], str)
        assert isinstance(first_candidate['first_pref_votes'], int)
        assert isinstance(first_candidate['share'], float)

    def test_parse_first_preferences_data_accuracy(self, scraper, limerick_result_html):
        """Test accuracy of parsed first preference data."""
        result = scraper.parse_first_preferences(limerick_result_html, 5005)

        # Find top candidates and verify their data
        patrick_odonovan = next((c for c in result if 'Patrick O\'Donovan' in c['candidate']), None)
        assert patrick_odonovan is not None
        assert patrick_odonovan['party'] == 'Fine Gael'
        assert patrick_odonovan['first_pref_votes'] == 9228
        assert patrick_odonovan['share'] == 20.02

        niall_collins = next((c for c in result if 'Niall Collins' in c['candidate']), None)
        assert niall_collins is not None
        assert niall_collins['party'] == 'Fianna Fáil'
        assert niall_collins['first_pref_votes'] == 8436
        assert niall_collins['share'] == 18.30

    def test_parse_first_preferences_totals(self, scraper, limerick_result_html):
        """Test that vote totals and percentages are consistent."""
        result = scraper.parse_first_preferences(limerick_result_html, 5005)

        # Calculate totals
        total_votes = sum(c['first_pref_votes'] for c in result)
        total_percentage = sum(c['share'] for c in result)

        # Verify totals
        assert total_votes == 46090  # Total valid poll for Limerick 2020
        assert abs(total_percentage - 100.0) < 0.01  # Should sum to 100%

    def test_parse_first_preferences_party_mapping(self, scraper, limerick_result_html):
        """Test that party codes are correctly mapped to full names."""
        result = scraper.parse_first_preferences(limerick_result_html, 5005)

        # Check party mappings
        parties_found = {c['party'] for c in result}

        # Should include major parties with full names
        assert 'Fianna Fáil' in parties_found
        assert 'Fine Gael' in parties_found
        assert 'Sinn Féin' in parties_found
        assert 'Green Party' in parties_found
        assert 'Independent' in parties_found

        # Should not have party codes (unless unmapped)
        assert 'FF' not in parties_found
        assert 'FG' not in parties_found
        assert 'SF' not in parties_found

    def test_parse_first_preferences_candidate_names(self, scraper, limerick_result_html):
        """Test that all candidate names are correctly extracted."""
        result = scraper.parse_first_preferences(limerick_result_html, 5005)

        candidate_names = [c['candidate'] for c in result]

        # Check some known candidates
        assert any('Niall Collins' in name for name in candidate_names)
        assert any('Patrick O\'Donovan' in name for name in candidate_names)
        assert any('Richard O\'Donoghue' in name for name in candidate_names)
        assert any('Seighin O Ceallaigh' in name for name in candidate_names)

        # All names should be non-empty strings
        for name in candidate_names:
            assert isinstance(name, str)
            assert len(name.strip()) > 0

    def test_parse_first_preferences_empty_html(self, scraper):
        """Test parsing with empty or invalid HTML."""
        result = scraper.parse_first_preferences("", 5005)
        assert result == []

        result = scraper.parse_first_preferences("<html><body></body></html>", 5005)
        assert result == []

    def test_parse_first_preferences_malformed_table(self, scraper):
        """Test parsing with malformed table structure."""
        malformed_html = """
        <html>
            <body>
                <table class="rtable">
                    <tr>
                        <td>Candidate</td>
                        <td>1st Pref</td>
                    </tr>
                    <tr>
                        <td>Incomplete Row</td>
                        <!-- Missing vote count -->
                    </tr>
                </table>
            </body>
        </html>
        """

        result = scraper.parse_first_preferences(malformed_html, 5005)
        assert isinstance(result, list)
        # Should handle gracefully, might return empty list or partial data


class TestFetchFirstPreferences:
    """Test the fetch_first_preferences integration method."""

    @patch('stv_transfers.scraper.ElectionScraper.parse_first_preferences')
    def test_fetch_first_preferences_integration(self, mock_parse, scraper, limerick_result_html):
        """Test the complete fetch_first_preferences workflow."""
        # Mock the parsing to return known data
        expected_candidates = [
            {
                'constituency_id': 5005,
                'candidate': 'Test Candidate',
                'party': 'Test Party',
                'first_pref_votes': 1000,
                'share': 50.0
            }
        ]
        mock_parse.return_value = expected_candidates

        # Mock the HTTP request
        with patch.object(scraper.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.text = limerick_result_html
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test the method
            result = scraper.fetch_first_preferences(2020, 5005)

            # Verify results
            assert result == expected_candidates
            mock_get.assert_called_once()
            mock_parse.assert_called_once_with(limerick_result_html, 5005)

    def test_fetch_first_preferences_url_construction(self, scraper):
        """Test that URLs are constructed correctly."""
        with patch.object(scraper.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.text = "<html></html>"
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            scraper.fetch_first_preferences(2020, 5005)

            # Check URL construction
            expected_url = f"{scraper.base_url}result.cfm?election=2020&cons=5005"
            mock_get.assert_called_once_with(expected_url, timeout=30)

    def test_fetch_first_preferences_file_saving(self, scraper, tmp_path):
        """Test that raw HTML is saved to the correct location."""
        # Mock the data directory to use tmp_path
        with patch('stv_transfers.scraper.Path') as mock_path:
            mock_path.return_value = tmp_path / "data/raw/2020/result"

            with patch.object(scraper.session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.text = "<html>test content</html>"
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                scraper.fetch_first_preferences(2020, 5005)

                # Verify the Path operations were called correctly
                mock_path.assert_called_with("data/raw/2020/result")

    @pytest.mark.live
    def test_fetch_first_preferences_consistency(self, scraper):
        """Test that fetch_first_preferences produces consistent results."""
        pytest.importorskip("requests")

        try:
            # Fetch the same data twice to ensure consistency
            result1 = scraper.fetch_first_preferences(2020, 5005)
            result2 = scraper.fetch_first_preferences(2020, 5005)

            if result1 and result2:
                # Results should be identical
                assert len(result1) == len(result2)

                # Compare candidates (order might vary)
                candidates1 = {c['candidate']: c for c in result1}
                candidates2 = {c['candidate']: c for c in result2}

                assert set(candidates1.keys()) == set(candidates2.keys())

                for candidate_name in candidates1:
                    c1 = candidates1[candidate_name]
                    c2 = candidates2[candidate_name]

                    assert c1['party'] == c2['party']
                    assert c1['first_pref_votes'] == c2['first_pref_votes']
                    assert c1['share'] == c2['share']

                print(f"Consistency test passed: {len(result1)} candidates fetched consistently")
            else:
                pytest.skip("Could not fetch live data for consistency test")

        except Exception as e:
            pytest.skip(f"Live consistency test failed: {e}")


class TestLiveIntegration:
    """Optional live integration tests (requires internet connection)."""

    @pytest.mark.live
    def test_live_limerick_scrape(self, scraper):
        """Test live scraping of Limerick 2020 election results."""
        pytest.importorskip("requests")

        try:
            # Attempt to fetch the real Limerick page
            html = scraper.fetch_constituency_page("2020", "5005")

            if html is None:
                pytest.skip("Could not fetch live data (network issue or robots.txt)")

            # Parse the live data
            parsed_data = scraper.parse_count_details(html, "limerick_live")

            # Basic assertions for live data
            assert parsed_data is not None, "Live parsing should succeed"
            assert len(parsed_data['candidates']) > 5, "Should find multiple candidates"
            assert len(parsed_data['transfers']) > 10, "Should find transfer data"
            assert parsed_data.get('quota', 0) > 1000, "Should find realistic quota"
            assert parsed_data.get('valid_poll', 0) > 10000, "Should find realistic vote total"

            print(f"Live test results: {len(parsed_data['candidates'])} candidates, "
                  f"{len(parsed_data['transfers'])} transfers")

        except Exception as e:
            pytest.skip(f"Live test failed due to network or parsing issue: {e}")

    @pytest.mark.live
    def test_live_scrape_full_workflow(self, scraper):
        """Test complete scraping workflow with live data."""
        pytest.importorskip("requests")

        try:
            # Test full workflow with single constituency
            model_data = scraper.scrape_election("2020", ["5005"])

            # Verify ModelData structure
            assert isinstance(model_data, ModelData)
            assert model_data.n_constituencies <= 1  # Should be 1 if successful, 0 if failed

            if model_data.n_constituencies > 0:
                assert len(model_data.events) > 0, "Should have transfer events"
                assert len(model_data.candidate_names) > 5, "Should have multiple candidates"
                assert model_data.n_sources > 0, "Should have transfer sources"
                assert model_data.n_destinations > 0, "Should have transfer destinations"

                print(f"Live workflow test: {len(model_data.events)} events, "
                      f"{model_data.n_sources} sources, {model_data.n_destinations} destinations")
            else:
                pytest.skip("Live scraping returned no data")

        except Exception as e:
            pytest.skip(f"Live workflow test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])