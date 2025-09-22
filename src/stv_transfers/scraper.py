"""
Data collection tools for STV electoral results.

This module provides functions for scraping and downloading electoral
data from various official sources, with a focus on Irish elections.
"""

from typing import List, Dict, Optional, Union, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import json
import re
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from .data_structures import ModelData, EventData


class ElectionScraper:
    """
    Robust scraper for STV electoral data with consistent candidate indexing.

    This class handles the collection of STV election results with proper
    data validation, robots.txt compliance, and consistent candidate indexing
    across constituencies.
    """

    def __init__(self, base_url: str, rate_limit: float = 1.0):
        """
        Initialize the scraper.

        Parameters
        ----------
        base_url : str
            Base URL for the electoral commission website
        rate_limit : float
            Minimum seconds between requests (be respectful!)
        """
        self.base_url = base_url
        self.rate_limit = max(rate_limit, 1.0)  # Minimum 1 second delay

        # Candidate indexing: consistent across constituencies
        # Index 0 is reserved for non-transferable votes
        self.candidate_index: Dict[str, int] = {"non-transferable": 0}
        self.next_index = 1

        # Session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'STV-Transfers Research Tool (Academic Use) - Contact: research@university.edu'
        })

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Check robots.txt compliance
        self._check_robots_txt()

    def _check_robots_txt(self) -> None:
        """Check robots.txt compliance."""
        try:
            robots_url = urljoin(self.base_url, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            user_agent = self.session.headers.get('User-Agent', '*')
            if not rp.can_fetch(user_agent, self.base_url):
                self.logger.warning(f"robots.txt disallows access to {self.base_url}")
            else:
                self.logger.info("robots.txt compliance check passed")

        except Exception as e:
            self.logger.warning(f"Could not check robots.txt: {e}")

    def _get_candidate_index(self, candidate_name: str) -> int:
        """Get or create consistent index for candidate."""
        if candidate_name not in self.candidate_index:
            self.candidate_index[candidate_name] = self.next_index
            self.next_index += 1
        return self.candidate_index[candidate_name]

    def fetch_constituency_page(self, election: str, constituency: str) -> Optional[str]:
        """
        Fetch constituency page with proper rate limiting and error handling.

        Parameters
        ----------
        election : str
            Election identifier
        constituency : str
            Constituency identifier

        Returns
        -------
        str or None
            HTML content if successful, None if failed
        """
        url = urljoin(self.base_url, f"elections/{election}/constituency/{constituency}")

        # Create output directory
        output_dir = Path(f"data/raw/{election}/html")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{constituency}.html"

        # Check if already cached
        if output_file.exists():
            self.logger.info(f"Using cached file: {output_file}")
            return output_file.read_text(encoding='utf-8')

        # Exponential backoff retry logic
        max_retries = 3
        base_delay = self.rate_limit

        for attempt in range(max_retries):
            try:
                # Wait with exponential backoff
                delay = base_delay * (2 ** attempt)
                if attempt > 0:
                    self.logger.info(f"Retry {attempt}, waiting {delay:.1f}s")
                time.sleep(delay)

                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                # Save HTML to file
                html_content = response.text
                output_file.write_text(html_content, encoding='utf-8')

                self.logger.info(f"Successfully fetched and saved: {constituency}")
                return html_content

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {constituency}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All retry attempts failed for {constituency}")
                    return None

        return None

    def parse_count_details(self, html: str, constituency_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse STV count details from HTML with robust error handling.

        Parameters
        ----------
        html : str
            HTML content of constituency page
        constituency_id : str
            Constituency identifier for error reporting

        Returns
        -------
        dict or None
            Parsed count data with transfers, or None if parsing failed
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Look for count table (ElectionsIreland.org and other formats)
            count_table = None
            for selector in [
                'table.rtable',  # ElectionsIreland.org main results table
                'table.count-details',
                'table#transfers',
                'table[class*="count"]',
                'table[id*="transfer"]'
            ]:
                count_table = soup.select_one(selector)
                if count_table:
                    break

            if not count_table:
                self.logger.warning(f"No count table found for {constituency_id}")
                return None

            # Parse table structure
            parsed_data = {
                'constituency_id': constituency_id,
                'candidates': set(),
                'transfers': [],
                'quota': None,
                'valid_poll': None
            }

            # Extract quota and valid poll from header or metadata
            self._extract_metadata(soup, parsed_data)

            # Check if this is ElectionsIreland.org format
            if self._is_elections_ireland_format(count_table):
                transfer_data = self._parse_elections_ireland_table(count_table, constituency_id)
                if transfer_data:
                    parsed_data.update(transfer_data)
            else:
                # Parse transfer rows (generic format)
                rows = count_table.find_all('tr')
                for row_idx, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:  # Need at least: from, to, count
                        continue

                    transfer_data = self._parse_transfer_row(cells, constituency_id, row_idx)
                    if transfer_data:
                        parsed_data['transfers'].extend(transfer_data)
                        # Add candidates to set
                        for transfer in transfer_data:
                            parsed_data['candidates'].add(transfer['from_candidate'])
                            parsed_data['candidates'].add(transfer['to_candidate'])

            # Validate parsed data
            validation_errors = self._validate_parsed_data(parsed_data, constituency_id)
            if validation_errors:
                for error in validation_errors:
                    self.logger.warning(f"{constituency_id}: {error}")

            # Convert candidates set to list
            parsed_data['candidates'] = list(parsed_data['candidates'])

            return parsed_data

        except Exception as e:
            self.logger.error(f"Failed to parse {constituency_id}: {e}")
            return None

    def _extract_metadata(self, soup: BeautifulSoup, parsed_data: Dict) -> None:
        """Extract quota and valid poll from HTML."""
        # Look for quota information (handle comma-separated numbers)
        quota_patterns = [
            r'quota[:\s]+([\d,]+)',
            r'elected.*?([\d,]+)',
            r'threshold[:\s]+([\d,]+)'
        ]

        for pattern in quota_patterns:
            match = re.search(pattern, soup.get_text(), re.IGNORECASE)
            if match:
                # Remove commas and convert to int
                quota_str = match.group(1).replace(',', '')
                parsed_data['quota'] = int(quota_str)
                break

        # Look for valid poll (handle comma-separated numbers)
        poll_patterns = [
            r'valid.*?poll[:\s]+([\d,]+)',
            r'total.*?valid[:\s]+([\d,]+)',
            r'turnout[:\s]+([\d,]+)'
        ]

        for pattern in poll_patterns:
            match = re.search(pattern, soup.get_text(), re.IGNORECASE)
            if match:
                # Remove commas and convert to int
                poll_str = match.group(1).replace(',', '')
                parsed_data['valid_poll'] = int(poll_str)
                break

    def _parse_transfer_row(self, cells: List, constituency_id: str, row_idx: int) -> List[Dict]:
        """Parse a single transfer row, handling various formats."""
        transfers = []

        try:
            # Common patterns for transfer rows:
            # [Round] [From Candidate] [To Candidate] [Count] [Value]
            # or
            # [Elimination of X] [To A: 100] [To B: 50] [Non-transferable: 20]

            row_text = ' '.join(cell.get_text(strip=True) for cell in cells)

            # Extract from candidate from appropriate cell (description usually in cell 1)
            # Handle patterns like "Elimination of Walsh", "Walsh eliminated", "O'Sullivan elected"
            description_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""

            # Extract candidate names from description
            from_candidates = self._extract_candidates_from_description(description_text)

            # Handle different transfer formats
            if 'elimination' in row_text.lower():
                transfers.extend(self._parse_elimination_transfers(cells, from_candidates))
            elif 'surplus' in row_text.lower():
                transfers.extend(self._parse_surplus_transfers(cells, from_candidates))
            else:
                # Try generic transfer parsing
                transfer = self._parse_generic_transfer(cells)
                if transfer:
                    transfers.append(transfer)

        except Exception as e:
            self.logger.warning(f"Failed to parse row {row_idx} in {constituency_id}: {e}")

        return transfers

    def _extract_candidates_from_description(self, description: str) -> List[str]:
        """Extract candidate names from description text."""
        candidates = []

        # Pattern 1: "Elimination of [Name]" or "Elimination of [Name] and [Name]"
        if 'elimination' in description.lower():
            # Extract after "of"
            match = re.search(r'elimination\s+of\s+(.+)', description, re.IGNORECASE)
            if match:
                candidate_text = match.group(1)
                candidates = self._split_candidate_names(candidate_text)

        # Pattern 2: "[Name] elected" or "[Name] eliminated"
        elif 'elected' in description.lower() or 'eliminated' in description.lower():
            # Extract before "elected/eliminated"
            match = re.search(r'(.+?)\s+(?:elected|eliminated)', description, re.IGNORECASE)
            if match:
                candidate_text = match.group(1)
                candidates = self._split_candidate_names(candidate_text)

        # Pattern 3: Just candidate name(s)
        if not candidates:
            candidates = self._split_candidate_names(description)

        return candidates

    def _split_candidate_names(self, candidate_text: str) -> List[str]:
        """Split candidate names on 'and', '&', or ','."""
        # Split on 'and', 'AND', '&', or ','
        candidates = re.split(r'\s+and\s+|\s+AND\s+|\s*&\s*|\s*,\s*', candidate_text.strip())
        return [c.strip() for c in candidates if c.strip()]

    def _parse_elimination_transfers(self, cells: List, from_candidates: List[str]) -> List[Dict]:
        """Parse elimination transfer row."""
        transfers = []

        # Look for transfer destinations in subsequent cells
        for cell in cells[2:]:  # Skip round and from candidate
            cell_text = cell.get_text(strip=True)

            # Handle different transfer formats:
            # "Murphy: 2,450, Kelly: 1,890, Non-transferable: 160"
            # "Murphy (890), Kelly (1,240)"

            # Split on common delimiters and extract name:count pairs
            # First try comma-separated format
            parts = re.split(r',\s*', cell_text)

            for part in parts:
                # Pattern: "Name: number" or "Name (number)"
                match = re.search(r'([^:()]+?)[:,\s]*[:\(]\s*([\d,]+)', part)
                if match:
                    to_candidate = match.group(1).strip()
                    count_str = match.group(2).replace(',', '')

                    try:
                        count = float(count_str)

                        # Handle non-transferable variations
                        if 'non' in to_candidate.lower() and 'transfer' in to_candidate.lower():
                            to_candidate = 'non-transferable'

                        for from_candidate in from_candidates:
                            transfers.append({
                                'from_candidate': from_candidate.strip(),
                                'to_candidate': to_candidate,
                                'transfer_count': count,
                                'transfer_type': 'elimination',
                                'transfer_value': 1.0  # Full value for eliminations
                            })

                    except (ValueError, AttributeError):
                        continue

        return transfers

    def _parse_surplus_transfers(self, cells: List, from_candidates: List[str]) -> List[Dict]:
        """Parse surplus transfer row (fractional transfers)."""
        transfers = []

        # Look for transfer value (fractional)
        transfer_value = 1.0
        for cell in cells:
            cell_text = cell.get_text(strip=True)
            value_match = re.search(r'value[:\s]+(0\.\d+|\d+\.\d+)', cell_text, re.IGNORECASE)
            if value_match:
                transfer_value = float(value_match.group(1))
                break

        # Parse destination transfers similar to elimination
        for cell in cells[2:]:
            cell_text = cell.get_text(strip=True)

            # Split on commas and extract name:count pairs
            parts = re.split(r',\s*', cell_text)

            for part in parts:
                # Pattern: "Name: number" or "Name (number)"
                match = re.search(r'([^:()]+?)[:,\s]*[:\(]\s*([\d,]+)', part)
                if match:
                    to_candidate = match.group(1).strip()
                    count_str = match.group(2).replace(',', '')

                    try:
                        count = float(count_str)

                        # Handle non-transferable variations
                        if 'non' in to_candidate.lower() and 'transfer' in to_candidate.lower():
                            to_candidate = 'non-transferable'

                        for from_candidate in from_candidates:
                            transfers.append({
                                'from_candidate': from_candidate.strip(),
                                'to_candidate': to_candidate,
                                'transfer_count': count,
                                'transfer_type': 'surplus',
                                'transfer_value': transfer_value
                            })

                    except (ValueError, AttributeError):
                        continue

        return transfers

    def _parse_generic_transfer(self, cells: List) -> Optional[Dict]:
        """Parse generic transfer format."""
        if len(cells) < 4:
            return None

        try:
            from_candidate = cells[1].get_text(strip=True)
            to_candidate = cells[2].get_text(strip=True)
            count_text = cells[3].get_text(strip=True)

            count = float(re.search(r'(\d+(?:\.\d+)?)', count_text).group(1))

            return {
                'from_candidate': from_candidate,
                'to_candidate': to_candidate,
                'transfer_count': count,
                'transfer_type': 'transfer',
                'transfer_value': 1.0
            }

        except (ValueError, AttributeError, IndexError):
            return None

    def _validate_parsed_data(self, data: Dict, constituency_id: str) -> List[str]:
        """Validate parsed transfer data."""
        errors = []

        # Check for negative transfers
        for transfer in data['transfers']:
            if transfer['transfer_count'] < 0:
                errors.append(f"Negative transfer count: {transfer}")

        # Validate transfer values
        for transfer in data['transfers']:
            if not 0 <= transfer['transfer_value'] <= 1:
                errors.append(f"Invalid transfer value: {transfer['transfer_value']}")

        # Check for missing candidates
        if not data['candidates']:
            errors.append("No candidates found")

        # Check for missing transfers
        if not data['transfers']:
            errors.append("No transfers found")

        return errors

    def _is_elections_ireland_format(self, table) -> bool:
        """Check if table follows ElectionsIreland.org format."""
        if not table:
            return False

        # Look for the characteristic rtable class
        if table.get('class') and 'rtable' in table.get('class'):
            return True

        # Also check for characteristic header patterns
        header_cells = table.find_all('th')
        for cell in header_cells:
            text = cell.get_text().lower()
            if 'transfer of' in text and 'votes from' in text:
                return True

        return False

    def _parse_elections_ireland_table(self, table, constituency_id: str) -> Dict:
        """Parse ElectionsIreland.org format table."""
        transfers = []
        candidates = set()

        try:
            rows = table.find_all('tr')
            if not rows:
                return {'transfers': transfers, 'candidates': list(candidates)}

            # ElectionsIreland.org format:
            # - Row 0: Transfer descriptions in specific columns
            # - Row 1: Column headers like "Candidate", "Count 1", etc.
            # - Data rows: Alternating candidate name rows and vote count rows

            transfer_columns = {}  # Maps column index to transfer info

            # Extract transfer descriptions from row 0
            if len(rows) > 0:
                header_cells = rows[0].find_all(['td', 'th'])
                for col_idx, cell in enumerate(header_cells):
                    transfer_info = self._extract_elections_ireland_transfer_info(cell)
                    if transfer_info:
                        transfer_columns[col_idx] = transfer_info

            # Process data rows starting from row 2
            # Pattern: candidate name rows are followed by vote count rows
            max_rows = min(len(rows), 100)  # Safety limit to prevent infinite loops
            for i in range(2, max_rows):
                if i >= len(rows):
                    break

                row = rows[i]
                cells = row.find_all(['td', 'th'])

                # Skip empty rows
                if len(cells) <= 1:
                    continue

                # Check if this is a candidate name row (has name in column 1)
                candidate_name = None
                if len(cells) > 1:
                    potential_name = cells[1].get_text(strip=True)
                    # Candidate names are typically longer than 3 chars and contain letters
                    if (len(potential_name) > 3 and
                        any(c.isalpha() for c in potential_name) and
                        potential_name.lower() not in ['candidate', 'party']):
                        candidate_name = potential_name

                if candidate_name:
                    candidates.add(candidate_name)

                    # Check each transfer column for this candidate in the same row
                    for col_idx, transfer_info in transfer_columns.items():
                        if col_idx < len(cells):
                            count_cell = cells[col_idx]
                            count_text = count_cell.get_text(strip=True)

                            # Look for numerical values (may have + or - prefix)
                            count_match = re.search(r'[+-]?([\d,]+(?:\.\d+)?)', count_text)

                            if count_match:
                                try:
                                    count = float(count_match.group(1).replace(',', ''))
                                    if count > 0:  # Only record positive transfers
                                        for from_candidate in transfer_info['from_candidates']:
                                            candidates.add(from_candidate)
                                            transfers.append({
                                                'from_candidate': from_candidate,
                                                'to_candidate': candidate_name,
                                                'transfer_count': count,
                                                'transfer_type': transfer_info['transfer_type'],
                                                'transfer_value': transfer_info.get('transfer_value', 1.0)
                                            })
                                except ValueError:
                                    continue

            return {
                'transfers': transfers,
                'candidates': list(candidates)
            }

        except Exception as e:
            self.logger.warning(f"Error parsing ElectionsIreland table: {e}")
            return {'transfers': [], 'candidates': []}

    def _extract_elections_ireland_transfer_info(self, cell) -> Optional[Dict]:
        """Extract transfer information from ElectionsIreland.org format cell."""
        text = cell.get_text()

        # Look for transfer patterns like "Transfer of votes from Name"
        if 'transfer of' not in text.lower():
            return None

        from_candidates = []
        transfer_type = 'elimination'
        transfer_value = 1.0

        # Extract candidate names from <strong> tags
        strong_tags = cell.find_all('strong')
        for strong in strong_tags:
            name = strong.get_text(strip=True)
            if name:
                from_candidates.append(name)

        # If no strong tags, parse the text format
        # Example: "Transfer ofvotes fromO'Donoghueand O'Donnelland Creminand Daltonand Ní Mhaoldhomhnaigh"
        if not from_candidates:
            # Simple regex approach to extract candidate names
            # Look for pattern after "from" and split on "and"
            from_match = re.search(r'from(.+)', text, re.IGNORECASE)
            if from_match:
                names_text = from_match.group(1).strip()
                # Split on "and" to get individual names
                candidate_parts = re.split(r'\s*and\s*', names_text, flags=re.IGNORECASE)
                from_candidates = [name.strip() for name in candidate_parts if name.strip()]

        # Check for surplus distribution (indicates fractional transfer value)
        if 'surplus' in text.lower():
            transfer_type = 'surplus'
            # Try to extract transfer value if specified
            value_match = re.search(r'value[:\s]+([\d.]+)', text, re.IGNORECASE)
            if value_match:
                try:
                    transfer_value = float(value_match.group(1))
                except ValueError:
                    pass

        if from_candidates:
            return {
                'from_candidates': from_candidates,
                'transfer_type': transfer_type,
                'transfer_value': transfer_value
            }

        return None

    def parse_first_preferences(self, html: str, constituency_id: int) -> List[Dict]:
        """
        Parse first preference vote data from ElectionsIreland.org result pages.

        Parameters
        ----------
        html : str
            Raw HTML content from a constituency result page
        constituency_id : int
            Constituency identifier

        Returns
        -------
        List[Dict]
            List of candidate dictionaries with first preference data
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Find the main results table (rtable class with candidate data)
            results_table = None
            for table in soup.find_all('table', class_='rtable'):
                rows = table.find_all('tr')
                if len(rows) > 10:  # Should have multiple candidate rows
                    header_row = rows[0]
                    header_cells = header_row.find_all(['td', 'th'])
                    header_text = [cell.get_text().strip().lower() for cell in header_cells]

                    # Look for table with candidate and first preference columns
                    if any('candidate' in h for h in header_text) and any('1st' in h or 'pref' in h for h in header_text):
                        results_table = table
                        break

            if not results_table:
                self.logger.warning(f"No results table found for constituency {constituency_id}")
                return []

            candidates = []
            rows = results_table.find_all('tr')

            # Process candidate rows (skip header and empty rows)
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])

                # Skip rows with too few cells or empty rows
                if len(cells) < 10:
                    continue

                # Extract candidate name (column 1)
                candidate_name = cells[1].get_text(strip=True) if len(cells) > 1 else ''

                # Extract first preference votes (column 5)
                votes_text = cells[5].get_text(strip=True) if len(cells) > 5 else ''

                # Extract percentage share (column 7)
                share_text = cells[7].get_text(strip=True) if len(cells) > 7 else ''

                # Skip if missing essential data
                if not candidate_name or not votes_text:
                    continue

                # Skip non-candidate rows (totals, instructions, etc.)
                if any(keyword in candidate_name.lower() for keyword in
                       ['total', 'valid', 'invalid', 'spoilt', 'click', 'data', 'table']):
                    continue

                # Extract party from party cell (column 3) - look for party code in href
                party = ''
                if len(cells) > 3:
                    party_cell = cells[3]
                    party_link = party_cell.find('a')
                    if party_link and party_link.get('href'):
                        href = party_link.get('href')
                        # Extract party code from URL like "party.cfm?election=2020&party=FF"
                        import re
                        party_match = re.search(r'party=([^&]+)', href)
                        if party_match:
                            party_code = party_match.group(1)
                            # Map common party codes to full names
                            party_map = {
                                'FF': 'Fianna Fáil',
                                'FG': 'Fine Gael',
                                'SF': 'Sinn Féin',
                                'LAB': 'Labour',
                                'GP': 'Green Party',
                                'SOC': 'Social Democrats',
                                'AAA': 'Anti-Austerity Alliance',
                                'PBP': 'People Before Profit',
                                'IND': 'Independent',
                                'INDL': 'Independent Left',
                                'INDR': 'Independent Right'
                            }
                            party = party_map.get(party_code, party_code)

                # Clean and convert vote count
                try:
                    votes_clean = votes_text.replace(',', '').replace(' ', '')
                    first_pref_votes = int(votes_clean)
                except ValueError:
                    self.logger.warning(f"Could not parse vote count '{votes_text}' for {candidate_name}")
                    continue

                # Clean and convert percentage
                try:
                    share_clean = share_text.replace('%', '').replace(' ', '')
                    if share_clean:
                        share = float(share_clean)
                    else:
                        share = 0.0
                except ValueError:
                    self.logger.warning(f"Could not parse percentage '{share_text}' for {candidate_name}")
                    share = 0.0

                candidates.append({
                    'constituency_id': constituency_id,
                    'candidate': candidate_name,
                    'party': party,
                    'first_pref_votes': first_pref_votes,
                    'share': share
                })

            self.logger.info(f"Parsed {len(candidates)} candidates from constituency {constituency_id}")
            return candidates

        except Exception as e:
            self.logger.error(f"Failed to parse first preferences for constituency {constituency_id}: {e}")
            return []

    def convert_to_model_format(self, parsed_data_list: List[Dict]) -> ModelData:
        """
        Convert parsed constituency data to ModelData format.

        Parameters
        ----------
        parsed_data_list : list
            List of parsed constituency data dictionaries

        Returns
        -------
        ModelData
            Structured data ready for Bayesian modeling
        """
        events = []
        all_candidates = set()

        # Collect all candidates across constituencies
        for data in parsed_data_list:
            if data:
                all_candidates.update(data['candidates'])

        # Ensure non-transferable is included
        all_candidates.add('non-transferable')

        # Create final candidate index mapping
        candidate_names = {}
        for candidate in sorted(all_candidates):
            idx = self._get_candidate_index(candidate)
            candidate_names[idx] = candidate

        # Create constituency index mapping
        constituency_map = {}
        for idx, data in enumerate(parsed_data_list):
            if data:
                constituency_map[data['constituency_id']] = idx

        # Convert transfers to EventData format
        for const_idx, data in enumerate(parsed_data_list):
            if not data or not data['transfers']:
                continue

            # Group transfers by source candidate
            source_transfers = {}
            for transfer in data['transfers']:
                from_name = transfer['from_candidate']
                from_idx = self._get_candidate_index(from_name)

                if from_idx not in source_transfers:
                    source_transfers[from_idx] = []
                source_transfers[from_idx].append(transfer)

            # Create EventData for each source
            for source_idx, transfers in source_transfers.items():
                # Determine active destinations (those that received transfers)
                destination_counts = {}
                total_transfers = 0

                for transfer in transfers:
                    to_name = transfer['to_candidate']
                    to_idx = self._get_candidate_index(to_name)
                    count = transfer['transfer_count'] * transfer['transfer_value']

                    if to_idx not in destination_counts:
                        destination_counts[to_idx] = 0
                    destination_counts[to_idx] += count
                    total_transfers += count

                if total_transfers > 0:
                    # Create arrays for EventData
                    active_indices = sorted(destination_counts.keys())
                    transfer_counts = [destination_counts[idx] for idx in active_indices]

                    event = EventData(
                        constituency_idx=const_idx,
                        source_indices=np.array([source_idx]),
                        active_indices=np.array(active_indices),
                        transfer_counts=np.array(transfer_counts, dtype=np.float32),
                        total_transfers=float(total_transfers)
                    )
                    events.append(event)

        return ModelData(
            n_sources=len([name for idx, name in candidate_names.items() if name != 'non-transferable']),
            n_destinations=len(candidate_names),
            n_constituencies=len([d for d in parsed_data_list if d]),
            events=events,
            candidate_names=candidate_names
        )

    def scrape_election(self, election: str, constituency_ids: List[str]) -> ModelData:
        """
        Scrape complete election data and convert to ModelData format.

        Parameters
        ----------
        election : str
            Election identifier
        constituency_ids : list
            List of constituency identifiers to scrape

        Returns
        -------
        ModelData
            Complete structured data ready for modeling
        """
        self.logger.info(f"Starting scrape of {election} with {len(constituency_ids)} constituencies")

        parsed_data_list = []

        # Fetch and parse each constituency
        for const_id in constituency_ids:
            self.logger.info(f"Processing constituency: {const_id}")

            # Fetch HTML
            html = self.fetch_constituency_page(election, const_id)
            if html is None:
                self.logger.error(f"Failed to fetch {const_id}, skipping")
                parsed_data_list.append(None)
                continue

            # Parse count details
            parsed_data = self.parse_count_details(html, const_id)
            parsed_data_list.append(parsed_data)

            if parsed_data:
                self.logger.info(f"Successfully parsed {const_id}: "
                               f"{len(parsed_data['transfers'])} transfers, "
                               f"{len(parsed_data['candidates'])} candidates")
            else:
                self.logger.warning(f"No data extracted from {const_id}")

        # Convert to ModelData format
        model_data = self.convert_to_model_format(parsed_data_list)

        # Save processed data
        output_dir = Path(f"data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{election}_model_data.json"

        # Convert to JSON-serializable format
        model_data_dict = {
            'n_sources': model_data.n_sources,
            'n_destinations': model_data.n_destinations,
            'n_constituencies': model_data.n_constituencies,
            'candidate_names': model_data.candidate_names,
            'events': [
                {
                    'constituency_idx': event.constituency_idx,
                    'source_indices': event.source_indices.tolist(),
                    'active_indices': event.active_indices.tolist(),
                    'transfer_counts': event.transfer_counts.tolist(),
                    'total_transfers': event.total_transfers
                }
                for event in model_data.events
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(model_data_dict, f, indent=2)

        self.logger.info(f"Saved processed data to {output_file}")
        self.logger.info(f"Final data: {len(model_data.events)} events, "
                        f"{model_data.n_sources} sources, "
                        f"{model_data.n_destinations} destinations")

        return model_data

    def fetch_first_preferences(self, election: int, cons: int) -> List[Dict]:
        """
        Fetch and parse first preference votes for a constituency.

        Parameters
        ----------
        election : int
            Election year (e.g., 2020)
        cons : int
            Constituency identifier

        Returns
        -------
        List[Dict]
            List of candidate dictionaries with first preference data
        """
        # Construct result page URL
        url = f"{self.base_url}result.cfm?election={election}&cons={cons}"

        self.logger.info(f"Fetching first preferences: {url}")

        try:
            # Fetch the HTML content
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Save raw HTML to data directory
            raw_dir = Path(f"data/raw/{election}/result")
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / f"{cons}.html"

            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            self.logger.info(f"Saved raw HTML to {raw_file}")

            # Parse first preferences
            candidates = self.parse_first_preferences(response.text, cons)

            return candidates

        except Exception as e:
            self.logger.error(f"Failed to fetch first preferences for {election}/{cons}: {e}")
            return []



def fetch_election_results(election_id: str, source: str = "ireland") -> Dict:
    """
    Convenience function to fetch election results.

    Parameters
    ----------
    election_id : str
        Election identifier
    source : str
        Data source ("ireland", "northern_ireland", etc.)

    Returns
    -------
    dict
        Election results data
    """
    source_urls = {
        "ireland": "https://www.rte.ie/news/election-2020/",
        "northern_ireland": "https://www.eoni.org.uk/",
        "scotland": "https://www.gov.scot/",
    }

    if source not in source_urls:
        raise ValueError(f"Unknown source: {source}")

    scraper = ElectionScraper(source_urls[source])
    return scraper.fetch_election_results(election_id)


def validate_election_data(data: Dict) -> List[str]:
    """
    Validate scraped election data for completeness and consistency.

    Parameters
    ----------
    data : dict
        Raw election data

    Returns
    -------
    list
        List of validation warnings/errors
    """
    warnings = []

    # Check required fields
    required_fields = ['election_id', 'candidates', 'transfer_data']
    for field in required_fields:
        if field not in data:
            warnings.append(f"Missing required field: {field}")

    # TODO: Add more comprehensive validation
    # - Check transfer counts sum correctly
    # - Verify candidate IDs are consistent
    # - Check for missing transfer data

    return warnings