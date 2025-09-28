#!/usr/bin/env python3
"""
Comprehensive Irish election data scraper.

Downloads and parses both first preference votes and transfer count details
for all constituencies in Irish general elections from electionsireland.org.

This script:
1. Discovers all constituencies for each election year
2. Downloads raw HTML for first preferences and transfers
3. Parses and structures the data using the existing ElectionScraper class
4. Combines data into election-wide JSON datasets
5. Includes retry logic, logging, and error handling

Usage:
    python scripts/scrape_all_elections.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import requests
from bs4 import BeautifulSoup
import sys
import re
from urllib.parse import urljoin, urlparse

# Add project path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stv_transfers.scraper import ElectionScraper
from stv_transfers.data_structures import ModelData


class ElectionDiscovery:
    """Discovers constituencies and election metadata from electionsireland.org."""

    def __init__(self, base_url: str = "https://electionsireland.org", rate_limit: float = 0.0):
        """
        Initialize election discovery service.

        Parameters
        ----------
        base_url : str
            Base URL for electionsireland.org
        rate_limit : float
            Minimum seconds between requests (0 = no rate limiting)
        """
        self.base_url = base_url
        self.rate_limit = rate_limit

        # Session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Irish-Transfer-Model Research Tool (Academic Use)'
        })

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def discover_constituencies(self, year: int) -> Dict[str, Any]:
        """
        Discover all constituencies for a given election year.

        Parameters
        ----------
        year : int
            Election year (e.g., 2020)

        Returns
        -------
        Dict[str, Any]
            Dictionary with constituency IDs, names, and metadata
        """
        self.logger.info(f"Discovering constituencies for {year} election")

        # Map years to Dáil numbers
        year_to_dail = {
            2016: "32dail",
            2020: "33dail",
            2024: "34dail"
        }

        if year not in year_to_dail:
            self.logger.error(f"Unknown election year: {year}")
            return {"year": year, "constituencies": {}, "metadata": {}}

        # Try to fetch the general election page
        url = f"{self.base_url}/results/general/{year_to_dail[year]}.cfm"

        try:
            if self.rate_limit > 0:
                time.sleep(self.rate_limit)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            constituencies = self._parse_constituency_page(response.text, year)

            self.logger.info(f"Found {len(constituencies['constituencies'])} constituencies for {year}")
            return constituencies

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch constituency list for {year}: {e}")
            return {"year": year, "constituencies": {}, "metadata": {}}

    def _parse_constituency_page(self, html: str, year: int) -> Dict[str, Any]:
        """Parse constituency listing page to extract constituency IDs and names."""
        soup = BeautifulSoup(html, 'html.parser')
        constituencies = {}

        # Look for links to constituency result pages
        # ElectionsIreland.org typically uses patterns like:
        # result.cfm?election=2020&cons=5001
        # or direct links with constituency names

        # Find all links that contain 'result.cfm' or 'cons=' parameters
        result_links = soup.find_all('a', href=re.compile(r'result\.cfm.*cons=\d+'))

        for link in result_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)

            # Extract constituency ID from URL
            cons_match = re.search(r'cons=(\d+)', href)
            if cons_match:
                cons_id = cons_match.group(1)

                # Clean up constituency name
                constituency_name = text.strip()
                # Remove common prefixes/suffixes if present
                constituency_name = re.sub(r'^(constituency\s*)?', '', constituency_name, flags=re.IGNORECASE)
                constituency_name = re.sub(r'\s*(constituency)?\s*$', '', constituency_name, flags=re.IGNORECASE)

                if constituency_name and cons_id:
                    constituencies[cons_id] = {
                        "name": constituency_name,
                        "url": urljoin(self.base_url, href)
                    }

        # If no direct result links found, try alternative parsing strategies
        if not constituencies:
            self.logger.warning(f"No result links found for {year}, trying alternative parsing")
            constituencies = self._parse_alternative_format(soup, year)

        return {
            "year": year,
            "constituencies": constituencies,
            "metadata": {
                "total_constituencies": len(constituencies),
                "discovery_timestamp": time.time()
            }
        }

    def _parse_alternative_format(self, soup: BeautifulSoup, year: int) -> Dict[str, Dict]:
        """Try alternative parsing strategies for constituency discovery."""
        constituencies = {}

        # Strategy 1: Look for table rows with constituency data
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    for cell in cells:
                        links = cell.find_all('a')
                        for link in links:
                            href = link.get('href', '')
                            if 'result.cfm' in href and f'election={year}' in href:
                                cons_match = re.search(r'cons=(\d+)', href)
                                if cons_match:
                                    cons_id = cons_match.group(1)
                                    name = link.get_text(strip=True)
                                    if name and cons_id:
                                        constituencies[cons_id] = {
                                            "name": name,
                                            "url": urljoin(self.base_url, href)
                                        }

        # Strategy 2: Generate common constituency IDs if still no results
        if not constituencies:
            self.logger.warning(f"Using fallback constituency generation for {year}")
            # Irish constituencies typically have IDs in ranges like 5001-5040
            # This is a fallback - in practice you'd want to verify these exist
            for i in range(5001, 5041):  # Typical range for Irish constituencies
                constituencies[str(i)] = {
                    "name": f"Constituency_{i}",
                    "url": f"{self.base_url}/result.cfm?election={year}&cons={i}"
                }

        return constituencies


class ComprehensiveElectionScraper:
    """
    Comprehensive scraper that combines constituency discovery with data extraction.
    """

    def __init__(self, base_url: str = "https://electionsireland.org", rate_limit: float = 0.0):
        """
        Initialize comprehensive election scraper.

        Parameters
        ----------
        base_url : str
            Base URL for electionsireland.org
        rate_limit : float
            Minimum seconds between requests (0 = no rate limiting)
        """
        self.base_url = base_url
        self.rate_limit = rate_limit

        # Initialize discovery service and scraper
        self.discovery = ElectionDiscovery(base_url, rate_limit)
        # Ensure base_url has trailing slash for ElectionScraper
        scraper_base_url = base_url.rstrip('/') + '/'
        self.scraper = ElectionScraper(scraper_base_url, rate_limit)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Track candidate indexing across elections
        self.global_candidate_index = {"non-transferable": 0}
        self.next_global_index = 1

    def scrape_all_elections(self, years: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Scrape all specified elections comprehensively.

        Parameters
        ----------
        years : List[int]
            List of election years to scrape

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Complete election dataset organized by year
        """
        self.logger.info(f"Starting comprehensive scrape of elections: {years}")

        all_elections = {}

        for year in years:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing {year} election")
            self.logger.info(f"{'='*50}")

            election_data = self.scrape_single_election(year)
            all_elections[year] = election_data

            # Brief pause between elections
            if self.rate_limit > 0:
                time.sleep(self.rate_limit)

        self.logger.info(f"\nCompleted scraping {len(years)} elections")
        return all_elections

    def scrape_single_election(self, year: int) -> Dict[str, Any]:
        """
        Scrape a single election year comprehensively.

        Parameters
        ----------
        year : int
            Election year

        Returns
        -------
        Dict[str, Any]
            Complete election dataset
        """
        # Step 1: Discover constituencies
        discovery_result = self.discovery.discover_constituencies(year)
        constituencies_info = discovery_result["constituencies"]

        if not constituencies_info:
            self.logger.error(f"No constituencies found for {year}")
            return {
                "year": year,
                "constituencies": {},
                "metadata": {"error": "No constituencies discovered"}
            }

        # Step 2: Create directory structure
        self._create_directories(year)

        # Step 3: Scrape each constituency
        election_data = {
            "year": year,
            "constituencies": {},
            "metadata": {
                "scrape_timestamp": time.time(),
                "total_constituencies": len(constituencies_info),
                "successful_constituencies": 0,
                "failed_constituencies": 0
            }
        }

        for cons_id, cons_info in constituencies_info.items():
            self.logger.info(f"Processing constituency: {cons_id} - {cons_info['name']}")

            # Scrape constituency data
            constituency_data = self._scrape_constituency_comprehensive(year, cons_id, cons_info)

            if constituency_data:
                election_data["constituencies"][cons_id] = constituency_data
                election_data["metadata"]["successful_constituencies"] += 1

                # Print progress
                first_pref_status = "✅" if constituency_data.get("first_prefs") else "❌"
                transfer_status = "✅" if constituency_data.get("transfer_events") else "❌"
                print(f"[{year} - {cons_info['name']}] {first_pref_status} first prefs, {transfer_status} transfers")
            else:
                election_data["metadata"]["failed_constituencies"] += 1
                print(f"[{year} - {cons_info['name']}] ❌ failed")

        # Step 4: Save combined election data
        self._save_election_data(election_data)

        return election_data

    def _scrape_constituency_comprehensive(self, year: int, cons_id: str, cons_info: Dict) -> Optional[Dict[str, Any]]:
        """
        Scrape comprehensive data for a single constituency.

        Parameters
        ----------
        year : int
            Election year
        cons_id : str
            Constituency ID
        cons_info : Dict
            Constituency information from discovery

        Returns
        -------
        Optional[Dict[str, Any]]
            Constituency data with first preferences and transfers
        """
        constituency_data = {
            "constituency_id": cons_id,
            "name": cons_info["name"],
            "first_prefs": [],
            "transfer_events": []
        }

        # Fetch first preferences with retry logic
        first_prefs = self._fetch_first_preferences_with_retry(year, int(cons_id))
        if first_prefs:
            constituency_data["first_prefs"] = first_prefs

        # Fetch transfer data with retry logic
        transfers = self._fetch_transfers_with_retry(year, cons_id)
        if transfers:
            # Convert to model format
            transfer_events = self._convert_transfers_to_events(transfers, cons_id)
            constituency_data["transfer_events"] = transfer_events

        # Only return data if we got at least some information
        if constituency_data["first_prefs"] or constituency_data["transfer_events"]:
            return constituency_data

        return None

    def _fetch_first_preferences_with_retry(self, year: int, cons_id: int) -> List[Dict]:
        """Fetch first preferences with retry logic."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if attempt > 0 and self.rate_limit > 0:
                    wait_time = self.rate_limit * (2 ** attempt)
                    self.logger.info(f"Retry {attempt} for first prefs {year}/{cons_id}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)

                # Use the existing scraper method
                candidates = self.scraper.fetch_first_preferences(year, cons_id)
                return candidates

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for first prefs {year}/{cons_id}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All retry attempts failed for first prefs {year}/{cons_id}")

        return []

    def _fetch_transfers_with_retry(self, year: int, cons_id: str) -> Optional[Dict]:
        """Fetch transfer data with retry logic."""
        max_retries = 3

        # Check if we already have cached HTML for transfers
        raw_file = Path(f"data/raw/{year}/counts/{cons_id}.html")
        if raw_file.exists():
            self.logger.info(f"Using cached transfer HTML: {raw_file}")
            html_content = raw_file.read_text(encoding='utf-8')
            return self.scraper.parse_count_details(html_content, cons_id)

        for attempt in range(max_retries):
            try:
                if attempt > 0 and self.rate_limit > 0:
                    wait_time = self.rate_limit * (2 ** attempt)
                    self.logger.info(f"Retry {attempt} for transfers {year}/{cons_id}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)

                # Construct transfer/count URL - may vary by site format
                # Common patterns: counts.cfm, result.cfm with different params
                transfer_url = f"{self.base_url}/counts.cfm?election={year}&cons={cons_id}"

                response = self.scraper.session.get(transfer_url, timeout=30)
                response.raise_for_status()

                # Save raw HTML
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                raw_file.write_text(response.text, encoding='utf-8')

                # Parse transfer data
                parsed_data = self.scraper.parse_count_details(response.text, cons_id)
                return parsed_data

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for transfers {year}/{cons_id}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All retry attempts failed for transfers {year}/{cons_id}")

        return None

    def _convert_transfers_to_events(self, parsed_data: Dict, cons_id: str) -> List[Dict]:
        """Convert parsed transfer data to event format."""
        if not parsed_data or 'transfers' not in parsed_data:
            return []

        events = []
        for transfer in parsed_data['transfers']:
            # Ensure candidate indexing consistency
            from_candidate = transfer.get('from_candidate', '')
            to_candidate = transfer.get('to_candidate', '')

            # Add to global candidate index
            self._get_global_candidate_index(from_candidate)
            self._get_global_candidate_index(to_candidate)

            event = {
                'from_candidate': from_candidate,
                'to_candidate': to_candidate,
                'transfer_count': transfer.get('transfer_count', 0),
                'transfer_type': transfer.get('transfer_type', ''),
                'transfer_value': transfer.get('transfer_value', 1.0)
            }
            events.append(event)

        return events

    def _get_global_candidate_index(self, candidate_name: str) -> int:
        """Get or create global candidate index across all elections."""
        if candidate_name not in self.global_candidate_index:
            self.global_candidate_index[candidate_name] = self.next_global_index
            self.next_global_index += 1
        return self.global_candidate_index[candidate_name]

    def _create_directories(self, year: int) -> None:
        """Create necessary directory structure for data storage."""
        directories = [
            f"data/raw/{year}/result",      # First preference HTML
            f"data/raw/{year}/counts",      # Transfer count HTML
            f"data/processed/{year}"        # Processed JSON
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _save_election_data(self, election_data: Dict[str, Any]) -> None:
        """Save election data to JSON with pretty formatting."""
        year = election_data["year"]

        # Save to processed directory
        output_file = Path(f"data/processed/{year}/combined_election_data.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(election_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved combined election data to: {output_file}")

        # Also save separate files for first prefs and transfers
        self._save_separate_datasets(election_data)

    def _save_separate_datasets(self, election_data: Dict[str, Any]) -> None:
        """Save separate first preferences and transfers JSON files."""
        year = election_data["year"]

        # Combine first preferences across constituencies
        all_first_prefs = []
        for cons_id, cons_data in election_data["constituencies"].items():
            if "first_prefs" in cons_data:
                all_first_prefs.extend(cons_data["first_prefs"])

        # Combine transfers across constituencies
        all_transfers = []
        for cons_id, cons_data in election_data["constituencies"].items():
            if "transfer_events" in cons_data:
                all_transfers.extend(cons_data["transfer_events"])

        # Save first preferences
        first_prefs_file = Path(f"data/processed/{year}/first_prefs.json")
        with open(first_prefs_file, 'w', encoding='utf-8') as f:
            json.dump(all_first_prefs, f, indent=2, ensure_ascii=False)

        # Save transfers
        transfers_file = Path(f"data/processed/{year}/transfers.json")
        with open(transfers_file, 'w', encoding='utf-8') as f:
            json.dump(all_transfers, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(all_first_prefs)} first preference records to: {first_prefs_file}")
        self.logger.info(f"Saved {len(all_transfers)} transfer records to: {transfers_file}")


def setup_logging(verbose: bool = True) -> None:
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/scraper_{int(time.time())}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main execution function."""
    print("🗳️  Irish Election Data Scraper")
    print("=" * 50)

    # Setup logging
    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)

    # Hardcoded election years as specified
    ELECTION_YEARS = [2016, 2020, 2024]

    logger.info(f"Starting comprehensive scrape of elections: {ELECTION_YEARS}")

    try:
        # Initialize comprehensive scraper
        scraper = ComprehensiveElectionScraper(
            base_url="https://electionsireland.org",
            rate_limit=0.0  # No rate limiting for faster execution
        )

        # Scrape all elections
        all_elections = scraper.scrape_all_elections(ELECTION_YEARS)

        # Print summary
        print("\n" + "=" * 50)
        print("📊 SCRAPING SUMMARY")
        print("=" * 50)

        for year, election_data in all_elections.items():
            total = election_data["metadata"]["total_constituencies"]
            successful = election_data["metadata"]["successful_constituencies"]
            failed = election_data["metadata"]["failed_constituencies"]
            success_rate = (successful / total * 100) if total > 0 else 0

            print(f"{year}: {successful}/{total} constituencies ({success_rate:.1f}% success)")

        print(f"\n✅ Scraping completed! Data saved to data/processed/")

    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        print("\n⏹️  Scraping stopped by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        print(f"\n❌ Scraping failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())