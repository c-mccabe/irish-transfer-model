#!/usr/bin/env python3
"""
Demonstration script for the Irish election scraper.

This script shows how to use the scraper in a controlled way,
with options for testing and validation.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from scrape_all_elections import ComprehensiveElectionScraper, setup_logging


def demo_discovery_only(years):
    """Demonstrate constituency discovery without full scraping."""
    print("🔍 Testing constituency discovery...")

    scraper = ComprehensiveElectionScraper(
        base_url="https://electionsireland.org",
        rate_limit=2.0
    )

    for year in years:
        print(f"\n📅 Discovering constituencies for {year}:")

        try:
            result = scraper.discovery.discover_constituencies(year)
            constituencies = result["constituencies"]

            print(f"  Found {len(constituencies)} constituencies:")
            for cons_id, cons_info in list(constituencies.items())[:5]:  # Show first 5
                print(f"    {cons_id}: {cons_info['name']}")

            if len(constituencies) > 5:
                print(f"    ... and {len(constituencies) - 5} more")

        except Exception as e:
            print(f"  ❌ Error: {e}")


def demo_single_constituency(year, cons_id):
    """Demonstrate scraping a single constituency."""
    print(f"🏛️  Testing single constituency scrape: {year}/{cons_id}")

    scraper = ComprehensiveElectionScraper(
        base_url="https://electionsireland.org",
        rate_limit=2.0
    )

    # Mock constituency info for single scrape
    cons_info = {"name": f"Constituency_{cons_id}", "url": f"test_url_{cons_id}"}

    try:
        result = scraper._scrape_constituency_comprehensive(year, cons_id, cons_info)

        if result:
            print(f"  ✅ Success!")
            print(f"    Name: {result['name']}")
            print(f"    First preferences: {len(result.get('first_prefs', []))} candidates")
            print(f"    Transfer events: {len(result.get('transfer_events', []))} events")

            # Show sample data
            if result.get('first_prefs'):
                print(f"    Sample candidate: {result['first_prefs'][0].get('candidate', 'Unknown')}")

        else:
            print(f"  ❌ No data retrieved")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def demo_data_structure():
    """Demonstrate the expected data structure."""
    print("📋 Expected data structure:")

    sample_structure = {
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

    print(json.dumps(sample_structure, indent=2))


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Demonstrate Irish election scraper")
    parser.add_argument("--discovery-only", action="store_true",
                       help="Only test constituency discovery")
    parser.add_argument("--single-constituency", nargs=2, metavar=("YEAR", "CONS_ID"),
                       help="Test scraping a single constituency")
    parser.add_argument("--show-structure", action="store_true",
                       help="Show expected data structure")
    parser.add_argument("--years", nargs="+", type=int, default=[2020],
                       help="Years to process (default: 2020)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    print("🗳️  Irish Election Scraper Demo")
    print("=" * 40)

    if args.show_structure:
        demo_data_structure()
        return

    if args.discovery_only:
        demo_discovery_only(args.years)
        return

    if args.single_constituency:
        year, cons_id = int(args.single_constituency[0]), args.single_constituency[1]
        demo_single_constituency(year, cons_id)
        return

    # Default: show help and available options
    print("Available demo options:")
    print("  --discovery-only          Test constituency discovery")
    print("  --single-constituency     Test single constituency scraping")
    print("  --show-structure          Show expected data structure")
    print("")
    print("Examples:")
    print("  python scripts/demo_scraper.py --discovery-only --years 2020")
    print("  python scripts/demo_scraper.py --single-constituency 2020 5005")
    print("  python scripts/demo_scraper.py --show-structure")


if __name__ == "__main__":
    main()