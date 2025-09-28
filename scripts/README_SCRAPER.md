# Irish Election Data Scraper

A comprehensive scraper for Irish general election data that downloads and parses both first preference votes and transfer count details from electionsireland.org.

## Overview

This scraper automatically:
1. Discovers all constituencies for specified election years
2. Downloads raw HTML for first preferences and transfer counts
3. Parses and structures the data using the existing ElectionScraper class
4. Combines data into election-wide JSON datasets
5. Includes retry logic, logging, and error handling

## Usage

### Full Election Scraping

```bash
python scripts/scrape_all_elections.py
```

This will scrape all hardcoded elections (2016, 2020, 2024) and save data to:
- Raw HTML: `data/raw/{year}/result/{cons_id}.html` and `data/raw/{year}/counts/{cons_id}.html`
- Processed JSON: `data/processed/{year}/first_prefs.json` and `data/processed/{year}/transfers.json`
- Combined dataset: `data/processed/{year}/combined_election_data.json`

### Demo and Testing

```bash
# Show expected data structure
python scripts/demo_scraper.py --show-structure

# Test constituency discovery only
python scripts/demo_scraper.py --discovery-only --years 2020

# Test single constituency scraping
python scripts/demo_scraper.py --single-constituency 2020 5005

# Show all demo options
python scripts/demo_scraper.py
```

### Running Tests

```bash
# Run scraper tests
pytest tests/test_scrape_all.py -v

# Run with coverage
pytest tests/test_scrape_all.py --cov=scripts.scrape_all_elections
```

## Output Format

### Combined Election Data
```json
{
  "year": 2020,
  "constituencies": {
    "5005": {
      "name": "Limerick",
      "first_prefs": [...],
      "transfer_events": [...]
    }
  }
}
```

### First Preferences
```json
[
  {
    "constituency_id": 5005,
    "candidate": "Patrick O'Donovan",
    "party": "Fine Gael",
    "first_pref_votes": 9228,
    "share": 20.02
  }
]
```

### Transfer Events
```json
[
  {
    "from_candidate": "O'Brien",
    "to_candidate": "O'Donovan",
    "transfer_count": 800,
    "transfer_type": "elimination",
    "transfer_value": 1.0
  }
]
```

## Features

- **Constituency Discovery**: Automatically finds all constituencies for each election year
- **Retry Logic**: Handles network failures with exponential backoff
- **Rate Limiting**: Respectful 2-second delays between requests
- **Progress Reporting**: Shows real-time progress as `[2020 - Limerick] ✅ first prefs, ✅ transfers`
- **Comprehensive Logging**: Detailed logs saved to `logs/` directory
- **Data Validation**: Built-in validation and quality checks
- **Caching**: Reuses existing HTML files to avoid re-downloading
- **Global Candidate Indexing**: Consistent candidate indexing across constituencies

## Architecture

### Main Components

1. **ElectionDiscovery**: Discovers constituencies from listing pages
2. **ComprehensiveElectionScraper**: Orchestrates the full scraping process
3. **ElectionScraper**: Existing class for parsing HTML (from `src/stv_transfers/scraper.py`)

### Directory Structure

```
data/
├── raw/
│   └── {year}/
│       ├── result/     # First preference HTML
│       └── counts/     # Transfer count HTML
└── processed/
    └── {year}/
        ├── first_prefs.json
        ├── transfers.json
        └── combined_election_data.json

logs/
└── scraper_{timestamp}.log
```

## Configuration

### Election Years
Currently hardcoded in `main()`:
```python
ELECTION_YEARS = [2016, 2020, 2024]
```

### Rate Limiting
Default 2-second delay between requests. Configurable in initialization:
```python
scraper = ComprehensiveElectionScraper(
    base_url="https://electionsireland.org",
    rate_limit=2.0  # seconds
)
```

## Error Handling

- Network failures trigger retry with exponential backoff
- Missing constituencies are logged as warnings but don't stop processing
- Parse failures are logged but don't crash the scraper
- Progress is displayed showing success/failure for each constituency

## Testing

The test suite (`tests/test_scrape_all.py`) includes:
- Mocked network requests to avoid hitting live servers
- Constituency discovery validation
- Data structure validation
- Error handling tests
- Progress reporting verification

## Dependencies

- `requests`: HTTP requests
- `beautifulsoup4`: HTML parsing
- `pathlib`: File system operations
- Existing project dependencies for `stv_transfers` module

## Legal and Ethical Use

- Includes respectful rate limiting (2+ second delays)
- Proper User-Agent headers identifying academic use
- Checks robots.txt compliance (inherited from ElectionScraper)
- Caches data to minimize server load

## Troubleshooting

### Common Issues

1. **No constituencies found**: Check if the electionsireland.org URL structure has changed
2. **Network timeouts**: Increase rate_limit parameter
3. **Permission errors**: Ensure write access to `data/` and `logs/` directories
4. **Import errors**: Ensure you're running from the project root directory

### Debug Mode

Enable verbose logging:
```bash
python scripts/scrape_all_elections.py --verbose
```

Or modify the script to use debug logging:
```python
setup_logging(verbose=True)
```