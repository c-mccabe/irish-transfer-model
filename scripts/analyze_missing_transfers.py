#!/usr/bin/env python3
"""
Analyze constituencies with missing transfer data and create action plan.

This script examines the scraped data to identify:
1. Which constituencies have no transfer data
2. Why transfer data extraction failed
3. Alternative data sources or extraction methods
4. Quality assessment of available data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re

# Add project path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def analyze_missing_transfers() -> Dict[str, Any]:
    """
    Comprehensive analysis of missing transfer data across all elections.

    Returns
    -------
    Dict[str, Any]
        Analysis results with missing constituencies and patterns
    """
    years = [2016, 2020, 2024]
    analysis = {
        'summary': {},
        'by_year': {},
        'patterns': {
            'consistently_missing': [],
            'sometimes_missing': [],
            'never_missing': []
        },
        'html_analysis': {}
    }

    # Track constituencies across years
    constituency_coverage = defaultdict(list)

    for year in years:
        data_file = Path(f"data/processed/{year}/combined_election_data.json")

        if not data_file.exists():
            print(f"⚠️  Warning: {data_file} not found")
            continue

        with open(data_file, 'r') as f:
            data = json.load(f)

        year_analysis = analyze_year_data(data, year)
        analysis['by_year'][year] = year_analysis

        # Track constituency patterns
        for const in year_analysis['with_transfers']:
            constituency_coverage[const['name']].append(f"{year}:✅")
        for const in year_analysis['without_transfers']:
            constituency_coverage[const['name']].append(f"{year}:❌")

    # Analyze patterns across years
    analysis['patterns'] = analyze_coverage_patterns(constituency_coverage)

    # Generate summary statistics
    analysis['summary'] = generate_summary_stats(analysis['by_year'])

    # Analyze HTML files for missing constituencies
    analysis['html_analysis'] = analyze_html_files(analysis['by_year'])

    return analysis

def analyze_year_data(data: Dict, year: int) -> Dict[str, Any]:
    """Analyze transfer data coverage for a single year."""
    total_constituencies = len(data['constituencies'])
    with_transfers = []
    without_transfers = []

    for const_id, constituency in data['constituencies'].items():
        transfer_count = len(constituency.get('transfer_events', []))
        const_info = {
            'id': const_id,
            'name': constituency['name'],
            'candidates': len(constituency.get('first_prefs', [])),
            'transfer_count': transfer_count
        }

        if transfer_count > 0:
            with_transfers.append(const_info)
        else:
            without_transfers.append(const_info)

    coverage_rate = len(with_transfers) / total_constituencies * 100 if total_constituencies > 0 else 0

    return {
        'total_constituencies': total_constituencies,
        'with_transfers': with_transfers,
        'without_transfers': without_transfers,
        'coverage_rate': coverage_rate,
        'transfer_events_total': sum(c['transfer_count'] for c in with_transfers)
    }

def analyze_coverage_patterns(constituency_coverage: Dict) -> Dict[str, List]:
    """Analyze which constituencies consistently have or lack transfer data."""
    patterns = {
        'consistently_missing': [],
        'sometimes_missing': [],
        'consistently_present': [],
        'mixed_coverage': []
    }

    for const_name, coverage_list in constituency_coverage.items():
        present_count = sum(1 for c in coverage_list if '✅' in c)
        total_count = len(coverage_list)

        if present_count == 0:
            patterns['consistently_missing'].append({
                'name': const_name,
                'coverage': coverage_list,
                'present_rate': 0.0
            })
        elif present_count == total_count:
            patterns['consistently_present'].append({
                'name': const_name,
                'coverage': coverage_list,
                'present_rate': 1.0
            })
        else:
            patterns['mixed_coverage'].append({
                'name': const_name,
                'coverage': coverage_list,
                'present_rate': present_count / total_count
            })

    return patterns

def analyze_html_files(year_data: Dict) -> Dict[str, Any]:
    """Analyze raw HTML files to understand why transfer parsing failed."""
    html_analysis = {
        'file_sizes': {},
        'content_analysis': {},
        'missing_files': [],
        'empty_files': [],
        'problematic_files': []
    }

    for year, data in year_data.items():
        html_analysis['file_sizes'][year] = {}
        html_analysis['content_analysis'][year] = {}

        for const in data['without_transfers']:
            const_id = const['id']

            # Check if HTML files exist
            result_file = Path(f"data/raw/{year}/result/{const_id}.html")
            counts_file = Path(f"data/raw/{year}/counts/{const_id}.html")

            file_info = {
                'name': const['name'],
                'result_exists': result_file.exists(),
                'counts_exists': counts_file.exists(),
                'result_size': result_file.stat().st_size if result_file.exists() else 0,
                'counts_size': counts_file.stat().st_size if counts_file.exists() else 0
            }

            # Analyze file content
            if counts_file.exists():
                content_analysis = analyze_html_content(counts_file)
                file_info.update(content_analysis)

                # Categorize issues
                if file_info['counts_size'] < 1000:  # Very small files
                    html_analysis['empty_files'].append(file_info)
                elif not content_analysis.get('has_table', False):
                    html_analysis['problematic_files'].append(file_info)
            else:
                html_analysis['missing_files'].append(file_info)

            html_analysis['file_sizes'][year][const_id] = file_info

    return html_analysis

def analyze_html_content(html_file: Path) -> Dict[str, Any]:
    """Analyze HTML file content to understand structure."""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        analysis = {
            'has_table': 'table' in content.lower(),
            'has_rtable': 'rtable' in content.lower(),
            'has_count_text': 'count' in content.lower(),
            'has_transfer_text': 'transfer' in content.lower(),
            'has_elimination_text': 'elimination' in content.lower(),
            'table_count': content.lower().count('<table'),
            'content_indicators': []
        }

        # Look for specific indicators
        indicators = [
            ('quota', r'quota[:\s]*(\d+)'),
            ('valid_poll', r'valid.*poll[:\s]*(\d+)'),
            ('candidate_names', r'<td[^>]*>[A-Z][a-zA-Z\s\']+</td>'),
            ('vote_numbers', r'\d{2,6}'),
            ('percentage_signs', r'\d+\.\d+%')
        ]

        for indicator, pattern in indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                analysis['content_indicators'].append({
                    'type': indicator,
                    'count': len(matches),
                    'samples': matches[:3]  # First 3 matches
                })

        return analysis

    except Exception as e:
        return {
            'error': str(e),
            'has_table': False,
            'has_rtable': False
        }

def generate_summary_stats(year_data: Dict) -> Dict[str, Any]:
    """Generate overall summary statistics."""
    total_constituencies = sum(data['total_constituencies'] for data in year_data.values())
    total_with_transfers = sum(len(data['with_transfers']) for data in year_data.values())
    total_transfer_events = sum(data['transfer_events_total'] for data in year_data.values())

    return {
        'total_constituencies_all_years': total_constituencies,
        'constituencies_with_transfers': total_with_transfers,
        'constituencies_without_transfers': total_constituencies - total_with_transfers,
        'overall_coverage_rate': (total_with_transfers / total_constituencies * 100) if total_constituencies > 0 else 0,
        'total_transfer_events': total_transfer_events,
        'years_analyzed': list(year_data.keys())
    }

def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print a comprehensive analysis report."""
    print("🔍 MISSING TRANSFER DATA ANALYSIS REPORT")
    print("=" * 60)
    print()

    # Summary statistics
    summary = analysis['summary']
    print("📊 OVERALL SUMMARY:")
    print(f"  Total constituencies (all years): {summary['total_constituencies_all_years']}")
    print(f"  With transfer data: {summary['constituencies_with_transfers']}")
    print(f"  Without transfer data: {summary['constituencies_without_transfers']}")
    print(f"  Overall coverage rate: {summary['overall_coverage_rate']:.1f}%")
    print(f"  Total transfer events captured: {summary['total_transfer_events']:,}")
    print()

    # Year-by-year breakdown
    print("📅 YEAR-BY-YEAR BREAKDOWN:")
    for year, data in analysis['by_year'].items():
        print(f"  {year}:")
        print(f"    Constituencies: {data['total_constituencies']}")
        print(f"    Coverage: {len(data['with_transfers'])}/{data['total_constituencies']} ({data['coverage_rate']:.1f}%)")
        print(f"    Transfer events: {data['transfer_events_total']:,}")
    print()

    # Coverage patterns
    print("🔄 COVERAGE PATTERNS:")
    patterns = analysis['patterns']
    print(f"  Consistently missing transfer data: {len(patterns['consistently_missing'])} constituencies")
    print(f"  Mixed coverage: {len(patterns['mixed_coverage'])} constituencies")
    print(f"  Consistently present: {len(patterns['consistently_present'])} constituencies")

    if patterns['consistently_missing']:
        print("  \n  Always missing:")
        for const in patterns['consistently_missing'][:10]:
            print(f"    • {const['name']}")
        if len(patterns['consistently_missing']) > 10:
            print(f"    ... and {len(patterns['consistently_missing']) - 10} more")
    print()

    # HTML analysis
    html_analysis = analysis['html_analysis']
    print("📄 HTML FILE ANALYSIS:")
    total_empty = sum(len(files) for files in html_analysis['empty_files'])
    total_missing = sum(len(files) for files in html_analysis['missing_files'])
    total_problematic = sum(len(files) for files in html_analysis['problematic_files'])

    print(f"  Empty/very small HTML files: {total_empty}")
    print(f"  Missing HTML files: {total_missing}")
    print(f"  Files without proper table structure: {total_problematic}")
    print()

def generate_action_plan(analysis: Dict[str, Any]) -> None:
    """Generate action plan based on analysis."""
    print("🛠️  ACTION PLAN:")
    print("=" * 40)
    print()

    html_analysis = analysis['html_analysis']
    patterns = analysis['patterns']

    print("1. **IMMEDIATE ACTIONS:**")

    # Check for alternative URL patterns
    if html_analysis['empty_files'] or html_analysis['problematic_files']:
        print("   • Investigate alternative URL patterns for transfer data")
        print("   • Some constituencies may use different count page formats")
        print("   • Check if electionsireland.org has count data under different paths")

    # Re-scrape problematic files
    if html_analysis['missing_files']:
        print("   • Re-attempt download of missing HTML files")
        print("   • Check for URL pattern changes or redirects")

    print()
    print("2. **ENHANCED PARSING:**")
    print("   • Develop alternative parsers for different HTML formats")
    print("   • Extract data from result pages if count pages unavailable")
    print("   • Implement fuzzy parsing for non-standard table formats")

    print()
    print("3. **DATA COMPLETION STRATEGIES:**")
    consistently_missing = len(patterns['consistently_missing'])
    mixed_coverage = len(patterns['mixed_coverage'])

    if consistently_missing > 0:
        print(f"   • {consistently_missing} constituencies never have transfer data")
        print("   • May indicate systematic data unavailability")
        print("   • Consider manual verification or alternative sources")

    if mixed_coverage > 0:
        print(f"   • {mixed_coverage} constituencies have partial coverage")
        print("   • Retry failed extractions with enhanced parsers")
        print("   • Pattern analysis may reveal extraction issues")

    print()
    print("4. **QUALITY IMPROVEMENT:**")
    print("   • Create constituency-specific extraction profiles")
    print("   • Implement validation checks for extracted transfer data")
    print("   • Add metadata tracking for extraction success/failure reasons")

def save_analysis_report(analysis: Dict[str, Any]) -> None:
    """Save detailed analysis to JSON file."""
    output_file = Path("data/analysis/missing_transfers_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Detailed analysis saved to: {output_file}")

def main():
    """Main execution function."""
    print("Starting missing transfer data analysis...\n")

    try:
        analysis = analyze_missing_transfers()
        print_analysis_report(analysis)
        generate_action_plan(analysis)
        save_analysis_report(analysis)

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())