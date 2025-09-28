#!/usr/bin/env python3
"""
Comprehensive transfer data cleaning and enhancement script.

This script addresses the key issues identified in the scraped transfer data:
1. Candidate name inconsistencies (surname vs full name)
2. Missing constituency IDs in transfer records
3. Missing elimination order information
4. Tracking and reporting of constituencies without transfer data

The script produces enhanced, research-ready datasets with consistent naming,
complete metadata, and proper sequencing information.
"""

import json
import sys
import re
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import logging

# Add project path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TransferDataCleaner:
    """Comprehensive transfer data cleaning and enhancement."""

    def __init__(self, verbose: bool = True):
        """
        Initialize the data cleaner.

        Parameters
        ----------
        verbose : bool
            Enable verbose logging
        """
        self.verbose = verbose
        self.logger = self._setup_logging()

        # Statistics tracking
        self.stats = {
            'name_mappings_created': 0,
            'ambiguous_names_resolved': 0,
            'transfers_enhanced': 0,
            'elimination_orders_inferred': 0,
            'validation_errors': 0
        }

        # Issue tracking
        self.issues = {
            'unresolved_names': [],
            'ambiguous_mappings': [],
            'missing_eliminations': [],
            'validation_failures': []
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def clean_all_elections(self, years: List[int] = [2016, 2020, 2024]) -> Dict[str, Any]:
        """
        Clean and enhance transfer data for all specified election years.

        Parameters
        ----------
        years : List[int]
            Election years to process

        Returns
        -------
        Dict[str, Any]
            Enhanced dataset with cleaned transfer data
        """
        self.logger.info("Starting comprehensive transfer data cleaning")

        enhanced_data = {}
        from datetime import datetime
        metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'years_processed': years,
            'statistics': {},
            'issues_summary': {},
            'constituencies_without_transfers': {}
        }

        for year in years:
            self.logger.info(f"Processing {year} election data")
            enhanced_election = self._clean_election_year(year)

            if enhanced_election:
                enhanced_data[str(year)] = enhanced_election
                self.logger.info(f"✅ {year} election data processed successfully")
            else:
                self.logger.warning(f"❌ Failed to process {year} election data")

        # Add metadata
        metadata['statistics'] = self.stats.copy()
        metadata['issues_summary'] = {
            key: len(value) for key, value in self.issues.items()
        }
        enhanced_data['_metadata'] = metadata

        return enhanced_data

    def _clean_election_year(self, year: int) -> Optional[Dict[str, Any]]:
        """Clean transfer data for a single election year."""
        data_file = Path(f"data/processed/{year}/combined_election_data.json")

        if not data_file.exists():
            self.logger.error(f"Data file not found: {data_file}")
            return None

        try:
            with open(data_file, 'r') as f:
                raw_data = json.load(f)

            enhanced_data = copy.deepcopy(raw_data)

            # Track constituencies without transfer data
            constituencies_without_transfers = []
            constituencies_with_transfers = []

            for const_id, constituency in enhanced_data['constituencies'].items():
                if not constituency.get('transfer_events'):
                    constituencies_without_transfers.append({
                        'id': const_id,
                        'name': constituency['name'],
                        'candidates': len(constituency.get('first_prefs', [])),
                        'reason': 'no_transfer_data_extracted'
                    })
                else:
                    constituencies_with_transfers.append(const_id)

            # Process constituencies with transfer data
            for const_id in constituencies_with_transfers:
                self._clean_constituency_data(enhanced_data['constituencies'][const_id], const_id, year)

            # Add metadata about missing transfers
            enhanced_data['_transfer_coverage'] = {
                'total_constituencies': len(enhanced_data['constituencies']),
                'with_transfers': len(constituencies_with_transfers),
                'without_transfers': len(constituencies_without_transfers),
                'coverage_rate': len(constituencies_with_transfers) / len(enhanced_data['constituencies']) * 100,
                'missing_details': constituencies_without_transfers
            }

            return enhanced_data

        except Exception as e:
            self.logger.error(f"Error processing {year} data: {e}")
            return None

    def _clean_constituency_data(self, constituency: Dict, const_id: str, year: int) -> None:
        """Clean transfer data for a single constituency."""
        self.logger.debug(f"Cleaning constituency {const_id} - {constituency['name']}")

        # Step 1: Create candidate name mapping
        name_mapping = self._create_candidate_mapping(constituency)

        # Step 2: Standardize transfer candidate names
        self._standardize_transfer_names(constituency['transfer_events'], name_mapping, const_id)

        # Step 3: Add constituency metadata to transfers
        self._add_constituency_metadata(constituency['transfer_events'], const_id, constituency['name'])

        # Step 4: Infer elimination order
        self._infer_elimination_order(constituency)

        # Step 5: Validate cleaned data
        self._validate_constituency_data(constituency, const_id)

    def _create_candidate_mapping(self, constituency: Dict) -> Dict[str, str]:
        """Create mapping from surname/partial names to full candidate names."""
        full_names = [c['candidate'] for c in constituency.get('first_prefs', [])]
        mapping = {}

        # Add non-transferable as special case
        mapping['non-transferable'] = 'non-transferable'
        mapping['Non-transferable'] = 'non-transferable'
        mapping['Non-Transferable'] = 'non-transferable'

        for full_name in full_names:
            # Extract possible surname variations
            surname_variations = self._extract_name_variations(full_name)

            for variation in surname_variations:
                if variation in mapping and mapping[variation] != full_name:
                    # Handle ambiguous mappings
                    self.logger.warning(f"Ambiguous name mapping: '{variation}' could be '{mapping[variation]}' or '{full_name}'")
                    self.issues['ambiguous_mappings'].append({
                        'variation': variation,
                        'candidates': [mapping[variation], full_name]
                    })
                    # Use longer name as default
                    if len(full_name) > len(mapping[variation]):
                        mapping[variation] = full_name
                else:
                    mapping[variation] = full_name

        self.stats['name_mappings_created'] += len(mapping)
        return mapping

    def _extract_name_variations(self, full_name: str) -> List[str]:
        """Extract possible name variations that could appear in transfer records."""
        variations = []

        # Clean the name
        name = full_name.strip()
        parts = re.split(r'\s+', name)

        if len(parts) >= 2:
            # Primary surname (last part)
            surname = parts[-1]
            variations.append(surname)

            # Handle compound surnames (O'Brien, Ní Mhaoldhomhnaigh, etc.)
            if len(parts) >= 2 and parts[-2] in ["O'", "Ó", "Mac", "Mc", "Ní", "de"]:
                compound_surname = f"{parts[-2]}{parts[-1]}"
                variations.append(compound_surname)

            # Handle cases where transferable name might be "O'Brien" but full name is "Patrick O'Brien"
            if parts[-1].startswith(("O'", "Ó")):
                variations.append(parts[-1])

        # Add the full name itself (in case it appears complete in transfers)
        variations.append(name)

        # Handle special characters and common variations
        for i, variation in enumerate(variations):
            # Remove fada marks for matching
            no_fada = variation.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            if no_fada != variation:
                variations.append(no_fada)

        return list(set(variations))  # Remove duplicates

    def _standardize_transfer_names(self, transfers: List[Dict], name_mapping: Dict[str, str], const_id: str) -> None:
        """Standardize candidate names in transfer records."""
        for transfer in transfers:
            original_from = transfer.get('from_candidate', '')
            original_to = transfer.get('to_candidate', '')

            # Map from_candidate
            if original_from in name_mapping:
                transfer['from_candidate'] = name_mapping[original_from]
            else:
                # Try fuzzy matching for common issues
                mapped_from = self._fuzzy_name_match(original_from, name_mapping)
                if mapped_from:
                    transfer['from_candidate'] = mapped_from
                    self.stats['ambiguous_names_resolved'] += 1
                else:
                    self.logger.warning(f"Unresolved from_candidate: '{original_from}' in {const_id}")
                    self.issues['unresolved_names'].append({
                        'constituency': const_id,
                        'name': original_from,
                        'type': 'from_candidate'
                    })

            # Map to_candidate (usually already full names, but check for consistency)
            if original_to in name_mapping:
                transfer['to_candidate'] = name_mapping[original_to]
            elif original_to not in [c for c in name_mapping.values()]:
                # Check if this is a variation of a known candidate
                mapped_to = self._fuzzy_name_match(original_to, name_mapping)
                if mapped_to:
                    transfer['to_candidate'] = mapped_to
                    self.stats['ambiguous_names_resolved'] += 1
                else:
                    self.logger.warning(f"Unresolved to_candidate: '{original_to}' in {const_id}")
                    self.issues['unresolved_names'].append({
                        'constituency': const_id,
                        'name': original_to,
                        'type': 'to_candidate'
                    })

            self.stats['transfers_enhanced'] += 1

    def _fuzzy_name_match(self, name: str, name_mapping: Dict[str, str]) -> Optional[str]:
        """Attempt fuzzy matching for unresolved names."""
        name_lower = name.lower().strip()

        # Try partial matches
        for variation, full_name in name_mapping.items():
            if name_lower == variation.lower().strip():
                return full_name

        # Try substring matches (with caution)
        for variation, full_name in name_mapping.items():
            if len(name) > 3 and name_lower in variation.lower():
                return full_name
            if len(variation) > 3 and variation.lower() in name_lower:
                return full_name

        return None

    def _add_constituency_metadata(self, transfers: List[Dict], const_id: str, const_name: str) -> None:
        """Add constituency information to each transfer record."""
        for transfer in transfers:
            transfer['constituency_id'] = const_id
            transfer['constituency_name'] = const_name

    def _infer_elimination_order(self, constituency: Dict) -> None:
        """Infer elimination order from transfer patterns and data."""
        transfers = constituency.get('transfer_events', [])
        if not transfers:
            return

        # Group transfers by source candidate
        transfers_by_source = defaultdict(list)
        for transfer in transfers:
            source = transfer.get('from_candidate')
            if source:
                transfers_by_source[source].append(transfer)

        # Analyze transfer patterns to infer elimination sequence
        elimination_candidates = []
        for source_candidate, source_transfers in transfers_by_source.items():
            # Check if this looks like an elimination (high transfer counts, "elimination" type)
            elimination_transfers = [t for t in source_transfers if t.get('transfer_type') == 'elimination']

            if elimination_transfers:
                total_eliminated = sum(t.get('transfer_count', 0) for t in elimination_transfers)
                elimination_candidates.append({
                    'candidate': source_candidate,
                    'total_transferred': total_eliminated,
                    'transfer_count': len(elimination_transfers)
                })

        # Sort by total transferred (rough proxy for elimination order - lower transfers = eliminated earlier)
        elimination_candidates.sort(key=lambda x: x['total_transferred'])

        # Assign elimination orders
        for order, candidate_info in enumerate(elimination_candidates, 1):
            candidate = candidate_info['candidate']
            # Add elimination order to all transfers from this candidate
            for transfer in transfers:
                if transfer.get('from_candidate') == candidate:
                    transfer['elimination_order'] = order

        self.stats['elimination_orders_inferred'] += len(elimination_candidates)

        # Store elimination sequence metadata
        constituency['_elimination_sequence'] = [
            {
                'order': i + 1,
                'candidate': candidate_info['candidate'],
                'total_transferred': candidate_info['total_transferred']
            }
            for i, candidate_info in enumerate(elimination_candidates)
        ]

    def _validate_constituency_data(self, constituency: Dict, const_id: str) -> None:
        """Validate cleaned constituency data for consistency."""
        first_pref_names = {c['candidate'] for c in constituency.get('first_prefs', [])}
        first_pref_names.add('non-transferable')  # Always valid

        validation_errors = []

        for transfer in constituency.get('transfer_events', []):
            from_candidate = transfer.get('from_candidate')
            to_candidate = transfer.get('to_candidate')

            # Check that all candidates in transfers exist in first preferences
            if from_candidate and from_candidate not in first_pref_names:
                validation_errors.append(f"from_candidate '{from_candidate}' not in first preferences")

            if to_candidate and to_candidate not in first_pref_names:
                validation_errors.append(f"to_candidate '{to_candidate}' not in first preferences")

            # Check required fields
            required_fields = ['constituency_id', 'constituency_name']
            for field in required_fields:
                if field not in transfer:
                    validation_errors.append(f"Missing field '{field}' in transfer")

        if validation_errors:
            self.logger.warning(f"Validation errors in {const_id}: {len(validation_errors)} issues")
            self.issues['validation_failures'].extend([
                {'constituency': const_id, 'error': error} for error in validation_errors
            ])
            self.stats['validation_errors'] += len(validation_errors)

    def save_enhanced_data(self, enhanced_data: Dict[str, Any], output_dir: str = "data/cleaned") -> Dict[str, Path]:
        """Save enhanced data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save complete enhanced dataset
        complete_file = output_path / "enhanced_election_data.json"
        with open(complete_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        saved_files['complete'] = complete_file

        # Save year-specific enhanced files
        for year_str, election_data in enhanced_data.items():
            if year_str.startswith('_'):  # Skip metadata
                continue

            year_file = output_path / f"enhanced_{year_str}.json"
            with open(year_file, 'w') as f:
                json.dump(election_data, f, indent=2, ensure_ascii=False)
            saved_files[year_str] = year_file

        # Save issue reports
        issues_file = output_path / "data_cleaning_issues.json"
        with open(issues_file, 'w') as f:
            json.dump({
                'statistics': self.stats,
                'issues': self.issues,
                'summary': {
                    'total_issues': sum(len(issue_list) for issue_list in self.issues.values()),
                    'resolution_rate': (
                        (self.stats['transfers_enhanced'] - self.stats['validation_errors']) /
                        max(self.stats['transfers_enhanced'], 1) * 100
                    )
                }
            }, f, indent=2)
        saved_files['issues'] = issues_file

        return saved_files

    def generate_cleaning_report(self) -> str:
        """Generate a comprehensive cleaning report."""
        report = []
        report.append("🧹 TRANSFER DATA CLEANING REPORT")
        report.append("=" * 50)
        report.append("")

        # Statistics
        report.append("📊 CLEANING STATISTICS:")
        report.append(f"  Candidate name mappings created: {self.stats['name_mappings_created']:,}")
        report.append(f"  Transfer records enhanced: {self.stats['transfers_enhanced']:,}")
        report.append(f"  Ambiguous names resolved: {self.stats['ambiguous_names_resolved']:,}")
        report.append(f"  Elimination orders inferred: {self.stats['elimination_orders_inferred']:,}")
        report.append(f"  Validation errors: {self.stats['validation_errors']:,}")
        report.append("")

        # Issues summary
        report.append("⚠️  ISSUES ENCOUNTERED:")
        report.append(f"  Unresolved names: {len(self.issues['unresolved_names'])}")
        report.append(f"  Ambiguous mappings: {len(self.issues['ambiguous_mappings'])}")
        report.append(f"  Missing eliminations: {len(self.issues['missing_eliminations'])}")
        report.append(f"  Validation failures: {len(self.issues['validation_failures'])}")
        report.append("")

        # Success rate
        total_issues = sum(len(issue_list) for issue_list in self.issues.values())
        success_rate = max(0, (self.stats['transfers_enhanced'] - total_issues) / max(self.stats['transfers_enhanced'], 1) * 100)
        report.append(f"✅ OVERALL SUCCESS RATE: {success_rate:.1f}%")

        return "\n".join(report)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean and enhance Irish election transfer data")
    parser.add_argument("--years", nargs="+", type=int, default=[2016, 2020, 2024],
                       help="Election years to process")
    parser.add_argument("--output-dir", default="data/cleaned",
                       help="Output directory for enhanced data")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    print("🧹 Starting comprehensive transfer data cleaning...")
    print("=" * 60)

    try:
        cleaner = TransferDataCleaner(verbose=args.verbose)
        enhanced_data = cleaner.clean_all_elections(args.years)

        if enhanced_data:
            saved_files = cleaner.save_enhanced_data(enhanced_data, args.output_dir)

            print("\n" + cleaner.generate_cleaning_report())
            print("\n📁 FILES SAVED:")
            for file_type, file_path in saved_files.items():
                print(f"  {file_type}: {file_path}")

            print(f"\n✅ Data cleaning completed successfully!")
            print(f"Enhanced data saved to: {args.output_dir}")

        else:
            print("❌ No data was processed successfully")
            return 1

    except Exception as e:
        print(f"❌ Data cleaning failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())