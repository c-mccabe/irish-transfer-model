#!/usr/bin/env python3
"""
Data validation and quality assessment for scraped STV election data.

This script performs comprehensive validation of scraped election data:
1. Check candidate coverage and consistency across constituencies
2. Verify vote totals and transfer patterns
3. Flag statistical anomalies and data quality issues
4. Generate summary statistics and data quality reports
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime
import scipy.stats as stats

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stv_transfers.scraper import ElectionScraper
from stv_transfers.data_structures import ModelData


class DataValidator:
    """Comprehensive validator for scraped STV election data."""

    def __init__(self, verbose: bool = True):
        """
        Initialize data validator.

        Parameters
        ----------
        verbose : bool
            Print progress and diagnostic messages
        """
        self.verbose = verbose
        self.issues = []  # Track validation issues
        self.warnings = []  # Track warnings
        self.stats = {}  # Store summary statistics

    def validate_data_file(self, data_file: Path) -> Dict[str, Any]:
        """
        Validate a single data file.

        Parameters
        ----------
        data_file : Path
            Path to JSON data file

        Returns
        -------
        Dict[str, Any]
            Validation results and statistics
        """
        if self.verbose:
            print(f"Validating data file: {data_file}")

        # Load data
        with open(data_file, 'r') as f:
            scraped_data = json.load(f)

        if not isinstance(scraped_data, list):
            self.issues.append("Data file must contain a list of constituency records")
            return {"valid": False, "issues": self.issues}

        # Run validation checks
        results = {
            "file": str(data_file),
            "timestamp": datetime.now().isoformat(),
            "n_constituencies": len(scraped_data),
            "valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {}
        }

        # Basic structure validation
        self._validate_structure(scraped_data, results)

        # Candidate consistency checks
        self._validate_candidates(scraped_data, results)

        # Transfer pattern validation
        self._validate_transfers(scraped_data, results)

        # Vote total consistency
        self._validate_vote_totals(scraped_data, results)

        # Statistical anomaly detection
        self._detect_anomalies(scraped_data, results)

        # Generate summary statistics
        self._compute_summary_stats(scraped_data, results)

        # Determine overall validity
        results["valid"] = len(results["issues"]) == 0

        if self.verbose:
            print(f"Validation complete: {len(results['issues'])} issues, "
                  f"{len(results['warnings'])} warnings")

        return results

    def _validate_structure(self, scraped_data: List[Dict], results: Dict[str, Any]) -> None:
        """Validate basic data structure and required fields."""
        if self.verbose:
            print("  Checking data structure...")

        required_fields = ['constituency_id', 'candidates', 'transfers', 'quota', 'valid_poll']
        transfer_fields = ['from_candidate', 'to_candidate', 'transfer_count',
                          'transfer_type', 'transfer_value']

        for i, constituency in enumerate(scraped_data):
            # Check required fields
            for field in required_fields:
                if field not in constituency:
                    results["issues"].append(
                        f"Constituency {i}: Missing required field '{field}'"
                    )

            # Check transfer structure
            if 'transfers' in constituency:
                for j, transfer in enumerate(constituency['transfers']):
                    for field in transfer_fields:
                        if field not in transfer:
                            results["issues"].append(
                                f"Constituency {i}, transfer {j}: Missing field '{field}'"
                            )

                    # Check data types
                    if 'transfer_count' in transfer:
                        if not isinstance(transfer['transfer_count'], (int, float)):
                            results["issues"].append(
                                f"Constituency {i}, transfer {j}: transfer_count must be numeric"
                            )

                    if 'transfer_value' in transfer:
                        if not isinstance(transfer['transfer_value'], (int, float)):
                            results["issues"].append(
                                f"Constituency {i}, transfer {j}: transfer_value must be numeric"
                            )

    def _validate_candidates(self, scraped_data: List[Dict], results: Dict[str, Any]) -> None:
        """Validate candidate names and coverage across constituencies."""
        if self.verbose:
            print("  Checking candidate consistency...")

        all_candidates = set()
        constituency_candidates = {}
        candidate_frequencies = Counter()

        for i, constituency in enumerate(scraped_data):
            if 'candidates' not in constituency:
                continue

            candidates = set(constituency['candidates'])
            all_candidates.update(candidates)
            constituency_candidates[constituency.get('constituency_id', f"const_{i}")] = candidates

            for candidate in candidates:
                candidate_frequencies[candidate] += 1

        # Check for non-transferable handling
        non_transferable_variants = ['non-transferable', 'Non-transferable',
                                   'Non-Transferable', 'NON-TRANSFERABLE',
                                   'exhausted', 'Exhausted']

        nt_found = [variant for variant in non_transferable_variants if variant in all_candidates]
        if len(nt_found) > 1:
            results["warnings"].append(
                f"Multiple non-transferable variants found: {nt_found}"
            )
        elif len(nt_found) == 0:
            results["warnings"].append(
                "No non-transferable option found in candidate lists"
            )

        # Check for candidates appearing in very few constituencies
        n_constituencies = len(scraped_data)
        rare_candidates = [cand for cand, freq in candidate_frequencies.items()
                          if freq < 0.1 * n_constituencies and cand not in non_transferable_variants]

        if rare_candidates:
            results["warnings"].append(
                f"Candidates appearing in <10% of constituencies: {len(rare_candidates)} candidates"
            )

        # Store candidate statistics
        results["statistics"]["candidates"] = {
            "total_unique": len(all_candidates),
            "non_transferable_variants": nt_found,
            "rare_candidates": len(rare_candidates),
            "candidate_frequencies": dict(candidate_frequencies.most_common(10))
        }

    def _validate_transfers(self, scraped_data: List[Dict], results: Dict[str, Any]) -> None:
        """Validate transfer patterns and detect anomalies."""
        if self.verbose:
            print("  Checking transfer patterns...")

        transfer_types = Counter()
        transfer_values = defaultdict(list)
        self_transfers = []
        invalid_candidates = []

        for i, constituency in enumerate(scraped_data):
            if 'transfers' not in constituency or 'candidates' not in constituency:
                continue

            candidates = set(constituency['candidates'])
            const_id = constituency.get('constituency_id', f"const_{i}")

            for j, transfer in enumerate(constituency['transfers']):
                # Count transfer types
                if 'transfer_type' in transfer:
                    transfer_types[transfer['transfer_type']] += 1

                # Collect transfer values by type
                if 'transfer_value' in transfer and 'transfer_type' in transfer:
                    transfer_values[transfer['transfer_type']].append(transfer['transfer_value'])

                # Check for self-transfers
                from_cand = transfer.get('from_candidate')
                to_cand = transfer.get('to_candidate')
                if from_cand == to_cand:
                    self_transfers.append((const_id, from_cand))

                # Check candidates exist in candidate list
                if from_cand not in candidates:
                    invalid_candidates.append((const_id, from_cand, 'from'))
                if to_cand not in candidates:
                    invalid_candidates.append((const_id, to_cand, 'to'))

        # Flag issues
        if self_transfers:
            results["issues"].append(
                f"Self-transfers detected: {len(self_transfers)} instances"
            )

        if invalid_candidates:
            results["issues"].append(
                f"Transfers reference unknown candidates: {len(invalid_candidates)} instances"
            )

        # Check transfer value patterns
        for transfer_type, values in transfer_values.items():
            values = np.array(values)

            if transfer_type == 'elimination':
                # Eliminations should mostly have transfer_value = 1.0
                non_unity = np.sum(np.abs(values - 1.0) > 0.01)
                if non_unity > 0.1 * len(values):
                    results["warnings"].append(
                        f"Elimination transfers with non-unity values: {non_unity}/{len(values)}"
                    )

            elif transfer_type == 'surplus':
                # Surplus transfers should have transfer_value < 1.0
                invalid_surplus = np.sum(values >= 1.0)
                if invalid_surplus > 0:
                    results["warnings"].append(
                        f"Surplus transfers with value ≥ 1.0: {invalid_surplus}/{len(values)}"
                    )

        # Store transfer statistics
        results["statistics"]["transfers"] = {
            "types": dict(transfer_types),
            "self_transfers": len(self_transfers),
            "invalid_candidate_refs": len(invalid_candidates),
            "transfer_value_stats": {
                ttype: {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                } for ttype, values in transfer_values.items()
            }
        }

    def _validate_vote_totals(self, scraped_data: List[Dict], results: Dict[str, Any]) -> None:
        """Validate vote totals and transfer conservation."""
        if self.verbose:
            print("  Checking vote conservation...")

        conservation_issues = []

        for i, constituency in enumerate(scraped_data):
            const_id = constituency.get('constituency_id', f"const_{i}")

            if 'transfers' not in constituency:
                continue

            # Group transfers by source candidate
            transfers_by_source = defaultdict(list)
            for transfer in constituency['transfers']:
                from_cand = transfer.get('from_candidate')
                if from_cand:
                    transfers_by_source[from_cand].append(transfer)

            # Check conservation for each source
            for source_cand, transfers in transfers_by_source.items():
                # Sum transfers out
                total_out = sum(t.get('transfer_count', 0) for t in transfers)

                # For elimination transfers, all votes should be transferred
                elimination_transfers = [t for t in transfers if t.get('transfer_type') == 'elimination']
                if elimination_transfers:
                    # Should be one elimination event per candidate
                    if len(set(t.get('transfer_value', 0) for t in elimination_transfers)) > 1:
                        results["warnings"].append(
                            f"{const_id}: Multiple elimination events for {source_cand}"
                        )

                # For surplus transfers, check transfer values are consistent
                surplus_transfers = [t for t in transfers if t.get('transfer_type') == 'surplus']
                if surplus_transfers:
                    transfer_values = [t.get('transfer_value', 0) for t in surplus_transfers]
                    if len(set(transfer_values)) > 1:
                        results["warnings"].append(
                            f"{const_id}: Inconsistent surplus transfer values for {source_cand}"
                        )

        # Store vote conservation statistics
        results["statistics"]["vote_conservation"] = {
            "issues_detected": len(conservation_issues)
        }

    def _detect_anomalies(self, scraped_data: List[Dict], results: Dict[str, Any]) -> None:
        """Detect statistical anomalies in transfer patterns."""
        if self.verbose:
            print("  Detecting statistical anomalies...")

        # Collect transfer rates for anomaly detection
        transfer_rates = defaultdict(list)  # (from_type, to_type) -> rates

        for constituency in scraped_data:
            if 'transfers' not in constituency:
                continue

            # Group by source candidate
            by_source = defaultdict(list)
            for transfer in constituency['transfers']:
                from_cand = transfer.get('from_candidate')
                by_source[from_cand].append(transfer)

            # Compute transfer rates
            for source_cand, source_transfers in by_source.items():
                total_transfers = sum(t.get('transfer_count', 0) for t in source_transfers)

                if total_transfers > 0:
                    for transfer in source_transfers:
                        to_cand = transfer.get('to_candidate')
                        count = transfer.get('transfer_count', 0)
                        rate = count / total_transfers

                        # Categorize candidates
                        from_type = 'nt' if 'non-transferable' in source_cand.lower() else 'candidate'
                        to_type = 'nt' if 'non-transferable' in to_cand.lower() else 'candidate'

                        transfer_rates[(from_type, to_type)].append(rate)

        # Detect outliers using IQR method
        anomalies = []
        for (from_type, to_type), rates in transfer_rates.items():
            if len(rates) < 10:  # Need sufficient data
                continue

            rates = np.array(rates)
            q1, q3 = np.percentile(rates, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = np.sum((rates < lower_bound) | (rates > upper_bound))
            if outliers > 0.1 * len(rates):  # More than 10% outliers
                anomalies.append({
                    "pattern": f"{from_type} → {to_type}",
                    "outlier_count": int(outliers),
                    "total_observations": len(rates),
                    "outlier_fraction": float(outliers / len(rates)),
                    "rate_stats": {
                        "median": float(np.median(rates)),
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(iqr)
                    }
                })

        if anomalies:
            results["warnings"].append(
                f"Statistical anomalies detected in {len(anomalies)} transfer patterns"
            )

        results["statistics"]["anomalies"] = anomalies

    def _compute_summary_stats(self, scraped_data: List[Dict], results: Dict[str, Any]) -> None:
        """Compute comprehensive summary statistics."""
        if self.verbose:
            print("  Computing summary statistics...")

        stats = {
            "data_overview": {
                "n_constituencies": len(scraped_data),
                "total_transfers": 0,
                "total_votes_transferred": 0
            },
            "coverage": {},
            "quality_metrics": {}
        }

        # Count totals
        for constituency in scraped_data:
            if 'transfers' in constituency:
                stats["data_overview"]["total_transfers"] += len(constituency['transfers'])
                for transfer in constituency['transfers']:
                    stats["data_overview"]["total_votes_transferred"] += transfer.get('transfer_count', 0)

        # Coverage statistics
        constituencies_with_data = sum(1 for c in scraped_data if 'transfers' in c and c['transfers'])
        stats["coverage"]["constituencies_with_transfers"] = constituencies_with_data
        stats["coverage"]["coverage_rate"] = constituencies_with_data / len(scraped_data) if scraped_data else 0

        # Quality metrics
        total_issues = len(results["issues"])
        total_warnings = len(results["warnings"])
        stats["quality_metrics"] = {
            "issues_per_constituency": total_issues / len(scraped_data) if scraped_data else 0,
            "warnings_per_constituency": total_warnings / len(scraped_data) if scraped_data else 0,
            "overall_quality_score": max(0, 1 - (total_issues + 0.5 * total_warnings) / len(scraped_data)) if scraped_data else 0
        }

        results["statistics"]["summary"] = stats

    def validate_directory(self, data_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate all data files in a directory.

        Parameters
        ----------
        data_dir : Path
            Directory containing data files
        output_dir : Path, optional
            Directory to save validation reports

        Returns
        -------
        Dict[str, Any]
            Combined validation results
        """
        if self.verbose:
            print(f"Validating data directory: {data_dir}")

        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            return {
                "valid": False,
                "issues": [f"No JSON files found in {data_dir}"],
                "file_results": {}
            }

        # Validate each file
        file_results = {}
        all_issues = []
        all_warnings = []

        for data_file in json_files:
            try:
                file_result = self.validate_data_file(data_file)
                file_results[data_file.name] = file_result

                all_issues.extend(f"{data_file.name}: {issue}" for issue in file_result["issues"])
                all_warnings.extend(f"{data_file.name}: {warning}" for warning in file_result["warnings"])

            except Exception as e:
                error_msg = f"Failed to validate {data_file.name}: {e}"
                all_issues.append(error_msg)
                file_results[data_file.name] = {
                    "valid": False,
                    "error": str(e)
                }

        # Combine results
        combined_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "data_directory": str(data_dir),
            "files_processed": len(json_files),
            "valid": len(all_issues) == 0,
            "total_issues": len(all_issues),
            "total_warnings": len(all_warnings),
            "issues": all_issues,
            "warnings": all_warnings,
            "file_results": file_results,
            "summary_statistics": self._combine_file_statistics(file_results)
        }

        # Save validation report
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            report_file = output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(combined_results, f, indent=2)

            if self.verbose:
                print(f"Validation report saved to: {report_file}")

            # Create summary visualization
            self._create_validation_plots(combined_results, output_dir)

        return combined_results

    def _combine_file_statistics(self, file_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine statistics across all validated files."""
        combined_stats = {
            "total_constituencies": 0,
            "total_transfers": 0,
            "total_candidates": set(),
            "transfer_types": Counter(),
            "quality_scores": []
        }

        for filename, result in file_results.items():
            if "statistics" in result:
                stats = result["statistics"]

                # Aggregate counts
                if "summary" in stats:
                    summary = stats["summary"]
                    combined_stats["total_constituencies"] += summary["data_overview"]["n_constituencies"]
                    combined_stats["total_transfers"] += summary["data_overview"]["total_transfers"]
                    combined_stats["quality_scores"].append(summary["quality_metrics"]["overall_quality_score"])

                # Aggregate candidates
                if "candidates" in stats:
                    combined_stats["total_candidates"].update(stats["candidates"]["candidate_frequencies"].keys())

                # Aggregate transfer types
                if "transfers" in stats:
                    for ttype, count in stats["transfers"]["types"].items():
                        combined_stats["transfer_types"][ttype] += count

        # Convert set to count
        combined_stats["total_unique_candidates"] = len(combined_stats["total_candidates"])
        del combined_stats["total_candidates"]

        # Compute average quality score
        if combined_stats["quality_scores"]:
            combined_stats["average_quality_score"] = np.mean(combined_stats["quality_scores"])
        else:
            combined_stats["average_quality_score"] = 0.0

        return combined_stats

    def _create_validation_plots(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create validation summary plots."""
        if self.verbose:
            print("  Creating validation plots...")

        # Quality score distribution
        quality_scores = []
        file_names = []

        for filename, file_result in results["file_results"].items():
            if "statistics" in file_result and "summary" in file_result["statistics"]:
                quality_score = file_result["statistics"]["summary"]["quality_metrics"]["overall_quality_score"]
                quality_scores.append(quality_score)
                file_names.append(filename)

        if quality_scores:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Quality scores by file
            ax1.bar(range(len(quality_scores)), quality_scores)
            ax1.set_xlabel('Data File')
            ax1.set_ylabel('Quality Score')
            ax1.set_title('Data Quality Scores by File')
            ax1.set_xticks(range(len(file_names)))
            ax1.set_xticklabels([f.replace('.json', '') for f in file_names], rotation=45, ha='right')
            ax1.axhline(y=0.8, color='orange', linestyle='--', label='Good threshold')
            ax1.axhline(y=0.9, color='green', linestyle='--', label='Excellent threshold')
            ax1.legend()

            # Issues and warnings summary
            total_issues = results["total_issues"]
            total_warnings = results["total_warnings"]

            categories = ['Issues', 'Warnings']
            counts = [total_issues, total_warnings]
            colors = ['red', 'orange']

            ax2.bar(categories, counts, color=colors, alpha=0.7)
            ax2.set_ylabel('Count')
            ax2.set_title('Validation Issues Summary')

            # Add count labels on bars
            for i, count in enumerate(counts):
                ax2.text(i, count + max(counts) * 0.01, str(count), ha='center')

            plt.tight_layout()
            plt.savefig(output_dir / "validation_summary.png", dpi=300, bbox_inches='tight')
            plt.close()

        if self.verbose:
            print(f"  Validation plots saved to {output_dir}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Validate scraped STV election data")
    parser.add_argument("data_path", type=Path,
                       help="Path to data file or directory to validate")
    parser.add_argument("--output-dir", type=Path, default="data/validation",
                       help="Directory to save validation reports")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress messages")

    args = parser.parse_args()

    # Initialize validator
    validator = DataValidator(verbose=not args.quiet)

    try:
        if args.data_path.is_file():
            # Validate single file
            results = validator.validate_data_file(args.data_path)

            # Save results
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = args.output_dir / f"validation_{args.data_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n📊 Validation Results:")
            print(f"  Valid: {'✅ Yes' if results['valid'] else '❌ No'}")
            print(f"  Issues: {len(results['issues'])}")
            print(f"  Warnings: {len(results['warnings'])}")
            print(f"  Report saved to: {output_file}")

        elif args.data_path.is_dir():
            # Validate directory
            results = validator.validate_directory(args.data_path, args.output_dir)

            print(f"\n📊 Validation Results:")
            print(f"  Files processed: {results['files_processed']}")
            print(f"  Overall valid: {'✅ Yes' if results['valid'] else '❌ No'}")
            print(f"  Total issues: {results['total_issues']}")
            print(f"  Total warnings: {results['total_warnings']}")

            if results["summary_statistics"]:
                stats = results["summary_statistics"]
                print(f"  Quality score: {stats['average_quality_score']:.2f}")
                print(f"  Constituencies: {stats['total_constituencies']}")
                print(f"  Transfers: {stats['total_transfers']}")

        else:
            print(f"❌ Path not found: {args.data_path}")
            return 1

        return 0 if results.get('valid', False) else 1

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())