# Test Fixtures

This directory contains test data files for the stv_transfers package tests.

## Structure

- `sample_elections/` - Sample election data files
- `transfer_data/` - Test transfer matrices and flow data
- `candidate_data/` - Test candidate information
- `validation_data/` - Known-good results for validation tests

## Usage

Test fixtures should be loaded using the fixture functions in `conftest.py` rather than directly accessing files. This ensures consistent data loading and makes tests more maintainable.

## Data Format

All test data should follow the same format as real electoral data to ensure comprehensive testing of the parsing and analysis pipelines.