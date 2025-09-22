# Scripts Directory

This directory contains utility scripts for working with the STV transfer analysis package.

## Available Scripts

### Data Collection
- `download_election_data.py` - Download electoral data from official sources
- `validate_data.py` - Validate downloaded data for completeness and consistency

### Analysis
- `run_transfer_analysis.py` - Run complete transfer analysis pipeline
- `generate_reports.py` - Generate summary reports and visualizations

### Model Management
- `fit_models.py` - Fit Bayesian models with different configurations
- `compare_models.py` - Compare model performance using information criteria

### Utilities
- `setup_environment.py` - Set up analysis environment and dependencies
- `clean_data.py` - Clean and preprocess raw electoral data

## Usage

Most scripts can be run from the project root directory:

```bash
# Set up Python path
export PYTHONPATH=src:$PYTHONPATH

# Run a script
python scripts/download_election_data.py --election 2020-general

# Get help for any script
python scripts/script_name.py --help
```

## Configuration

Scripts look for configuration files in the following order:
1. Command line arguments
2. `config/` directory files
3. Environment variables
4. Default values

See individual script documentation for specific configuration options.