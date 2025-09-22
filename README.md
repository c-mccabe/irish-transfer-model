# Irish Transfer Model: Bayesian Analysis of STV Transfer Patterns

A comprehensive Python package for analyzing Single Transferable Vote (STV) transfer patterns in Irish elections using Bayesian statistical modeling. This package provides tools for scraping electoral results, building statistical models, and conducting robust inference on voter transfer behavior while properly accounting for uncertainty and identifiability constraints.

## Project Summary

STV transfer analysis involves estimating how voters' preferences flow between candidates when their first-choice candidate is eliminated or when surplus votes are distributed. This package provides:

- **Web scraping tools** for collecting Irish electoral data from official sources
- **Bayesian modeling** of transfer patterns using NumPyro and JAX
- **STV simulation** capabilities for validation and counterfactual analysis
- **Comprehensive testing** with integration and unit test suites
- **Uncertainty quantification** for all transfer estimates
- **Statistical diagnostics** for model validation and convergence checking

The core challenge in transfer analysis is that individual-level transfer decisions are typically unobserved, requiring statistical inference from aggregate count data. Our Bayesian approach naturally handles this uncertainty and allows for incorporation of prior knowledge about voter behavior patterns.

## Installation

### Prerequisites

This package requires Python 3.10 or later and JAX for numerical computing. **Important**: JAX installation is platform-specific and not automatically handled by this package.

#### Installing JAX

**Before installing this package**, install the appropriate JAX version for your platform:

```bash
# For CPU-only (works on all platforms)
pip install "jax[cpu]"

# For CUDA (NVIDIA GPUs)
pip install "jax[cuda12]"  # or jax[cuda11] for older CUDA versions

# For Apple Silicon Macs
pip install "jax[metal]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for detailed platform-specific instructions.

#### Windows Users

**Windows Subsystem for Linux (WSL) is strongly recommended** for Windows users. JAX has limited native Windows support, and NumPyro works best in Unix-like environments.

### Package Installation

Once JAX is installed, install the package:

```bash
# Clone the repository
git clone https://github.com/your-username/stv-transfers.git
cd stv-transfers

# Install in development mode
pip install -e ".[dev]"

# Or install from PyPI (when available)
pip install stv-transfers
```

### Development Setup

For development work:

```bash
# Set up the package for development
export PYTHONPATH=src:$PYTHONPATH

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/
```

## Complete Workflow Example

This section demonstrates the full analysis pipeline from data collection to results interpretation.

### 1. Data Collection and Validation

```python
import json
from pathlib import Path
from stv_transfers.scraper import ElectionScraper

# Initialize scraper for Irish electoral data
scraper = ElectionScraper("https://www.rte.ie/news/election-2020")

# Scrape electoral results (handles pagination and data extraction)
constituency_data = scraper.scrape_all_constituencies()

# Save raw data
with open("data/raw/election_2020.json", "w") as f:
    json.dump(constituency_data, f, indent=2)

# Validate data quality
from scripts.validate_scraper import DataValidator
validator = DataValidator()
validation_results = validator.validate_directory(Path("data/raw/"))

print(f"Data quality score: {validation_results['summary_statistics']['average_quality_score']:.2f}")
```

### 2. Data Preprocessing and Model Setup

```python
from stv_transfers.data_structures import ModelData
from stv_transfers.model import build_model_exact

# Convert scraped data to model format
model_data = scraper.convert_to_model_format(constituency_data)

print(f"Loaded {model_data.n_constituencies} constituencies")
print(f"Found {model_data.n_sources} source candidates")
print(f"Analyzing {len(model_data.events)} transfer events")

# Build Bayesian model
bayesian_model = build_model_exact(model_data)
```

### 3. MCMC Inference with Diagnostics

```python
import jax.random as random
from numpyro.infer import MCMC, NUTS
from stv_transfers.diagnostics import compute_rhat, effective_sample_size

# Configure MCMC sampler
nuts_kernel = NUTS(bayesian_model)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    progress_bar=True
)

# Run inference
rng_key = random.PRNGKey(42)
mcmc.run(rng_key)

# Extract posterior samples
samples = mcmc.get_samples()

# Check convergence diagnostics
rhat = compute_rhat(samples['mu'])
ess = effective_sample_size(samples['mu'])

print(f"Max R-hat: {rhat.max():.3f} (should be < 1.1)")
print(f"Min ESS: {ess.min():.0f} (should be > 100)")

if rhat.max() > 1.1:
    print("⚠️  Warning: Chains may not have converged")
```

### 4. Results Analysis and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from stv_transfers.simulator import STVSimulator

# Extract posterior mean transfer matrix
mu_samples = samples['mu']  # Shape: (samples, sources, destinations)
posterior_mean = np.mean(mu_samples, axis=0)
posterior_std = np.std(mu_samples, axis=0)

# Create transfer matrix visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot mean transfer probabilities
im1 = ax1.imshow(posterior_mean, cmap='viridis', vmin=0, vmax=1)
ax1.set_title('Mean Transfer Probabilities')
ax1.set_xlabel('To Candidate')
ax1.set_ylabel('From Candidate')
plt.colorbar(im1, ax=ax1)

# Plot uncertainty (coefficient of variation)
cv = posterior_std / (posterior_mean + 1e-6)
im2 = ax2.imshow(cv, cmap='Reds')
ax2.set_title('Transfer Uncertainty')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('transfer_analysis.png', dpi=300)

# Run simulation with inferred parameters
simulator = STVSimulator(posterior_mean, seed=42)
ballots = [Ballot([0, 1, 2], 1.0) for _ in range(1000)]
result = simulator.count_stv(ballots, seats=3)

print(f"Simulated election winners: {result['elected']}")
```

### 5. Automated Pipeline Execution

For production analysis, use the provided scripts:

```bash
# Run complete MCMC analysis pipeline
python scripts/run_analysis.py \
    --data-dir data/processed/ \
    --results-dir data/results/ \
    --warmup 1000 \
    --samples 2000 \
    --chains 4

# Validate all data files
python scripts/validate_scraper.py data/processed/ \
    --output-dir data/validation/

# Check results
ls data/results/
# Shows: metadata_*.json, samples_*.npz, posterior_predictive_*.json, report_*.txt
```

## Interpreting High-Uncertainty Transfers

Transfer probability estimates come with uncertainty that researchers must understand and communicate properly:

### Sources of Uncertainty

1. **Statistical uncertainty**: Limited sample sizes in transfer counts (common in real elections)
2. **Model uncertainty**: Assumptions about voter behavior and independence
3. **Identifiability constraints**: Some transfer patterns may be poorly identified from the data
4. **Measurement error**: Inaccuracies in vote counting and data entry

### Understanding Credible Intervals

- **Narrow intervals** (±0.02): High confidence, reliable for interpretation
- **Wide intervals** (±0.15): Substantial uncertainty, interpret with caution
- **Very wide intervals** (±0.30+): Essentially uninformative, likely identifiability issues
- **Asymmetric intervals**: May indicate boundary effects (probabilities constrained to [0,1])

### Critical Warning Signs

```python
# Check convergence diagnostics
rhat = compute_rhat(samples['mu'])
ess = effective_sample_size(samples['mu'])

# Flag problematic parameters
high_rhat = rhat > 1.1
low_ess = ess < 100
wide_intervals = posterior_std > 0.20

if high_rhat.any():
    print("⚠️  Convergence issues detected - results unreliable")
if low_ess.any():
    print("⚠️  Low effective sample size - need longer chains")
if wide_intervals.any():
    print("⚠️  High uncertainty - some transfers poorly identified")
```

### Recommendations for High-Uncertainty Transfers

When posterior intervals are very wide (>0.20):

1. **Do not over-interpret**: Avoid strong claims about specific transfer rates
2. **Focus on robust patterns**: Look for consistent trends across multiple transfers
3. **Report uncertainty**: Always include credible intervals in results
4. **Consider aggregation**: Analyze party-level rather than candidate-level transfers
5. **Seek additional data**: Combine multiple elections or incorporate survey data

## Identifiability and Pooling

**Critical Warning**: Transfer patterns are often **not identifiable** without pooling or strong priors. This occurs when:

- Transfer counts are small (common in real elections)
- Many candidates compete (creating sparse transfer matrices)
- Similar candidates exist (confounding transfer destinations)

### Recommended Strategies

1. **Hierarchical pooling**: Share information across similar candidates
   ```python
   model = TransferModel(pooling_strategy="party_hierarchical")
   ```

2. **Party-level analysis**: Aggregate transfers by party affiliation
   ```python
   model = TransferModel(level="party")
   ```

3. **Informative priors**: Incorporate external knowledge
   ```python
   model = TransferModel(prior_transfers=expert_estimates)
   ```

4. **Regularization**: Prevent overfitting with sparse data
   ```python
   model = TransferModel(regularization=0.1)
   ```

### When Pooling Fails

If uncertainties remain very large even with pooling:
- Consider longer time series (multiple elections)
- Seek additional data sources (polls, surveys)
- Focus on aggregate patterns rather than individual transfers
- Report uncertainty honestly rather than overinterpreting point estimates

## Irish STV Electoral Law Reference

The Single Transferable Vote system used in Irish elections follows specific legal rules that this analysis assumes:

### Legal Framework

**Irish Constitution (Bunreacht na hÉireann)**
- Article 16.2.5°: PR-STV for Dáil elections
- Article 28A: PR-STV for European Parliament elections

**Electoral Acts**
- Electoral Act 1992 (as amended): Detailed STV counting rules
- Electoral (Amendment) Act 2001: Electronic counting provisions
- Presidential Elections Act 1993: STV for presidential elections

### STV Counting Rules (Per Electoral Act 1992)

**Quota Calculation (Section 118)**
```
Quota = floor(Valid Poll / (Seats + 1)) + 1
```
This is the Droop quota used throughout Irish elections.

**Transfer Rules (Sections 119-121)**

1. **Surplus Distribution**: When a candidate exceeds quota
   - Transfer value = Surplus ÷ Total transferable papers
   - Only papers that elected the candidate are examined
   - Gregory method used (fractional transfers)

2. **Elimination Transfers**: When a candidate is eliminated
   - All papers counted at full transfer value (1.0)
   - Papers transferred in reverse order of receipt
   - Non-transferable papers go to "exhausted pile"

3. **Transfer Order**: Preferences examined sequentially until transferable preference found

**Key Legal Assumptions in This Model**

- Voters mark preferences independently (no vote coordination)
- Transfer behavior reflects genuine voter preferences
- Counting is accurate and follows prescribed rules
- No tactical voting specifically targeting transfer patterns

**Important Legal Note**: This analysis is for research purposes only. Official election results must follow the exact procedures specified in electoral law and be conducted by authorized returning officers.

### References

- *Bunreacht na hÉireann* (Constitution of Ireland), 1937
- Electoral Act 1992 (No. 23 of 1992), as amended
- Department of Housing, Local Government and Heritage: *Electoral Acts*
- Houses of the Oireachtas: *Electoral Law and Procedure*

## Model Limitations and Warnings

### Critical Data Limitations

**⚠️  Censored Transfer Data**: This model has significant limitations due to data censoring in STV elections:

1. **Unobserved Preferences**: Lower preferences on ballots are only observed when higher preferences are eliminated
2. **Complex Dependencies**: Transfer patterns depend on elimination order, which varies by election
3. **Strategic Voting**: Voters may mark preferences strategically, violating independence assumptions
4. **Limited Observations**: Many transfer patterns have very few observations, leading to high uncertainty

### When Model Results Are Unreliable

**Do not use this model when**:
- Posterior uncertainty intervals exceed ±0.30
- R-hat statistics exceed 1.2 for key parameters
- Effective sample sizes are below 50
- Posterior predictive checks show systematic misfit
- Transfer counts are consistently below 20 per pattern

**Exercise extreme caution when**:
- Analyzing candidates with limited transfer events
- Making predictions for future elections with different candidates
- Interpreting transfers involving new or minor parties
- Combining data across different electoral contexts

### Statistical Health Warnings

1. **Correlation vs Causation**: Inferred transfer patterns reflect association, not causal voter behavior
2. **Out-of-Sample Prediction**: Model may not generalize to future elections with different candidates/context
3. **Multiple Comparisons**: When examining many transfer patterns simultaneously, some apparent effects may be spurious
4. **Model Assumptions**: Results depend critically on assumptions about voter independence and data quality

### Responsible Use Guidelines

✅ **Appropriate Uses**:
- Exploratory analysis of historical transfer patterns
- Identifying broad trends in voter behavior
- Academic research with proper uncertainty quantification
- Comparative studies across elections or constituencies

❌ **Inappropriate Uses**:
- Making definitive claims about specific transfer rates
- Predicting election outcomes without acknowledging uncertainty
- Tactical advice for candidates or parties
- Any use ignoring substantial posterior uncertainty

**Always report credible intervals and clearly communicate uncertainty in any results.**

## Citation and Legal Disclaimer

### Academic Citation

If you use this package in research, please cite:

```bibtex
@software{irish_transfer_model,
  title = {Irish Transfer Model: Bayesian Analysis of STV Transfer Patterns},
  author = {Research Team},
  year = {2024},
  url = {https://github.com/your-username/irish-transfer-model},
  note = {Software for analyzing Single Transferable Vote transfer patterns in Irish elections}
}
```

### Legal Disclaimer

**This software is provided for academic and research purposes only.**

- Results should not be used for commercial electoral advice or campaign strategy
- The model's assumptions may not hold in all electoral contexts
- Users are responsible for validating results and communicating uncertainty appropriately
- This analysis does not constitute official electoral analysis or prediction
- The software is provided "as is" without warranty of any kind

### Data Usage Compliance

When using data scraped from official sources:
- Comply with terms of service of data providers
- Respect copyright and intellectual property rights
- Acknowledge data sources in publications
- Follow ethical guidelines for data collection and analysis

### Ethical Considerations

- Report all relevant uncertainty and limitations
- Avoid overstating the precision or reliability of results
- Consider potential impacts on democratic processes
- Ensure transparency in methodology and assumptions

## Testing and Validation

This package includes comprehensive test suites:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_simulator.py          # STV simulation tests
pytest tests/test_integration.py        # End-to-end pipeline tests
pytest tests/test_scraper.py           # Data collection tests

# Run with coverage reporting
pytest --cov=stv_transfers tests/

# Check for slow integration tests
pytest -m "not slow"                   # Skip slow tests
pytest -m "slow"                       # Run only slow tests
```

### Continuous Integration

The project uses automated testing to ensure code quality:
- Unit tests for all core functionality
- Integration tests for complete pipeline
- Code coverage monitoring
- Style and type checking with `ruff` and `mypy`

## Contributing

Contributions welcome! Please follow these guidelines:

1. **Fork and clone** the repository
2. **Create a feature branch** for your changes
3. **Add tests** for new functionality
4. **Run the test suite** to ensure nothing breaks
5. **Update documentation** as needed
6. **Submit a pull request** with clear description

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all public functions
- Include docstrings for modules, classes, and functions
- Write tests for new features
- Maintain backward compatibility when possible

## License

**MIT License** - see `LICENSE` file for full details.

### Third-Party Dependencies

This package builds on excellent open-source software:
- **JAX**: High-performance numerical computing
- **NumPyro**: Probabilistic programming on JAX
- **NumPy**: Fundamental array computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **pytest**: Testing framework

See `pyproject.toml` for complete dependency list and version constraints.
