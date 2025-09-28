# Transfer Data Quality Issues & Fixing Plan

## 🔍 **Issues Identified**

Based on analysis of the scraped transfer data, there are three critical issues that need to be fixed:

### Issue 1: Inconsistent Candidate Names
- **Problem**: `from_candidate` uses surname-only ("O'Donoghue", "Collins", "Funchion")
- **Problem**: `to_candidate` uses full names ("Patrick O'Donovan", "Niall Collins", "Kathleen Funchion")
- **Impact**: Makes it impossible to properly link transfer sources and destinations

### Issue 2: Missing Constituency IDs
- **Problem**: Transfer records don't include constituency identifiers
- **Impact**: Cannot distinguish transfers from different constituencies or merge data properly

### Issue 3: Missing Elimination Order
- **Problem**: No sequence information indicating when candidates were eliminated
- **Impact**: Cannot reconstruct the elimination timeline or model transfer dynamics properly

## 🛠️ **Comprehensive Fixing Plan**

### Phase 1: Candidate Name Standardization System

**1.1 Create Candidate Name Mapping**
```python
# For each constituency, create mapping: surname → full_name
{
  "O'Donoghue": "Richard O'Donoghue",
  "Collins": "Niall Collins",
  "Funchion": "Kathleen Funchion"
}
```

**1.2 Handle Edge Cases**
- Multiple candidates with same surname (e.g., "Collins" in a constituency with multiple Collins candidates)
- Compound surnames (O'Sullivan, Ní Mhaoldhomhnaigh)
- Name variations and spellings
- Special characters and fada marks

**1.3 Ambiguity Resolution**
- Use transfer patterns to disambiguate (eliminated candidates can't receive transfers)
- Cross-reference with elimination sequence
- Flag unresolvable ambiguities for manual review

### Phase 2: Constituency ID Integration

**2.1 Add Constituency Context**
```python
# Enhanced transfer record format
{
  "constituency_id": "5005",
  "constituency_name": "Limerick",
  "from_candidate": "Richard O'Donoghue",  # Standardized
  "to_candidate": "Niall Collins",
  "transfer_count": 195.0,
  "transfer_type": "elimination",
  "transfer_value": 1.0,
  "elimination_order": 3  # New field
}
```

### Phase 3: Elimination Order Extraction

**3.1 Parse HTML Transfer Tables More Deeply**
- Re-examine the raw HTML to extract count-by-count information
- Identify elimination events and their sequence
- Map eliminations to transfer batches

**3.2 Infer Elimination Order from Transfer Patterns**
- Analyze transfer sequences to determine elimination order
- Use STV rules: eliminated candidates don't receive further transfers
- Cross-validate with vote totals and quota information

**3.3 Create Elimination Timeline**
```python
# Per constituency elimination sequence
{
  "constituency_id": "5005",
  "eliminations": [
    {"order": 1, "candidate": "First Eliminated", "count_round": 1},
    {"order": 2, "candidate": "Second Eliminated", "count_round": 2},
    # ...
  ]
}
```

### Phase 4: Implementation Strategy

**4.1 Enhanced HTML Parser**
- Modify `parse_count_details` to extract more structural information
- Parse count-by-count tables to get elimination sequence
- Extract candidate vote progressions across rounds

**4.2 Data Cleaning Pipeline**
```python
scripts/clean_transfer_data.py
├── load_raw_data()
├── create_candidate_mappings()
├── resolve_name_ambiguities()
├── add_constituency_context()
├── extract_elimination_orders()
├── validate_cleaned_data()
└── save_enhanced_dataset()
```

**4.3 Quality Assurance**
- Validate that all transfers have valid source/destination mappings
- Verify elimination sequences make sense (no posthumous transfers)
- Check transfer count conservation
- Cross-validate against first preference totals

## 📋 **Detailed Implementation Steps**

### Step 1: Candidate Name Mapping System
```python
def create_candidate_mapping(constituency_data):
    """Create surname → full_name mapping for a constituency"""
    full_names = [c['candidate'] for c in constituency_data['first_prefs']]
    mapping = {}

    for full_name in full_names:
        surname = extract_surname(full_name)
        if surname in mapping:
            # Handle duplicate surnames
            mapping[surname] = resolve_ambiguity(mapping[surname], full_name)
        else:
            mapping[surname] = full_name

    return mapping

def extract_surname(full_name):
    """Extract surname handling Irish naming conventions"""
    # Handle cases like "Patrick O'Donovan", "Seán Ó Fearghaíl"
    parts = full_name.split()
    if len(parts) >= 2 and parts[-2] in ["O'", "Ó", "Mac", "Mc"]:
        return " ".join(parts[-2:])  # "O' Donovan"
    return parts[-1]  # "Collins"
```

### Step 2: Enhanced HTML Parsing for Elimination Order
```python
def parse_elimination_sequence(html_content):
    """Parse detailed count-by-count information"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Look for count progression tables
    count_tables = soup.find_all('table', class_='rtable')

    eliminations = []
    for table in count_tables:
        # Parse each count round
        # Extract which candidates were eliminated when
        # Build elimination timeline
        pass

    return eliminations
```

### Step 3: Data Enhancement Pipeline
```python
def enhance_transfer_data(raw_data):
    """Complete data enhancement pipeline"""

    enhanced_data = copy.deepcopy(raw_data)

    for year, election in enhanced_data.items():
        for const_id, constituency in election['constituencies'].items():

            # 1. Create candidate name mapping
            name_mapping = create_candidate_mapping(constituency)

            # 2. Standardize transfer candidate names
            for transfer in constituency['transfer_events']:
                transfer['from_candidate'] = name_mapping.get(
                    transfer['from_candidate'],
                    transfer['from_candidate']
                )
                # Add constituency context
                transfer['constituency_id'] = const_id
                transfer['constituency_name'] = constituency['name']

            # 3. Extract and add elimination order
            elimination_order = extract_elimination_sequence(const_id, year)
            for transfer in constituency['transfer_events']:
                transfer['elimination_order'] = get_elimination_order(
                    transfer['from_candidate'], elimination_order
                )

    return enhanced_data
```

### Step 4: Validation Framework
```python
def validate_enhanced_data(enhanced_data):
    """Comprehensive validation of cleaned data"""

    issues = []

    for year, election in enhanced_data.items():
        for const_id, constituency in election['constituencies'].items():

            # Check name consistency
            first_pref_names = {c['candidate'] for c in constituency['first_prefs']}
            transfer_names = {t['from_candidate'] for t in constituency['transfer_events']}
            transfer_names.update({t['to_candidate'] for t in constituency['transfer_events']})

            # Validate all transfer names exist in first preferences
            invalid_names = transfer_names - first_pref_names - {'non-transferable'}
            if invalid_names:
                issues.append(f"{const_id}: Invalid names: {invalid_names}")

            # Check elimination order logic
            eliminations = sorted(
                [(t['elimination_order'], t['from_candidate'])
                 for t in constituency['transfer_events']
                 if t['elimination_order'] is not None],
                key=lambda x: x[0]
            )

            # Validate no candidate receives transfers after elimination
            # ... additional validation logic

    return issues
```

## 🎯 **Expected Outcomes**

After implementing this plan:

1. **Consistent Naming**: All candidate names standardized across first preferences and transfers
2. **Complete Context**: Every transfer record includes constituency ID and name
3. **Temporal Information**: Elimination order enables proper sequence modeling
4. **Data Integrity**: Validated transfers with no orphaned or invalid references
5. **Research Ready**: Clean dataset suitable for STV transfer modeling

## 📁 **New File Structure**
```
data/
├── processed/
│   ├── {year}/
│   │   ├── enhanced_transfers.json      # Cleaned transfer data
│   │   ├── elimination_sequences.json  # Elimination timelines
│   │   ├── candidate_mappings.json     # Name mapping dictionaries
│   │   └── validation_report.json      # Data quality metrics
└── cleaned/
    └── {year}/
        └── final_dataset.json          # Research-ready dataset
```

## 🚀 **Implementation Priority**

1. **HIGH**: Candidate name standardization (breaks current analysis)
2. **HIGH**: Constituency ID addition (needed for data integrity)
3. **MEDIUM**: Elimination order extraction (enhances modeling capability)
4. **LOW**: Additional validation and quality metrics

This plan addresses all identified issues systematically while maintaining data integrity and enabling proper STV transfer analysis.