# Amazon ML Challenge 2026

## Team Name
<your team name>

## Problem Statement
Business Entity Resolution

## Methodology

### Data Preprocessing
- Lowercasing
- Unicode normalization
- Address cleaning
- Country normalization

### Candidate Generation
- Name token blocking
- Address token blocking
- Domain token blocking
- Country filtering

Candidate Recall: 96.80%

### Feature Engineering
23 matching features

- Exact matches
- Similarity scores
- Token overlap
- Containment similarity
- Address number matching
- Missing indicators
- Length features

### Model
Random Forest Classifier

Parameters:
- n_estimators=200
- max_depth=12
- class_weight=balanced

### Validation

Candidate Recall:
96.80%

Optimized for F0.5 score.

### Output Generation

1. Generate candidate pairs
2. Compute matching features
3. Predict probabilities
4. Apply threshold
5. Generate final matches

### Files Submitted

- matching_results.tsv
- candidate_pairs.tsv
- Source code
- Documentation