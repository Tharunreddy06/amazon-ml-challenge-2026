# Amazon ML Challenge 2026 - Entity Resolution

## Problem Statement
Match business entities across multiple sources and identify records referring to the same real-world business.

## Dataset
### Training
- train_source1.tsv
- train_source2.tsv
- train_source3.tsv
- train_ground_truth.tsv

### Test
- test_source1.tsv
- test_source2.tsv
- test_source3.tsv

## Approach

### Preprocessing
- Text normalization
- Lowercasing
- Address cleaning
- Country normalization

### Candidate Generation
- Multi-strategy blocking
- Name blocking
- Address blocking
- Country filtering

Candidate Recall:
96.80%

### Feature Engineering
23 matching features:
- Exact matches
- Similarity features
- Token overlap features
- Containment features
- Missing value indicators
- Length features

### Model
Random Forest Classifier

Parameters:
- n_estimators=200
- max_depth=12
- class_weight=balanced

### Inference Pipeline
1. Build Source1 SQLite index
2. Generate candidates
3. Compute matching features
4. Predict match probabilities
5. Select best match
