# Fact-Based QA Benchmark for LLM Evaluation

This project automatically generates a factual QA dataset from Wikipedia football league tables and evaluates large language models (LLMs) on their ability to recall structured facts without retrieval tools.

## Overview
The dataset includes automatically generated football (soccer) questions from verified Wikipedia data, each with a deterministic gold answer and difficulty label. This benchmark assesses factual reasoning, precision, and consistency across question types.

## Motivation
Most LLM benchmarks focus on open-text understanding rather than structured factual recall. This project addresses that gap by generating verifiable, domain-specific questions grounded in real-world statistics.

## Key Features
- **Automated Dataset Creation:** Scrapes 50+ league tables into structured QA pairs.  
- **Dynamic Difficulty Tagging:** Based on league popularity, season recency, and stat complexity.  
- **Four Question Families:** single-season placement, single-season stat, two-seasons placement, pairwise higher.  
- **Robust Evaluation:** Combines exact, fuzzy, and LLM-based checks.  
- **Performance Analytics:** Accuracy breakdowns by difficulty, type, and league.

## Architecture
```
Wikipedia tables
     ↓
generate_data.py → examples.jsonl
     ↓
run_eval.py → predictions.jsonl
     ↓
evaluation.py → results_clean.jsonl
     ↓
compute_accuracies.py → accuracy_breakdowns.csv
```

## Example
```
Prompt: Which team finished higher in Premier League 2022–23: Liverpool or Brighton?
Gold: Liverpool
Predicted: Liverpool
✅ Correct
```

## Skills Demonstrated
Python (pandas, requests, BeautifulSoup), NLP evaluation design, LLM API integration, and data analytics.

## Run Instructions
```bash
python generate_data.py        # Create dataset
python run_eval.py             # Generate model predictions
python evaluation.py           # Evaluate predictions
python compute_accuracies.py   # Compute accuracy breakdowns
```

## Author
Developed by **Rotem Bornstein**  
Technion – Data Science & Engineering, NLP Final Project (Spring 2025)
