# Ragas-Pub

A comprehensive publication evaluation pipeline using the Ragas framework with custom metrics for automated assessment of publication metadata generation systems.

## Overview

This repository provides a complete workflow for evaluating publication processing systems that generate titles, summaries (TL;DR), tags, and references. It uses multiple evaluation metrics including semantic similarity, faithfulness, and custom Jaccard similarity metrics designed specifically for publication data.

## Features

- **Custom Metrics**: Specialized metrics for publication evaluation
  - **Response Conciseness**: LLM-based evaluation of response efficiency
  - **Jaccard Similarity**: Set-based comparison for tags and structured data
  - **References Jaccard**: Specialized metric for reference URL and title comparison
- **Built-in Ragas Metrics**: Semantic similarity and faithfulness evaluation
- **Modular Architecture**: Separate metric files and utility functions
- **Comprehensive Evaluation**: Multi-dimensional assessment of publication metadata

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### Environment Setup
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure

```
ragas/
+---code
|   |   paths.py
|   |   ragas_evals.py
|   |
|   +---metrics
|   |   |   conciseness.py
|   |   |   generate_test_set.py
|   |   |   references_jaccard.py
|   |   |   tags_jaccard.py
|   |   |   utils.py
|   |   |
|   |           jaccard_similarity.cpython-311.pyc
|   |           references_jaccard.cpython-311.pyc
|   |           tags_jaccard.cpython-311.pyc
|   |           utils.cpython-311.pyc
|   |
|           paths.cpython-311.pyc
|
\---data
        complete_evaluation_results.csv
        evaluation_results.csv
        golden_dataset.csv
        golden_dataset.json
        golden_dataset_with_references.csv
        ragas_dataset.csv
        test_set.csv

```

## Usage

### 1. Prepare Your Dataset

Your evaluation dataset should be a CSV file with the following columns:
- `publication_external_id`: Unique identifier for each publication
- `title_truth` / `title_generated`: Ground truth and generated titles
- `tldr_truth` / `tldr_generated`: Ground truth and generated summaries
- `tags_truth` / `tags_generated`: Ground truth and generated tags (pipe-delimited: `tag1|tag2|tag3`)
- `references_truth` / `references_generated`: Ground truth and generated references (JSON format)

Example reference format:
```json
[{"url": "https://example.com", "title": "Example Paper"}, {"url": "https://another.com", "title": "Another Paper"}]
```

### 2. Run Evaluation

Execute the evaluation pipeline:
```bash
python run_evals.py
```

This will:
- Load your publication dataset
- Evaluate each field using multiple metrics:
  - **Semantic Similarity**: Measures semantic closeness between generated and ground truth text
  - **Faithfulness**: Evaluates whether generated content is grounded in the source publication
  - **Jaccard Similarity**: Compares overlap for tags and references
- Save detailed results to CSV files

### 3. Review Results

The evaluation generates two output files:
- `data/evaluation_results.csv`: Metric scores only
- `data/complete_evaluation_results.csv`: Original data + metric scores

## Metrics Explained

### Core Metrics

1. **Semantic Similarity (0-1)**
   - Measures semantic closeness using embeddings
   - Higher scores indicate better content match

2. **Faithfulness (0-1)**
   - Evaluates grounding in source content
   - Higher scores indicate less hallucination

3. **Jaccard Similarity (0-1)**
   - Set-based overlap comparison
   - Perfect for tags and structured data
   - Formula: |intersection| / |union|

### Specialized Metrics

4. **Response Conciseness (0-1)**
   - LLM-based evaluation of response efficiency
   - Considers information density, redundancy, and clarity
   - Available for integration with conciseness evaluation

5. **References Jaccard**
   - Specialized for reference comparison
   - Separately evaluates URL and title overlap
   - Returns average of both scores

## Evaluation Fields

The system evaluates four key publication fields:

| Field | Semantic Similarity | Faithfulness | Jaccard Similarity |
|-------|-------------------|--------------|-------------------|
| **Title** | ✅ | ✅ | ❌ |
| **TL;DR** | ✅ | ✅ | ❌ |
| **Tags** | ✅ | ✅ | ✅ |
| **References** | ✅ | ✅ | ✅ |

## Custom Metric Details

### Tags Jaccard Metric
- Delimiter: `|` (pipe-separated)
- Case-insensitive comparison
- Handles empty tag sets gracefully

### References Jaccard Metric
- Parses JSON-formatted reference lists
- Compares URLs and titles separately
- Averages URL and title Jaccard scores
- Handles malformed JSON gracefully

## Customization

### Adding New Metrics
1. Create a new metric file in `src/`
2. Inherit from `SingleTurnMetric` or `MetricWithLLM`
3. Implement the `_single_turn_ascore` method
4. Import and use in `run_evals.py`

### Modifying Evaluation Logic
- Edit field evaluation in `run_evals.py`
- Customize prompts for faithfulness evaluation
- Adjust metric parameters in factory functions

### Utility Functions
All utility functions are centralized in `src/utils.py`:
- Data loading and preprocessing
- Result formatting and saving
- Context truncation for large texts
- Score calculation and summary statistics



### Performance Tips

- Use smaller models for faster evaluation (`gpt-3.5-turbo` vs `gpt-4`)
- Process datasets in batches for large evaluations
- Cache embeddings for repeated evaluations
- Use context truncation for memory management

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Add your custom metrics or improvements
4. Submit a pull request

### Adding New Metrics
Follow the existing pattern:
- Create metric file in `src/`
- Add factory functions
- Update imports in `run_evals.py`
- Document in README

## License

This project is open source. Feel free to use and modify for your evaluation needs.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the metric documentation
- Submit an issue on the repository
