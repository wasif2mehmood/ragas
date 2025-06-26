# Ragas-Pub

A comprehensive RAG (Retrieval-Augmented Generation) evaluation pipeline using the Ragas framework for automated assessment of RAG system performance.

## Overview

This repository provides a complete workflow for evaluating RAG systems using multiple metrics including faithfulness, answer relevancy, context precision, context recall, and more.

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

## Usage

### 1. Generate Seed Dataset
Run the test set generation script to create your initial evaluation dataset:
```bash
python generate_test_set.py
```
This will create a baseline dataset for evaluation.

### 2. Generate RAG Pipeline Outputs
Integrate your RAG pipeline and generate responses:
```bash
python run_pipeline.py
```
**Note:** You need to integrate your specific RAG pipeline implementation in this file. The script should:
- Load the seed dataset
- Process queries through your RAG system
- Generate answers and retrieve relevant contexts
- Save results to `./data/ragas_dataset.csv`

### 3. Run Evaluations
Execute the evaluation pipeline to assess your RAG system:
```bash
python run_evals.py
```
This will:
- Load the dataset with RAG outputs
- Evaluate using multiple Ragas metrics:
  - **Faithfulness**: Measures hallucination in generated answers
  - **Answer Relevancy**: Assesses how relevant answers are to questions
  - **Context Precision**: Evaluates retrieval quality
  - **Context Recall**: Measures completeness of retrieved context
  - **Context Entity Recall**: Checks entity coverage in context
  - **Answer Similarity**: Compares semantic similarity to reference answers
  - **Answer Correctness**: Overall answer quality assessment
- Save evaluation scores to `ragas_scores.csv`

## Dataset Format

Your dataset should be in CSV format with the following columns:
- `question`: The input query
- `answer`: Generated response from your RAG system
- `contexts`: List of retrieved context passages (will be parsed from string)
- `ground_truth`: Reference answer (optional, for some metrics)
- `reference_contexts`: Ground truth contexts (optional, will be parsed from string)

## Output

The evaluation results are saved to `ragas_scores.csv` containing:
- All original dataset columns
- Individual metric scores for each question
- Overall performance metrics

## Metrics Explained

- **Faithfulness (0-1)**: Higher scores indicate less hallucination
- **Answer Relevancy (0-1)**: Higher scores mean more relevant answers
- **Context Precision (0-1)**: Higher scores indicate better retrieval precision
- **Context Recall (0-1)**: Higher scores mean better context coverage
- **Answer Correctness (0-1)**: Overall answer quality score

## Customization

To adapt this pipeline for your RAG system:
1. Modify `run_pipeline.py` to integrate your specific RAG implementation
2. Ensure your output dataset matches the expected format
3. Adjust evaluation metrics in `run_evals.py` as needed

## Troubleshooting

- **OpenAI API Error**: Ensure your API key is correctly set in the `.env` file
- **Data Format Issues**: Check that list columns (contexts, reference_contexts) are properly formatted
- **Memory Issues**: For large datasets, consider processing in batches

## Contributing

Feel free to submit issues and enhancement requests!