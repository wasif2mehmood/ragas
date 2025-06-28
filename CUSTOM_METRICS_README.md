# Custom Metrics for Ragas Evaluation

This document describes the custom metrics implemented for evaluating RAG (Retrieval-Augmented Generation) systems using the Ragas framework.

## Custom Metrics Implemented

### 1. Response Conciseness Metric

**Purpose**: Evaluates how concise and efficient a response is while maintaining completeness and clarity.

**Evaluation Criteria**:
- **Information Density**: Does the response convey essential information without unnecessary words?
- **Redundancy**: Is there any repetitive or redundant information?
- **Directness**: Does the response get straight to the point?
- **Completeness vs Brevity**: Does the response maintain completeness while being concise?
- **Clarity**: Is the response clear despite being concise?

**Scoring Scale** (0.0 to 1.0):
- **1.0**: Perfect conciseness - conveys all necessary information with minimal words
- **0.8-0.9**: Very concise - minor unnecessary words but overall efficient
- **0.6-0.7**: Moderately concise - some redundancy or wordiness
- **0.4-0.5**: Poor conciseness - significant redundancy or unnecessary elaboration
- **0.0-0.3**: Very poor conciseness - extremely verbose or repetitive

**Required Columns**: `user_input`, `response`, `reference`

### 2. Technical Accuracy Metric

**Purpose**: Assesses the technical correctness and accuracy of responses, focusing on factual details, terminology usage, and best practices.

**Evaluation Criteria**:
- **Factual Correctness**: Are the technical details and facts mentioned correct?
- **Terminology Usage**: Are technical terms used correctly and appropriately?
- **Conceptual Understanding**: Does the response demonstrate proper understanding of concepts?
- **Implementation Details**: If applicable, are implementation details accurate?
- **Best Practices**: Does the response align with industry best practices?

**Scoring Scale** (0.0 to 1.0):
- **1.0**: Completely accurate - all technical details are correct
- **0.8-0.9**: Mostly accurate - minor technical inaccuracies
- **0.6-0.7**: Somewhat accurate - some technical errors but core concepts correct
- **0.4-0.5**: Poor accuracy - significant technical errors
- **0.0-0.3**: Very poor accuracy - major technical misconceptions

**Required Columns**: `user_input`, `response`, `reference`, `retrieved_contexts`

## Usage

### Basic Usage

```python
from custom_metrics import ResponseConcisenessMetric, TechnicalAccuracyMetric
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Initialize LLM
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))

# Initialize custom metrics
response_conciseness = ResponseConcisenessMetric(llm=evaluator_llm)
technical_accuracy = TechnicalAccuracyMetric(llm=evaluator_llm)

# Use in evaluation
from ragas import evaluate

bulk_score = evaluate(
    dataset,
    metrics=[
        response_conciseness,
        technical_accuracy,
        # ... other standard metrics
    ]
)
```

### Individual Metric Testing

```python
import asyncio
from ragas.dataset_schema import SingleTurnSample

# Create sample
sample = SingleTurnSample(
    user_input="Your question here",
    response="AI response here",
    reference="Ground truth here",
    retrieved_contexts=["Context documents here"]
)

# Test metrics
async def test_metrics():
    conciseness_score = await response_conciseness.single_turn_ascore(sample, callbacks=None)
    accuracy_score = await technical_accuracy.single_turn_ascore(sample, callbacks=None)
    
    print(f"Conciseness: {conciseness_score}")
    print(f"Technical Accuracy: {accuracy_score}")

asyncio.run(test_metrics())
```

## Integration with Existing Evaluation Pipeline

The custom metrics are designed to integrate seamlessly with your existing Ragas evaluation pipeline in `run_evals.py`. They will be included in the evaluation alongside standard metrics like faithfulness, answer relevancy, etc.

The results will be saved to your CSV output file with additional columns for:
- `response_conciseness`: Conciseness scores
- `technical_accuracy`: Technical accuracy scores

## Benefits

1. **Response Conciseness**: Helps identify responses that are unnecessarily verbose or contain redundant information, improving user experience.

2. **Technical Accuracy**: Ensures that technical information provided by the RAG system is factually correct and uses appropriate terminology.

3. **Complementary Evaluation**: These metrics complement existing Ragas metrics by focusing on specific aspects not covered by standard metrics.

4. **Quality Improvement**: Provides actionable insights for improving response quality in technical domains.

## Customization

Both metrics can be easily customized by:
- Modifying the evaluation prompts
- Adjusting scoring criteria
- Adding domain-specific evaluation aspects
- Changing the scoring scale or interpretation

## Dependencies

- `ragas`: Core evaluation framework
- `langchain_openai`: For LLM integration
- `typing`: For type annotations
- `re`: For score extraction from LLM outputs
- `dataclasses`: For metric class definitions
