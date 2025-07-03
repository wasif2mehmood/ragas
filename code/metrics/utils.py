import pandas as pd
import json
from paths import (
    GOLDEN_DATASET_CSV_STR,
    GOLDEN_DATASET_JSON_STR,
    EVALUATION_RESULTS_CSV_STR,
    COMPLETE_EVALUATION_RESULTS_CSV_STR
)


def truncate_context(text, max_tokens=8000):
    """
    Truncate context to fit within token limits.
    Rough estimation: 1 token ≈ 4 characters for English text.
    
    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated text string
    """
    if not text:
        return ""
    
    max_chars = max_tokens * 4  # Conservative estimate
    if len(text) <= max_chars:
        return text
    
    # Truncate and try to end at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:  # If we can find a period in the last 20%
        return truncated[:last_period + 1]
    else:
        return truncated + "..."


def load_dataset(csv_path=GOLDEN_DATASET_CSV_STR):
    """
    Load the main evaluation dataset.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """


    return pd.read_csv(csv_path, nrows=1)


def load_publication_descriptions(json_path=GOLDEN_DATASET_JSON_STR):
    """
    Load publication descriptions from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        dict: Mapping from publication_external_id to publication_description
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    
    return {
        item['publication_external_id']: item['publication_description'] 
        for item in golden_data
    }


class PublicationSample:
    """
    Simple sample class for publication data.
    Dynamically creates attributes from keyword arguments.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def format_score(score):
    """
    Format score for display.
    
    Args:
        score: Numeric score or None
        
    Returns:
        str: Formatted score string
    """
    return f"{score:.3f}" if score is not None else "N/A"


def print_evaluation_summary(results_df):
    """Print comprehensive evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Publications Evaluated: {len(results_df)}")
    print("Results saved to: evaluation_results.csv")
    print("Complete results saved to: complete_evaluation_results.csv")
    
    print("\nMETRIC STATISTICS:")
    print("-" * 50)
    
    # List of all metrics to display
    metrics = [
        ('title_semantic_similarity', 'Title Semantic Similarity'),
        ('tldr_semantic_similarity', 'Tldr Semantic Similarity'),
        ('references_semantic_similarity', 'References Semantic Similarity'),
        ('tags_semantic_similarity', 'Tags Semantic Similarity'),
        ('references_jaccard_similarity', 'References Jaccard Similarity'),
        ('tags_jaccard_similarity', 'Tags Jaccard Similarity'),
        ('title_faithfulness', 'Title Faithfulness'),
        ('tldr_faithfulness', 'Tldr Faithfulness'),
        ('references_faithfulness', 'References Faithfulness'),
        ('tags_faithfulness', 'Tags Faithfulness'),
        ('content_coherence', 'Content Coherence')  # Add this line
    ]
    
    for metric_col, metric_name in metrics:
        if metric_col in results_df.columns:
            valid_scores = results_df[metric_col].dropna()
            if len(valid_scores) > 0:
                print(f"\n{metric_name}:")
                print(f"  Count: {len(valid_scores)}")
                print(f"  Mean:  {valid_scores.mean():.3f}")
                print(f"  Std:   {valid_scores.std():.3f}")
                print(f"  Range: {valid_scores.min():.3f} - {valid_scores.max():.3f}")

def save_evaluation_results(results, df):
    """
    Save evaluation results to CSV files.
    
    Args:
        results: List of result dictionaries
        df: Original DataFrame
        
    Returns:
        tuple: (results_df, complete_results_df)
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original data for complete results
    complete_results = df.merge(results_df, on='publication_external_id', how='left')
    
    # Save results using paths from paths.py
    results_df.to_csv(EVALUATION_RESULTS_CSV_STR, index=False)
    complete_results.to_csv(COMPLETE_EVALUATION_RESULTS_CSV_STR, index=False)
    
    return results_df, complete_results


def calculate_metric_statistics(results_df):
    """
    Calculate statistics for all evaluation metrics.
    
    Args:
        results_df: DataFrame containing evaluation results
        
    Returns:
        dict: Statistics for each metric
    """
    metric_columns = [
        'title_semantic_similarity', 'tldr_semantic_similarity',
        'references_semantic_similarity', 'tags_semantic_similarity',
        'references_jaccard_similarity', 'tags_jaccard_similarity',
        'title_faithfulness', 'tldr_faithfulness',
        'references_faithfulness', 'tags_faithfulness'
    ]
    
    stats = {}
    for col in metric_columns:
        non_null_values = results_df[col].dropna()
        if len(non_null_values) > 0:
            stats[col] = {
                'count': len(non_null_values),
                'mean': non_null_values.mean(),
                'std': non_null_values.std(),
                'min': non_null_values.min(),
                'max': non_null_values.max()
            }
    
    return stats


def print_evaluation_summary(results_df):
    """
    Print comprehensive evaluation summary.
    
    Args:
        results_df: DataFrame containing evaluation results
    """
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Publications Evaluated: {len(results_df)}")
    print(f"Results saved to: evaluation_results.csv")
    print(f"Complete results saved to: complete_evaluation_results.csv")
    
    # Calculate and display statistics
    stats = calculate_metric_statistics(results_df)
    
    print("\nMETRIC STATISTICS:")
    print("-" * 50)
    for metric, stat in stats.items():
        metric_name = metric.replace('_', ' ').title()
        print(f"\n{metric_name}:")
        print(f"  Count: {stat['count']}")
        print(f"  Mean:  {stat['mean']:.3f}")
        print(f"  Std:   {stat['std']:.3f}")
        print(f"  Range: {stat['min']:.3f} - {stat['max']:.3f}")


def initialize_result_dict(publication_external_id):
    """Initialize a result dictionary with None values for all metrics."""
    return {
        'publication_external_id': publication_external_id,
        'title_semantic_similarity': None,
        'title_faithfulness': None,
        'tldr_semantic_similarity': None,
        'tldr_faithfulness': None,
        'references_semantic_similarity': None,
        'references_jaccard_similarity': None,
        'references_faithfulness': None,
        'tags_semantic_similarity': None,
        'tags_jaccard_similarity': None,
        'tags_faithfulness': None,
        'content_coherence': None  # Add this line
    }

def prepare_text_for_semantic_similarity(text, field_type=None):
    """
    Prepare text for semantic similarity evaluation.
    
    Args:
        text: Input text
        field_type: Type of field ('tags' to replace | with commas)
        
    Returns:
        str: Processed text
    """
    text = str(text)
    if field_type == 'tags':
        text = text.replace('|', ', ')
    return text

def print_evaluation_scores(result):
    """Print evaluation scores for a single publication."""
    scores = [
        ("Title Semantic", result.get('title_semantic_similarity')),
        ("Title Faithfulness", result.get('title_faithfulness')),
        ("TLDR Semantic", result.get('tldr_semantic_similarity')),
        ("TLDR Faithfulness", result.get('tldr_faithfulness')),
        ("References Semantic", result.get('references_semantic_similarity')),
        ("References Jaccard", result.get('references_jaccard_similarity')),
        ("References Faithfulness", result.get('references_faithfulness')),
        ("Tags Semantic", result.get('tags_semantic_similarity')),
        ("Tags Jaccard", result.get('tags_jaccard_similarity')),
        ("Tags Faithfulness", result.get('tags_faithfulness')),
        ("Content Coherence", result.get('content_coherence'))  # Add this line
    ]
    
    for name, score in scores:
        if score is not None:
            print(f"  {name}: {score:.3f}")