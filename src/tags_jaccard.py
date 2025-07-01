from dataclasses import dataclass, field
import typing as t
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from ragas.metrics.base import SingleTurnMetric, MetricType
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample


@dataclass
class TagsJaccardSimilarityMetric(SingleTurnMetric):
    """
    Custom metric to calculate Jaccard similarity between two sets of text using sklearn.
    
    Jaccard similarity = |intersection| / |union|
    Used for comparing any two columns containing delimited text data.
    """
    
    name: str = "tags_jaccard_similarity"
    
    # Configurable column names
    generated_column: str = field(default="response")
    truth_column: str = field(default="reference")
    
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default=None)
    
    delimiter: str = field(default=",")  # Default delimiter for splitting text
    case_sensitive: bool = field(default=False)  # Case sensitivity option
    strip_whitespace: bool = field(default=True)  # Strip whitespace from items
    
    def __post_init__(self):
        """Set required columns based on configured column names."""
        if self._required_columns is None:
            self._required_columns = {
                MetricType.SINGLE_TURN: {self.generated_column, self.truth_column}
            }
    
    def init(self, run_config):
        """Initialize the metric. Required by SingleTurnMetric."""
        pass
    
    def _preprocess_text(self, text: str) -> t.List[str]:
        """
        Preprocess text into a list of items.
        
        Args:
            text: Input text to process
            
        Returns:
            List of processed items
        """
        if not text or text.strip() == "":
            return []
        
        # Split by delimiter
        items = text.split(self.delimiter)
        
        # Process each item
        processed_items = []
        for item in items:
            if self.strip_whitespace:
                item = item.strip()
            if not self.case_sensitive:
                item = item.lower()
            if item:  # Only add non-empty items
                processed_items.append(item)
        
        return processed_items
    
    def _calculate_sklearn_jaccard(self, y_true: t.List[str], y_pred: t.List[str]) -> float:
        """
        Calculate Jaccard similarity using sklearn's jaccard_score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        try:
            # Handle empty lists
            if not y_true and not y_pred:
                return 1.0  # Perfect match for empty sets
            
            if not y_true or not y_pred:
                return 0.0  # No similarity if one set is empty
            
            # Create all unique labels from both sets
            all_labels = list(set(y_true + y_pred))
            
            if not all_labels:
                return 1.0  # Both empty after preprocessing
            
            # Use MultiLabelBinarizer to convert to binary format
            mlb = MultiLabelBinarizer(classes=all_labels)
            
            # Convert to binary matrices
            y_true_binary = mlb.fit_transform([y_true])
            y_pred_binary = mlb.transform([y_pred])
            
            # Calculate Jaccard score using sklearn
            score = jaccard_score(y_true_binary, y_pred_binary, average='samples', zero_division=0.0)
            
            return float(score)
            
        except Exception as e:
            print(f"Error in sklearn Jaccard calculation: {e}")
            return 0.0
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """
        Calculate Jaccard similarity for a single sample.
        
        Args:
            sample: The sample containing the specified columns
            callbacks: Callbacks for monitoring
            
        Returns:
            Float score between 0 and 1 indicating Jaccard similarity
        """
        try:
            # Extract text from specified columns
            generated_text = getattr(sample, self.generated_column, '') or ""
            truth_text = getattr(sample, self.truth_column, '') or ""
            
            # Convert to lists
            generated_list = self._preprocess_text(generated_text)
            truth_list = self._preprocess_text(truth_text)
            
            # Calculate Jaccard similarity using sklearn
            similarity = self._calculate_sklearn_jaccard(truth_list, generated_list)
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating Jaccard similarity: {e}")
            return 0.0  # Return 0 if calculation fails


def create_tags_jaccard_metric() -> TagsJaccardSimilarityMetric:
    """Create a Jaccard similarity metric for comparing tags."""
    return TagsJaccardSimilarityMetric(
        name="tags_jaccard_similarity",
        generated_column="tags_generated",
        truth_column="tags_truth",
        delimiter="|",
        case_sensitive=False
    )