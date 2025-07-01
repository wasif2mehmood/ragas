from dataclasses import dataclass, field
import typing as t
import json
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from ragas.metrics.base import SingleTurnMetric, MetricType
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample


@dataclass
class ReferencesJaccardMetric(SingleTurnMetric):
    """
    Custom metric to calculate Jaccard similarity for references using sklearn.
    
    Handles references as list of dictionaries with 'url' and 'title' keys.
    Calculates Jaccard scores separately for URLs and titles, then averages them.
    """
    
    name: str = "references_jaccard_similarity"
    
    generated_column: str = field(default="references_generated")
    truth_column: str = field(default="references_truth")
    
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default=None)
    
    case_sensitive: bool = field(default=False)
    strip_whitespace: bool = field(default=True)
    
    def __post_init__(self):
        """Set required columns based on configured column names."""
        if self._required_columns is None:
            self._required_columns = {
                MetricType.SINGLE_TURN: {self.generated_column, self.truth_column}
            }
    
    def init(self, run_config):
        """Initialize the metric. Required by SingleTurnMetric."""
        pass

    def _parse_references(self, references_data) -> t.Tuple[t.List[str], t.List[str]]:
        """
        Parse references data into lists of URLs and titles.
        
        Args:
            references_data: Either a string representation of list or actual list
            
        Returns:
            Tuple of (urls_list, titles_list)
        """
        urls = []
        titles = []
        
        try:
            # Handle string representation of list
            if isinstance(references_data, str):
                if references_data.strip():
                    # Use json.loads instead of eval for safety
                    try:
                        refs_list = json.loads(references_data)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try eval as fallback (with caution)
                        refs_list = eval(references_data)
                else:
                    refs_list = []
            elif isinstance(references_data, list):
                refs_list = references_data
            else:
                refs_list = []
            
            for ref in refs_list:
                if isinstance(ref, dict):
                    # Extract URL
                    url = ref.get('url', '').strip()
                    if url:
                        if not self.case_sensitive:
                            url = url.lower()
                        urls.append(url)
                    
                    # Extract title
                    title = ref.get('title', '').strip()
                    if title:
                        if not self.case_sensitive:
                            title = title.lower()
                        titles.append(title)
                        
        except Exception as e:
            print(f"Error parsing references: {e}")
            print(f"References data: {references_data}")
            
        return urls, titles

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
                return 0.0  # Perfect match for empty sets
            
            if not y_true or not y_pred:
                return 0.0  # No similarity if one set is empty
            
            # Create all unique labels from both sets
            all_labels = list(set(y_true + y_pred))
            
            if not all_labels:
                return 0.0  # Both empty after preprocessing
            
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
        Calculate Jaccard similarity for references.
        
        Args:
            sample: The sample containing the reference columns
            callbacks: Callbacks for monitoring
            
        Returns:
            Float score between 0 and 1 (average of URL and title Jaccard scores)
        """
        try:
            # Extract references data
            generated_refs = getattr(sample, self.generated_column, '') or ""
            truth_refs = getattr(sample, self.truth_column, '') or ""
            
            # Parse references into URLs and titles
            gen_urls, gen_titles = self._parse_references(generated_refs)
            truth_urls, truth_titles = self._parse_references(truth_refs)
            
            # Calculate Jaccard scores separately using sklearn
            url_jaccard = self._calculate_sklearn_jaccard(truth_urls, gen_urls)
            title_jaccard = self._calculate_sklearn_jaccard(truth_titles, gen_titles)
            
            # Return average of both scores
            return (url_jaccard + title_jaccard) / 2.0
            
        except Exception as e:
            print(f"Error calculating references Jaccard similarity: {e}")
            return 0.0


def create_references_jaccard_metric() -> ReferencesJaccardMetric:
    """Create a Jaccard similarity metric for comparing references."""
    return ReferencesJaccardMetric(
        name="references_jaccard_similarity",
        generated_column="references_generated",
        truth_column="references_truth",
        case_sensitive=False
    )