from datasets import Dataset
import pandas as pd
import ast
from dotenv import load_dotenv
from ragas.metrics import faithfulness,answer_relevancy,context_precision,context_recall,context_entity_recall,answer_similarity,answer_correctness
from ragas import evaluate

# Load environment variables from .env file
load_dotenv()

# Load the dataset from CSV
df = pd.read_csv('./data/ragas_dataset.csv')

# Convert string representations of lists back to actual lists
def parse_list_column(column):
    def parse_item(x):
        if isinstance(x, str):
            try:
                # Try to parse as a Python literal
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                # If that fails, return as is
                return x
        return x
    return column.apply(parse_item)
# Parse the context columns that should be lists
if 'contexts' in df.columns:
    df['contexts'] = parse_list_column(df['contexts'])
if 'reference_contexts' in df.columns:
    df['reference_contexts'] = parse_list_column(df['reference_contexts'])

# Debug: Check the data types after conversion
print("\nAfter conversion:")
print(f"contexts type: {type(df['contexts'].iloc[0])}")
print(f"contexts sample: {df['contexts'].iloc[0]}")

syn_dataset = Dataset.from_pandas(df)

bulk_score = evaluate(syn_dataset,metrics=[faithfulness,answer_relevancy,context_precision,context_recall,context_entity_recall,answer_similarity,answer_correctness])

# save the scores to a CSV file
scores_df = bulk_score.to_pandas()
scores_df.to_csv('ragas_scores.csv', index=False)
print("Scores saved to ragas_scores.csv")