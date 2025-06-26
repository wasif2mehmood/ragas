import pandas as pd
import ast

# Load bulk_df from test_set.csv
bulk_df = pd.read_csv('./data/test_set.csv')

# Rename the columns
bulk_df.rename(columns={'user_input': 'question', 'reference': 'ground_truth'}, inplace=True)


# Define a function to process the input message and extract the required fields
def process_and_extract(input_message):
    results = process_input_message(input_message)
    answer = results[3].content
    contexts = results[2].content.split('\n\n')
    return answer, contexts

# Apply the function to the question column and create new columns for the results
bulk_df[['answer', 'contexts']] = bulk_df['question'].apply(lambda x: pd.Series(process_and_extract(x)))

# Save to CSV
bulk_df.to_csv('./data/ragas_dataset.csv', index=False)
print("Saved processed dataset to ragas_dataset.csv")

