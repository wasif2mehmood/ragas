import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from paths import TEST_SET_CSV_STR


# Load environment variables from .env file
load_dotenv()



# Load documents from URL
loader = WebBaseLoader("https://python.langchain.com/docs/integrations/document_loaders")
docs = loader.load()



generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
# Convert to pandas DataFrame for easier viewing
df = dataset.to_pandas()
print(df.head())

# Save to CSV
df.to_csv(TEST_SET_CSV_STR, index=False)
print("Test set saved to test_set.csv")