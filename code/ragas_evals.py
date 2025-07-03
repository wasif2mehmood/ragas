import pandas as pd
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity, Faithfulness
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import glob
import os

# Import custom metrics
from metrics.tags_jaccard import create_tags_jaccard_metric
from metrics.references_jaccard import create_references_jaccard_metric
from metrics.coherence import ContentCoherenceMetric

# Import utility functions
from metrics.utils import (
    truncate_context,
    load_dataset,
    load_publication_descriptions,
    PublicationSample,
    print_evaluation_scores,
    save_evaluation_results,
    print_evaluation_summary,
    initialize_result_dict,
    prepare_text_for_semantic_similarity
)

from paths import GOLDEN_DATASET_CSV_STR, GOLDEN_DATASET_JSON_STR

load_dotenv()


class CoherenceSample:
    """Custom sample class for coherence evaluation."""
    def __init__(self, context, title_generated, tldr_generated, references_generated, tags_generated):
        self.context = context
        self.title_generated = title_generated
        self.tldr_generated = tldr_generated
        self.references_generated = references_generated
        self.tags_generated = tags_generated


async def evaluate_single_dataset(csv_file_path):
    """Evaluate a single CSV dataset."""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {os.path.basename(csv_file_path)}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(csv_file_path)
    pub_descriptions = load_publication_descriptions()
    
    # Initialize embeddings and metrics
    evaluator_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0))
    
    semantic_scorer = SemanticSimilarity(
        embeddings=LangchainEmbeddingsWrapper(evaluator_embedding)
    )
    
    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    
    # Initialize custom metrics
    tags_jaccard = create_tags_jaccard_metric()
    references_jaccard = create_references_jaccard_metric()
    coherence_scorer = ContentCoherenceMetric(llm=evaluator_llm)
    
    # Results storage
    results = []
    
    print(f"Evaluating {len(df)} publications...")
    
    for index, row in df.iterrows():
        print(f"Processing publication {index + 1}/{len(df)}: {row['publication_external_id']}")
        
        try:
            # Get publication description for context and truncate if needed
            pub_id = row['publication_external_id']
            raw_context = pub_descriptions.get(pub_id, "")
            context = truncate_context(raw_context, max_tokens=8000)
            
            if len(raw_context) > len(context):
                print(f"  Warning: Context truncated from {len(raw_context)} to {len(context)} characters")
            
            # Initialize result dictionary using utility function
            result = initialize_result_dict(row['publication_external_id'])
            
            # 1. Title Evaluation
            if pd.notna(row['title_truth']) and pd.notna(row['title_generated']):
                # Semantic Similarity
                title_sample = SingleTurnSample(
                    user_input="dummy",
                    response=str(row['title_generated']),
                    reference=str(row['title_truth'])
                )
                result['title_semantic_similarity'] = await semantic_scorer.single_turn_ascore(title_sample)
                
                # Faithfulness
                title_faithfulness_sample = SingleTurnSample(
                    user_input="Generate a concise and accurate title for the given content.",
                    response=str(row['title_generated']),
                    retrieved_contexts=[context] if context else [""]
                )
                result['title_faithfulness'] = await faithfulness_scorer.single_turn_ascore(title_faithfulness_sample)
            
            # 2. TLDR Evaluation
            if pd.notna(row['tldr_truth']) and pd.notna(row['tldr_generated']):
                # Semantic Similarity
                tldr_sample = SingleTurnSample(
                    user_input="dummy",
                    response=str(row['tldr_generated']),
                    reference=str(row['tldr_truth'])
                )
                result['tldr_semantic_similarity'] = await semantic_scorer.single_turn_ascore(tldr_sample)
                
                # Faithfulness
                tldr_faithfulness_sample = SingleTurnSample(
                    user_input="Provide a concise summary (TL;DR) for the given content that captures the main points and key takeaways.",
                    response=str(row['tldr_generated']),
                    retrieved_contexts=[context] if context else [""]
                )
                result['tldr_faithfulness'] = await faithfulness_scorer.single_turn_ascore(tldr_faithfulness_sample)
            
            # 3. References Evaluation
            if pd.notna(row['references_truth']) and pd.notna(row['references_generated']):
                # Semantic Similarity
                refs_truth_text = str(row['references_truth'])
                refs_generated_text = str(row['references_generated'])
                refs_semantic_sample = SingleTurnSample(
                    user_input="dummy",
                    response=refs_generated_text,
                    reference=refs_truth_text
                )
                result['references_semantic_similarity'] = await semantic_scorer.single_turn_ascore(refs_semantic_sample)
                
                # Jaccard Similarity
                refs_sample = PublicationSample(
                    references_generated=str(row['references_generated']),
                    references_truth=str(row['references_truth'])
                )
                result['references_jaccard_similarity'] = await references_jaccard._single_turn_ascore(refs_sample, callbacks=None)
                
                # Faithfulness
                refs_faithfulness_sample = SingleTurnSample(
                    user_input="Extract and list the relevant references and citations mentioned in the given content.",
                    response=refs_generated_text,
                    retrieved_contexts=[context] if context else [""]
                )
                result['references_faithfulness'] = await faithfulness_scorer.single_turn_ascore(refs_faithfulness_sample)
            
            # 4. Tags Evaluation
            if pd.notna(row['tags_truth']) and pd.notna(row['tags_generated']):
                # Semantic Similarity
                tags_truth_text = prepare_text_for_semantic_similarity(row['tags_truth'], 'tags')
                tags_generated_text = prepare_text_for_semantic_similarity(row['tags_generated'], 'tags')
                tags_semantic_sample = SingleTurnSample(
                    user_input="dummy",
                    response=tags_generated_text,
                    reference=tags_truth_text
                )
                result['tags_semantic_similarity'] = await semantic_scorer.single_turn_ascore(tags_semantic_sample)
                
                # Jaccard Similarity
                tags_sample = PublicationSample(
                    tags_generated=str(row['tags_generated']),
                    tags_truth=str(row['tags_truth'])
                )
                result['tags_jaccard_similarity'] = await tags_jaccard._single_turn_ascore(tags_sample, callbacks=None)
                
                # Faithfulness
                tags_faithfulness_sample = SingleTurnSample(
                    user_input="Generate relevant tags and keywords that accurately represent the main topics and themes of the given content.",
                    response=tags_generated_text,
                    retrieved_contexts=[context] if context else [""]
                )
                result['tags_faithfulness'] = await faithfulness_scorer.single_turn_ascore(tags_faithfulness_sample)
            
            # 5. Content Coherence Evaluation
            if (context and 
                pd.notna(row['title_generated']) and 
                pd.notna(row['tldr_generated']) and 
                pd.notna(row['references_generated']) and 
                pd.notna(row['tags_generated'])):
                
                # Create custom sample for coherence evaluation
                coherence_sample = CoherenceSample(
                    context=context,
                    title_generated=str(row['title_generated']),
                    tldr_generated=str(row['tldr_generated']),
                    references_generated=str(row['references_generated']),
                    tags_generated=str(row['tags_generated'])
                )
                
                result['content_coherence'] = await coherence_scorer._single_turn_ascore(coherence_sample, callbacks=None)
            
            results.append(result)
            
            # Print scores using utility function
            print_evaluation_scores(result)
            
        except Exception as e:
            print(f"Error processing publication {row['publication_external_id']}: {e}")
            import traceback
            traceback.print_exc()
            # Still add the result with None values using utility function
            result = initialize_result_dict(row['publication_external_id'])
            results.append(result)
    
    # Save results with dataset name prefix
    dataset_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    results_df = pd.DataFrame(results)
    
    # Merge with original data
    complete_results = df.merge(results_df, on='publication_external_id', how='left')
    
    # Save files with dataset name prefix
    results_filename = f"./data/{dataset_name}_evaluation_results.csv"
    complete_filename = f"./data/{dataset_name}_complete_evaluation_results.csv"
    
    results_df.to_csv(results_filename, index=False)
    complete_results.to_csv(complete_filename, index=False)
    
    print(f"\nResults saved to: {results_filename}")
    print(f"Complete results saved to: {complete_filename}")
    
    # Print summary
    print_evaluation_summary(results_df)
    
    return results_df, complete_results


async def evaluate_golden_dataset():
    """Evaluate the main golden dataset."""
    
    print("🚀 Starting evaluation of the Golden Dataset")
    print(f"📁 Dataset path: {GOLDEN_DATASET_CSV_STR}")
    
    try:
        results_df, complete_results = await evaluate_single_dataset(GOLDEN_DATASET_CSV_STR)
        print("\n✅ Golden dataset evaluation completed!")
        return results_df, complete_results
    except Exception as e:
        print(f"❌ Error evaluating golden dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None





if __name__ == "__main__":
    # Evaluate datasets
    asyncio.run(evaluate_golden_dataset())