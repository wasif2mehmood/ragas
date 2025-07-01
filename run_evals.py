import pandas as pd
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity, Faithfulness
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import custom metrics
from src.tags_jaccard import create_tags_jaccard_metric
from src.references_jaccard import create_references_jaccard_metric

# Import utility functions
from src.utils import (
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

load_dotenv()


async def evaluate_dataset():
    """Evaluate the golden dataset with semantic similarity, Jaccard metrics, and faithfulness."""
    
    # Load data using utility functions
    df = load_dataset()
    pub_descriptions = load_publication_descriptions()
    
    # Initialize embeddings and metrics
    evaluator_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0))
    
    semantic_scorer = SemanticSimilarity(
        embeddings=LangchainEmbeddingsWrapper(evaluator_embedding)
    )
    
    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    
    # Initialize Jaccard metrics
    tags_jaccard = create_tags_jaccard_metric()
    references_jaccard = create_references_jaccard_metric()
    
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
    
    # Save results and print summary using utility functions
    results_df, complete_results = save_evaluation_results(results, df)
    print_evaluation_summary(results_df)
    
    return results_df, complete_results


if __name__ == "__main__":
    # Evaluate entire dataset
    asyncio.run(evaluate_dataset())