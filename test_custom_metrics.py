"""
Test script for custom metrics
"""

import asyncio
from custom_metrics import ResponseConcisenessMetric
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample

async def test_custom_metrics():
    """Test the custom metrics with sample data."""
    
    # Initialize LLM
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    
    # Initialize metrics
    conciseness_metric = ResponseConcisenessMetric(llm=evaluator_llm)
    
    # Create a sample for testing
    sample = SingleTurnSample(
        user_input="How can a Data Integration Specialist utilize Telegram for data integration using document loaders?",
        response="To utilize Telegram for data integration, use the TelegramChatFileLoader from the messaging services document loaders.",
        reference="A Data Integration Specialist can utilize Telegram for data integration by using the TelegramChatFileLoader, which is a document loader designed to load data from Telegram messaging platforms.",
        retrieved_contexts=[
            "Social Platforms: Document loaders for social media platforms include TwitterTweetLoader, RedditPostsLoader. Messaging Services: TelegramChatFileLoader, WhatsAppChatLoader, DiscordChatLoader, FacebookChatLoader, MastodonTootsLoader."
        ]
    )
    
    print("Testing Custom Metrics...")
    print("=" * 50)
    
    # Test Response Conciseness Metric
    print("1. Testing Response Conciseness Metric:")
    print(f"Question: {sample.user_input}")
    print(f"Response: {sample.response}")
    print(f"Ground Truth: {sample.reference}")
    
    try:
        conciseness_score = await conciseness_metric._single_turn_ascore(sample, callbacks=None)
        print(f"Conciseness Score: {conciseness_score.score}")
        # print(f"Reasoning: {conciseness_score.reasoning}")
    except Exception as e:
        print(f"Error testing conciseness metric: {e}")
    
    print("\n" + "=" * 50)
    

    print("Custom Metrics Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_custom_metrics())
