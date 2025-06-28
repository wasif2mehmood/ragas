from dataclasses import dataclass, field
import typing as t
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field


class ConcisenessInput(BaseModel):
    user_input: str = Field(description="The user's question or request")
    response: str = Field(description="The AI response to evaluate")
    reference: str = Field(description="The reference/ground truth answer")


class ConcisenessOutput(BaseModel):
    score: float = Field(
        description="Conciseness score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the score")


class ConcisenessPrompt(PydanticPrompt[ConcisenessInput, ConcisenessOutput]):
    instruction = """You are an expert evaluator tasked with measuring how concise and efficient a response is.

Evaluate the conciseness of the given response compared to the reference answer.

Consider the following criteria:
1. Information Density: Does the response convey essential information without unnecessary words?
2. Redundancy: Is there any repetitive or redundant information?
3. Directness: Does the response get straight to the point?
4. Completeness vs Brevity: Does it maintain completeness while being concise?
5. Clarity: Is the response clear despite being concise?

Rate on a scale of 0 to 1:
- 1.0: Perfect conciseness - all necessary information with minimal words
- 0.8-0.9: Very concise - minor unnecessary words but overall efficient
- 0.6-0.7: Moderately concise - some redundancy or wordiness
- 0.4-0.5: Poor conciseness - significant redundancy or unnecessary elaboration
- 0.0-0.3: Very poor conciseness - extremely verbose or repetitive"""
    
    input_model = ConcisenessInput
    output_model = ConcisenessOutput
   


@dataclass
class ResponseConcisenessMetric(MetricWithLLM, SingleTurnMetric):
    """
    Custom metric to evaluate how concise and efficient a response is.
    
    This metric measures whether the AI response conveys information efficiently
    without unnecessary verbosity while maintaining completeness and clarity.
    """
    
    name: str = "response_conciseness"
    
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )
    
    conciseness_prompt: PydanticPrompt = field(default_factory=ConcisenessPrompt)
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """
        Evaluate the conciseness of a single response.
        
        Args:
            sample: The sample containing user_input, response, and reference
            callbacks: Callbacks for monitoring
            
        Returns:
            Float score between 0 and 1 indicating conciseness level
        """
        
        # Prepare the prompt input
        prompt_input = ConcisenessInput(
            user_input=sample.user_input,
            response=sample.response,
            reference=sample.reference if sample.reference else "Not provided"
        )
        
        try:
            # Generate response using the PydanticPrompt
            prompt_response = await self.conciseness_prompt.generate(
                data=prompt_input, 
                llm=self.llm,
                callbacks=callbacks
            )
            
            return prompt_response.score
            
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return 0.5  # Default score if LLM fails