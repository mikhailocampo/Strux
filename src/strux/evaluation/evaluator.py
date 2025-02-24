import json
import openai
from typing import Optional, Dict, Any
from datetime import datetime

from strux.evaluation.prompts import EVALUATOR_PAIRWISE_PREFERENCE_PROMPT
from strux.evaluation.utils import probability_from_two_logprobs

class LLMEvaluator:
    """
    A lightweight class wrapper around OpenAI API client (or similar).
    Provides an interface to query preference probabilities (A vs B)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.client = openai.OpenAI(
            api_key=api_key,
            **kwargs,
        )
        self.model = model
        self.temperature = temperature

    def preference_probability(
        self, 
        text_a: str, 
        text_b: str, 
        quality: str, 
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return a probability p indicating that text_a is preferred over text_b.
        Uses a prompt or logit-based approach to glean p from the LLM.
        
        We send top logprobs=1 to get the most likely token given the prompt. 
        This is a proxy for the probability of the token.
        
        For example, you could:
          - Send a few-shot or zero-shot prompt with instructions
          - Parse the token probabilities or the raw text output

        Placeholder code below for demonstration.
        """
        try:
            if prompt is None:
                prompt = EVALUATOR_PAIRWISE_PREFERENCE_PROMPT.format(
                    text_a=text_a, 
                    text_b=text_b, 
                    quality=quality
                )
            
            system_msg = {
                "role": "system",
                "content": prompt,
            }
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[system_msg],
                logprobs=True,
                top_logprobs=2,
            )

            choice = response.choices[0]
            text_output = json.loads(choice.message.content)
            logprobs_info = choice.logprobs
            
            # Extract the top two token log probabilities
            tokens_logprobs = logprobs_info.content[0].top_logprobs
            if len(tokens_logprobs) < 2:
                raise ValueError("Insufficient top logprobs available.")
            
            p = probability_from_two_logprobs(
                tokens_logprobs[0].logprob,
                tokens_logprobs[1].logprob,
            )
            
            return {
                "preference": text_output["preference"],
                "probability": p,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "preference": None,
                "probability": 0.5,  # Neutral probability on failure
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }