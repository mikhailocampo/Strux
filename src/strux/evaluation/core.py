import pandas as pd
from typing import List, Optional, Tuple
import os
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from strux.evaluation.pairwise import pairwise_merge, MergeStep
from strux.evaluation.evaluator import LLMEvaluator


class RankingResult:
    """Holds the results of a ranking operation with analysis utilities."""
    def __init__(
        self,
        responses: List[str],
        rankings: List[int],
        models: Optional[List[str]] = None,
        merge_history: Optional[List[MergeStep]] = None,
        evaluator: Optional[LLMEvaluator] = None,
        quality: str = "coherence"
    ):
        self.responses = responses
        self.rankings = rankings
        self.models = models or [None] * len(responses)
        self.merge_history = merge_history or []
        self.evaluator = evaluator
        self.quality = quality
        
    def get_ranked_responses(self) -> List[Tuple[int, str, Optional[str]]]:
        """Returns list of (rank, response, model) sorted by rank."""
        ranked_items = list(zip(self.rankings, self.responses, self.models))
        return sorted(ranked_items, key=lambda x: x[0])
    
    def get_preference_matrix(self) -> np.ndarray:
        """
        Create a preference matrix from merge history.
        Only includes pairs that were actually compared during merge sort.
        """
        n = len(self.responses)
        # Initialize with NaN to distinguish from actual 0.0 probabilities
        prob_matrix = np.full((n, n), np.nan)
        
        # Fill diagonal with 0.5 (self-comparison)
        np.fill_diagonal(prob_matrix, 0.5)
        
        # Fill matrix with only the comparisons that were made
        for step in self.merge_history:
            i, j = step.left_idx, step.right_idx
            prob_matrix[i, j] = step.probability
            # Fill the symmetric entry
            prob_matrix[j, i] = 1 - step.probability
            
        return prob_matrix
    
    def plot_preference_heatmap(self, figsize=(10, 8)):
        """
        Visualize the pairwise preferences that were actually computed during merge sort.
        Shows probability of row item being preferred over column item.
        """
        prob_matrix = self.get_preference_matrix()
        
        plt.figure(figsize=figsize)
        
        # Create mask for NaN values
        mask = np.isnan(prob_matrix)
        
        # Plot heatmap
        sns.heatmap(
            prob_matrix,
            annot=True,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            square=True,
            mask=mask,  # Hide NaN values
            fmt='.2f'   # Format numbers to 2 decimal places
        )
        
        # Add model labels if available
        if any(self.models):
            plt.xticks(range(len(self.models)), self.models, rotation=45, ha='right')
            plt.yticks(range(len(self.models)), self.models, rotation=0)
        
        plt.title("Pairwise Preference Probabilities\n(Only showing computed comparisons)")
        plt.xlabel("Model B")
        plt.ylabel("Model A")
        
        # Overlay merge paths
        for step in self.merge_history:
            color = 'white' if step.is_uncertain else 'black'
            style = '--' if step.is_uncertain else '-'
            plt.plot(
                [step.right_idx + 0.5, step.right_idx + 0.5],
                [step.left_idx + 0.5, step.left_idx + 0.5],
                color=color,
                linestyle=style,
                linewidth=2,
                alpha=0.7
            )
        
        plt.tight_layout()
        plt.show()

class PairwisePreferenceRanker:
    def __init__(
        self, 
        evaluator: LLMEvaluator,
        beam_size: int = 5,
        uncertainty_threshold: float = 0.3,
        quality: str = "coherence",
        checkpoint: bool = False,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.evaluator = evaluator
        self.beam_size = beam_size
        self.uncertainty_threshold = uncertainty_threshold
        self.quality = quality
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.merge_history = []

    def evaluate(
        self,
        df: pd.DataFrame,
        prompt_col: str = "prompt",
        responses_col: str = "response",
        model_col: Optional[str] = "model",
        prompt_id_col: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Given a DataFrame where each row has:
          - A 'prompt' (or 'prompt_id')
          - A 'response' column containing a list of response strings
          - Optional 'model' column containing a list of model names
        
        Run the merge-sort approach to rank all responses for each prompt.

        Returns a new DataFrame with original data plus rankings.
        """
        group_key = prompt_id_col if prompt_id_col else prompt_col

        if self.checkpoint and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, "ranker_checkpoint.parquet")

        if self.checkpoint and os.path.exists(checkpoint_path):
            return pd.read_parquet(checkpoint_path)

        results_list = []

        # Main progress bar for prompts
        with tqdm(total=len(df), desc="Ranking prompts", position=0, leave=True) as pbar:
            for idx, row in df.iterrows():
                prompt_val = row[group_key]
                responses = row[responses_col]
                models = row.get(model_col, [None] * len(responses)) if model_col else [None] * len(responses)

                if len(responses) <= 1:
                    results_list.append({
                        'prompt': prompt_val,
                        'responses': responses,
                        'models': models,
                        'rankings': [0] * len(responses)
                    })
                    pbar.update(1)
                    continue

                try:
                    # Update description to show current prompt
                    pbar.set_description(f"Ranking prompt: {prompt_val[:30]}...")
                    
                    # Clear merge history for new ranking
                    self.merge_history = []
                    
                    # Run merge sort with nested progress tracking
                    ranked_responses = self._merge_sort(
                        responses,
                        desc=f"Ranking {len(responses)} responses"
                    )
                    
                    rankings = [ranked_responses.index(r) for r in responses]
                    
                    # Create RankingResult object with evaluator
                    result = RankingResult(
                        responses=responses,
                        rankings=rankings,
                        models=models,
                        merge_history=self.merge_history,
                        evaluator=self.evaluator,  # Pass evaluator
                        quality=self.quality       # Pass quality metric
                    )
                    
                    results_list.append({
                        'prompt': prompt_val,
                        'responses': responses,
                        'models': models,
                        'rankings': rankings,
                        'ranking_result': result
                    })

                except Exception as e:
                    print(f"Error ranking prompt {prompt_val}: {e}")
                    results_list.append({
                        'prompt': prompt_val,
                        'responses': responses,
                        'models': models,
                        'rankings': list(range(len(responses)))
                    })
                
                pbar.update(1)

        ranked_df = pd.DataFrame(results_list)

        if self.checkpoint:
            ranked_df.to_parquet(checkpoint_path, index=False)
        if output_path:
            ranked_df.to_parquet(output_path, index=False)

        return ranked_df

    def _merge_sort(self, items: List[str], desc: str = "Ranking responses") -> List[str]:
        """
        Recursively split the list into halves and merge them using pairwise comparisons.
        Shows progress of comparisons being made.
        """
        # Calculate total number of comparisons needed
        n = len(items)
        if n <= 1:
            return items
        
        # Create progress bar for this merge sort operation
        total_comparisons = n * int(np.log2(n))  # Approximate number of comparisons
        with tqdm(total=total_comparisons, desc=desc, leave=False, position=1) as pbar:
            def _merge_sort_with_progress(items: List[str]) -> List[str]:
                if len(items) <= 1:
                    return items
                
                mid = len(items) // 2
                left_sorted = _merge_sort_with_progress(items[:mid])
                right_sorted = _merge_sort_with_progress(items[mid:])
                
                try:
                    result = pairwise_merge(
                        left_sorted,
                        right_sorted,
                        evaluator=self.evaluator,
                        beam_size=self.beam_size,
                        uncertainty_threshold=self.uncertainty_threshold,
                        quality=self.quality,
                        merge_history=self.merge_history
                    )
                    # Update progress based on number of comparisons in this merge
                    pbar.update(len(left_sorted) + len(right_sorted))
                    return result
                except Exception as e:
                    print(f"Error during merge: {e}")
                    return left_sorted + right_sorted
            
            return _merge_sort_with_progress(items)