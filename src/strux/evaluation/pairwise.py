import numpy as np
from typing import List, Any, Optional
from strux.evaluation.utils import compute_entropy
from strux.evaluation.evaluator import LLMEvaluator
from dataclasses import dataclass

@dataclass
class MergeStep:
    """Records a single comparison during merge sort."""
    left_idx: int
    right_idx: int
    probability: float
    uncertainty: float
    chosen_idx: int
    is_uncertain: bool

class BeamCandidate:
    def __init__(self, trajectory=None, i=0, j=0, log_prob=0.0):
        self.trajectory = trajectory if trajectory else []
        self.i = i
        self.j = j
        self.log_prob = log_prob

    def copy(self):
        return BeamCandidate(self.trajectory[:], self.i, self.j, self.log_prob)

    def extend(self, item: Any, p: float):
        """Extend trajectory with item, update log probability."""
        new_cand = self.copy()
        new_cand.trajectory.append(item)
        if p > 0:
            new_cand.log_prob += np.log(p + 1e-12)
        return new_cand


def pairwise_merge(
    a: List[str],
    b: List[str],
    evaluator: 'LLMEvaluator',
    beam_size: int = 5,
    uncertainty_threshold: float = 0.3,
    quality: str = "coherence",
    merge_history: Optional[List[MergeStep]] = None
) -> List[str]:
    """
    Uncertainty-guided beam merge operation.
    
    Args:
        a: First sorted subarray
        b: Second sorted subarray
        evaluator: Instance of Evaluator class to compute preferences
        beam_size: Maximum number of candidates to maintain
        uncertainty_threshold: Threshold for branching on uncertain comparisons
        quality: Quality to evaluate e.g, "coherence", "fluency", etc.
        merge_history: Optional list to track merge decisions
    """
    beam = [BeamCandidate([], 0, 0, 0.0)]
    total = len(a) + len(b)

    for _ in range(total):
        new_beam = []

        for cand in beam:
            i, j = cand.i, cand.j
            
            # If we exhausted one subarray, take from the other
            if i >= len(a) and j < len(b):
                cand_b = cand.extend(b[j], p=1.0)
                cand_b.j += 1
                new_beam.append(cand_b)
                continue

            if j >= len(b) and i < len(a):
                cand_a = cand.extend(a[i], p=1.0)
                cand_a.i += 1
                new_beam.append(cand_a)
                continue

            # Compare a[i] vs b[j] using the evaluator
            if i < len(a) and j < len(b):
                result = evaluator.preference_probability(a[i], b[j], quality=quality)
                p = result['probability']
                u = compute_entropy(p)

                # Record merge step if tracking is enabled
                if merge_history is not None:
                    merge_history.append(MergeStep(
                        left_idx=i,
                        right_idx=j,
                        probability=p,
                        uncertainty=u,
                        chosen_idx=i if p >= 0.5 else j,
                        is_uncertain=(u > uncertainty_threshold)
                    ))

                if u > uncertainty_threshold:
                    # Branch both ways when uncertain
                    cand_a = cand.extend(a[i], p=p)
                    cand_a.i += 1
                    new_beam.append(cand_a)

                    cand_b = cand.extend(b[j], p=(1-p))
                    cand_b.j += 1
                    new_beam.append(cand_b)
                else:
                    # Single branch when confident
                    if p >= 0.5:
                        cand_a = cand.extend(a[i], p=p)
                        cand_a.i += 1
                        new_beam.append(cand_a)
                    else:
                        cand_b = cand.extend(b[j], p=(1-p))
                        cand_b.j += 1
                        new_beam.append(cand_b)

        # Sort new_beam by log probability descending
        new_beam.sort(key=lambda x: x.log_prob, reverse=True)
        beam = new_beam[:beam_size]

    # Return trajectory of best candidate
    return beam[0].trajectory