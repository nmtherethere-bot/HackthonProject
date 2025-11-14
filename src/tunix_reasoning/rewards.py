from dataclasses import dataclass
from typing import Dict

from .utils import extract_trace_and_answer, normalized_answer


@dataclass
class RewardConfig:
    w_correct: float = 1.0
    w_format: float = 0.2
    w_length: float = -0.05
    w_step: float = 0.1
    w_stop: float = 0.1
    target_trace_min: int = 80
    target_trace_max: int = 600


def _format_score(text: str) -> float:
    has_reasoning = "<reasoning>" in text and "</reasoning>" in text
    has_final = "<final>" in text and "</final>" in text
    if has_reasoning and has_final:
        # reward well-ordered tags slightly higher
        order_ok = text.find("<reasoning>") < text.find("<final>")
        return 1.0 if order_ok else 0.8
    if has_reasoning or has_final:
        return 0.2
    return 0.0


def _length_score(trace: str, cfg: RewardConfig) -> float:
    n = len(trace)
    if n == 0:
        return -1.0
    if n < cfg.target_trace_min:
        return (n - cfg.target_trace_min) / float(cfg.target_trace_min)
    if n > cfg.target_trace_max:
        return (cfg.target_trace_max - n) / float(cfg.target_trace_max)
    return 1.0


def _stop_bonus(text: str) -> float:
    # Encourage proper closure
    ends_ok = text.strip().endswith("</final>")
    return 1.0 if ends_ok else 0.0


def compute_reward(sample: Dict, generated_text: str, cfg: RewardConfig) -> float:
    """Composite reward for GRPO/PPO style training.

    sample must contain the ground truth answer under key 'answer' (normalized numeric or string).
    generated_text should include <reasoning>...</reasoning> and <final>...</final> sections.
    """
    trace, final = extract_trace_and_answer(generated_text)
    # correctness (exact match after normalization)
    gt = normalized_answer(sample.get("answer", ""))
    pred = normalized_answer(final)
    correct = 1.0 if gt and pred and gt == pred else 0.0

    # formatting and length
    fmt = _format_score(generated_text)
    length = _length_score(trace, cfg)

    # simple step-consistency proxy: reasoning exists and final exists
    step_consistency = 1.0 if trace and final else 0.0

    # stop bonus
    stop = _stop_bonus(generated_text)

    reward = (
        cfg.w_correct * correct
        + cfg.w_format * fmt
        + cfg.w_length * length
        + cfg.w_step * step_consistency
        + cfg.w_stop * stop
    )
    return float(reward)