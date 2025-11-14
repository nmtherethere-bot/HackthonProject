import argparse
import json
import os
from pathlib import Path
from typing import List

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .utils import load_jsonl, extract_trace_and_answer, normalized_answer

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None


def generate(model, tokenizer, prompts: List[str], max_new_tokens=512, device="cpu"):
    toks = tokenizer(prompts, return_tensors="pt", padding=True)
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=False)
    ap.add_argument("--data", type=str, required=False, help="Path to jsonl with question/answer/prompt fields")
    ap.add_argument("--wandb", action="store_true", help="Log metrics and a few generations to Weights & Biases")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ckpt = args.checkpoint or cfg.get("checkpoint")
    assert ckpt, "Provide --checkpoint or set in eval.yaml"

    # device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok_name = cfg.get("tokenizer_name", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    # Ensure a pad token exists for batch padding (e.g., GPT2 family)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # small eval on a subset of the training data
    data_path = Path(args.data) if args.data else Path("data/processed/gsm8k_train.jsonl")
    rows = list(load_jsonl(str(data_path)))[: cfg.get("max_samples", 200)]
    prompts = [r["prompt"] for r in rows]
    outputs = generate(model, tokenizer, prompts, max_new_tokens=256, device=device)

    exact, fmt_ok, avg_len = 0, 0, 0
    for r, out in zip(rows, outputs):
        trace, final = extract_trace_and_answer(out)
        avg_len += len(trace)
        if trace and final:
            fmt_ok += 1
        if normalized_answer(final) == normalized_answer(r["answer"]):
            exact += 1

    n = max(1, len(rows))
    metrics = {
        "exact_match": exact / n,
        "format_ok": fmt_ok / n,
        "avg_trace_length": avg_len / n,
    }
    print(json.dumps(metrics, indent=2))

    if args.wandb and wandb is not None:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "tunix_gemma2b_reasoning"),
            name=f"eval-{Path(ckpt).name}",
            config={"checkpoint": ckpt, "data": str(data_path)},
        )
        wandb.log(metrics)
        # log a few generations
        samples = []
        for i, (r, out) in enumerate(zip(rows[:5], outputs[:5])):
            trace, final = extract_trace_and_answer(out)
            samples.append({
                "question": r.get("question"),
                "pred_trace": trace,
                "pred_final": final,
                "gt_final": r.get("answer"),
            })
        wandb.log({"samples": wandb.Table(data=[list(s.values()) for s in samples],
                                           columns=list(samples[0].keys()) if samples else [])})
        run.finish()


if __name__ == "__main__":
    main()