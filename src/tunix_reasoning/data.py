import argparse
import random
import re
from typing import List, Dict
from datasets import load_dataset
from .utils import save_jsonl


def normalize_boxed_answer(s: str) -> str:
    # GSM8K answers often look like "#### 42" in solution; prefer numeric
    s = s.strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return m.group(1) if m else s


def build_prompt_and_target(question: str, rationale: str, answer: str):
    prompt = (
        "Solve the problem. Think step by step inside <reasoning>...</reasoning>.\n"
        "Then output only the final numeric answer inside <final>...</final>.\n\n"
        f"Question: {question}\n<reasoning>\n"
    )
    target = f"{rationale}\n</reasoning>\n<final>" + normalize_boxed_answer(answer) + "</final>\n"
    return prompt, target


def process_gsm8k(split: str) -> List[Dict]:
    ds = load_dataset("gsm8k", "main", split=split)
    rows = []
    for ex in ds:
        q = ex["question"].strip()
        a = ex["answer"].strip()
        # extract final numeric answer after '####'
        final_match = re.search(r"####\s*(.*)$", a)
        final = final_match.group(1).strip() if final_match else a
        rationale = a.split("####")[0].strip()
        prompt, target = build_prompt_and_target(q, rationale, final)
        rows.append({
            "question": q,
            "answer": final,
            "prompt": prompt,
            "target": target,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k"], help="Reasoning dataset")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--out_val", type=str, default=None, help="Optional path to write a held-out validation set")
    ap.add_argument("--val_frac", type=float, default=0.05, help="Fraction for validation split if out_val is provided")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "gsm8k":
        rows = process_gsm8k(args.split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.out_val:
        rng = random.Random(args.seed)
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_frac))
        val_rows = shuffled[:n_val]
        train_rows = shuffled[n_val:]
        save_jsonl(args.out, train_rows)
        save_jsonl(args.out_val, val_rows)
        print(f"Wrote {len(train_rows)} train rows to {args.out}")
        print(f"Wrote {len(val_rows)} val rows to {args.out_val}")
    else:
        save_jsonl(args.out, rows)
        print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()