import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


PROMPT_TMPL = (
    "Solve the problem. Think step by step inside <reasoning>...</reasoning>.\n"
    "Then output only the final numeric answer inside <final>...</final>.\n\n"
    "Question: {question}\n<reasoning>\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to("cuda")

    prompt = PROMPT_TMPL.format(question=args.question)
    toks = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **toks,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()