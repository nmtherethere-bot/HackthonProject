import argparse
import json
from pathlib import Path

import yaml
from .config_utils import deep_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--init_checkpoint", type=str, required=True)
    ap.add_argument("--override", type=str, nargs="*", default=None, help="Optional YAML override files to merge")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.override:
        for o in args.override:
            o_cfg = yaml.safe_load(open(o))
            deep_update(cfg, o_cfg)
    cfg["data"]["train_path"] = args.data
    cfg.setdefault("init", {})["checkpoint"] = args.init_checkpoint

    # Prepare config and defer to Tunix GRPO trainer in the notebook/runner.
    out_dir = Path(cfg.get("output", {}).get("out_dir", "checkpoints/grpo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_cfg_path = out_dir / "derived_config.json"
    with open(derived_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Prepared GRPO config at:", derived_cfg_path)
    print("\nNext steps (on TPU VM / notebook):")
    print("  - Open notebooks/tunix_gemma_reasoning_pipeline.ipynb")
    print("  - Run the GRPO section, pointing to this derived_config.json and data path")
    if args.override:
        print("  - Applied overrides:", args.override)
    print("\nNote: GRPO is executed using Tunix's GRPO trainer in the notebook.")


if __name__ == "__main__":
    main()