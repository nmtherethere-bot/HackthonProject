import argparse
import json
from pathlib import Path

import yaml
from .config_utils import deep_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--override", type=str, nargs="*", default=None, help="Optional YAML override files to merge")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    # Apply overrides if provided
    if args.override:
        for o in args.override:
            o_cfg = yaml.safe_load(open(o))
            deep_update(cfg, o_cfg)
    cfg["data"]["train_path"] = args.data

    # This script prepares config and defers to Tunix SFT trainer via notebook/runner.
    out_dir = Path(cfg.get("output", {}).get("out_dir", "checkpoints/sft"))
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_cfg_path = out_dir / "derived_config.json"
    with open(derived_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Prepared SFT config at:", derived_cfg_path)
    print("\nNext steps (on TPU VM / notebook):")
    print("  - Open notebooks/tunix_gemma_reasoning_pipeline.ipynb")
    print("  - Run the SFT section, pointing to this derived_config.json and data path")
    if args.override:
        print("  - Applied overrides:", args.override)
    print("\nNote: SFT is executed using Tunix's SFT trainer in the notebook.")


if __name__ == "__main__":
    main()