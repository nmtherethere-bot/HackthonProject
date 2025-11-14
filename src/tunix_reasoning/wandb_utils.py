import os
from typing import Optional, Dict


def try_init_wandb(project: Optional[str] = None, name: Optional[str] = None, config: Optional[Dict] = None):
    try:
        import wandb
    except Exception:
        return None
    project = project or os.environ.get("WANDB_PROJECT", "tunix_gemma2b_reasoning")
    run = wandb.init(project=project, name=name, config=config)
    return run


def log_and_finish(run, metrics: Dict, samples: Optional[list] = None):
    if run is None:
        return
    try:
        import wandb
    except Exception:
        return
    run.log(metrics)
    if samples:
        cols = list(samples[0].keys())
        table = wandb.Table(data=[[s[c] for c in cols] for s in samples], columns=cols)
        run.log({"samples": table})
    run.finish()