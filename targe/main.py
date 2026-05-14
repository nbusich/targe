"""
Entry point for training/eval.

Everything is hydra-controlled. Examples:
    python -m targe.main                                  # full training run
    python -m targe.main +experiment=dry                  # full dry-run preset
    python -m targe.main training=sft_dry dry_run=true    # à la carte dry overrides
    python -m targe.main hub.repo_id=usich-n/foo train_output_dir=runs/foo
    python -m targe.main hub.enabled=false                # skip Hub upload
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from targe.data.chartqa import load_chartqa
from targe.model.build import build_model
from targe.train.train_mm import train as train_stage


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model, processor = build_model(cfg.model)
    train_ds, eval_ds, test_ds = load_chartqa(cfg.data)

    if cfg.get("train", False):
        train_stage(model, processor, train_ds, eval_ds, test_ds, cfg)

    if cfg.get("eval", False):
        raise NotImplementedError("standalone eval stage not wired yet")

if __name__ == "__main__":
    train_main()
