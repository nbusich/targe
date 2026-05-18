"""
Entry point for training/eval.

Everything is hydra-controlled. Examples:
    python -m targe.main                                  # full SFT training run
    python -m targe.main trainer=custom                   # hand-rolled accelerate loop
    python -m targe.main +experiment=dry                  # SFT dry-run preset
    python -m targe.main +experiment=dry trainer=custom   # custom-loop dry-run
    python -m targe.main hub.repo_id=usich-n/foo train_output_dir=runs/foo
    python -m targe.main hub.enabled=false                # skip Hub upload
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from targe.data.chartqa import load_chartqa
from targe.model.build import build_model
from targe.train.custom_train import custom_train
from targe.train.train_mm import train as sft_train
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model, processor = build_model(cfg.model)
    train_ds, eval_ds, test_ds = load_chartqa(cfg.data)

    if cfg.get("train", False):
        trainer = cfg.get("trainer", "sft")
        if trainer == "sft":
            sft_train(model, processor, train_ds, eval_ds, test_ds, cfg)
        elif trainer == "custom":
            print("entering training")
            custom_train(model, processor, train_ds, eval_ds, test_ds, cfg)
        else:
            raise ValueError(f"Unknown trainer={trainer!r}. Use 'sft' or 'custom'.")

    if cfg.get("eval", False):
        raise NotImplementedError("standalone eval stage not wired yet")


if __name__ == "__main__":
    train_main()
