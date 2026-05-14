import json
from pathlib import Path

import trackio
from hydra.utils import instantiate
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf
from trl import SFTTrainer

from targe.infer import generate_text_from_sample_selector


def print_trainable_breakdown(model) -> None:
    breakdown: dict[str, int] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            component = name.split(".")[0] if "." in name else name
            breakdown[component] = breakdown.get(component, 0) + param.numel()

    print("Trainable Parameter Breakdown:")
    for component, count in breakdown.items():
        print(f"  {component}: {count:,}")


def _normalize(text: str) -> str:
    return text.strip().lower()


def evaluate_on_test(model, processor, test_ds, device: str = "cuda") -> tuple[dict, list[dict]]:
    """Greedy exact-match eval on the held-out test set."""
    records: list[dict] = []
    correct = 0
    for i, sample in enumerate(test_ds):
        gold = sample["messages"][-1]["content"][0]["text"]
        try:
            pred = generate_text_from_sample_selector(model, processor, sample, device=device)
        except Exception as exc:  # noqa: BLE001 — capture per-sample failures for the report
            pred = f"<error: {exc}>"
        is_correct = _normalize(pred) == _normalize(gold)
        correct += int(is_correct)
        records.append({"idx": i, "gold": gold, "pred": pred, "correct": is_correct})

    metrics = {
        "exact_match": correct / max(1, len(test_ds)),
        "n_samples": len(test_ds),
    }
    return metrics, records


def _push_to_hub(hub_cfg: DictConfig, output_dir: str, results_path: Path) -> None:
    if not hub_cfg.get("enabled", False):
        print("[hub] hub.enabled=false, skipping upload")
        return

    api = HfApi()
    repo_id = hub_cfg.repo_id
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=hub_cfg.get("private", True),
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hub_cfg.get("checkpoint_path_in_repo", "checkpoint"),
        commit_message=f"{hub_cfg.get('commit_message', 'targe run upload')}: checkpoint",
    )
    api.upload_file(
        path_or_fileobj=str(results_path),
        repo_id=repo_id,
        path_in_repo=hub_cfg.get("test_results_path_in_repo", "evaluation/test_results.json"),
        commit_message=f"{hub_cfg.get('commit_message', 'targe run upload')}: test results",
    )
    print(f"[hub] uploaded checkpoint + test_results.json to https://huggingface.co/{repo_id}")


def train(model, processor, train_ds, eval_ds, test_ds, cfg: DictConfig) -> None:
    dry_run: bool = cfg.get("dry_run", False)
    dry_run_samples: int = cfg.get("dry_run_samples", 10)
    output_dir: str = cfg.get("train_output_dir", "smolvlm-run") + ("-dryrun" if dry_run else "")

    # Materialize training + peft configs from hydra (`_target_` powered).
    training_args = instantiate(cfg.training)
    training_args.output_dir = output_dir
    peft_config = instantiate(cfg.peft)

    trackio.init(
        project=cfg.tracking.get("project", "targe-chartqa"),
        name=cfg.tracking.get("run_name") or (f"dryrun" if dry_run else None),
        space_id=cfg.tracking.get("space_id"),
        dataset_id=cfg.tracking.get("dataset_id"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    if dry_run:
        train_ds = train_ds[:dry_run_samples]
        eval_ds = eval_ds[:dry_run_samples]
        test_ds = test_ds[:dry_run_samples]

    print_trainable_breakdown(model)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=processor,
    )

    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(output_dir)

    # Held-out test eval — exact-match accuracy logged to trackio
    metrics, records = evaluate_on_test(model, processor, test_ds)
    trackio.log({f"test/{k}": v for k, v in metrics.items()})

    results_path = Path(output_dir) / "test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump({"metrics": metrics, "records": records}, f, indent=2)

    _push_to_hub(cfg.hub, output_dir, results_path)

    trackio.finish()
