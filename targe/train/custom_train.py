import json
from pathlib import Path
from typing import Optional

import torch
import trackio
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from targe.train.train_mm import (
    evaluate_on_test,
    print_trainable_breakdown,
    push_to_hub,
)


def _resolve_image_token_id(processor) -> Optional[int]:
    """Find the image-token id so we can mask it out of the labels (loss target)."""
    token_id = getattr(processor, "image_token_id", None)
    if token_id is not None:
        return token_id
    token_str = getattr(processor, "image_token", None)
    if token_str is None:
        return None
    return processor.tokenizer.convert_tokens_to_ids(token_str)


def _make_collate_fn(processor):
    """Apply chat template, run the processor, build labels with pad+image tokens masked."""
    pad_id = processor.tokenizer.pad_token_id
    image_token_id = _resolve_image_token_id(processor)

    def collate(batch):
        texts = [processor.apply_chat_template(s["messages"], tokenize=False) for s in batch]
        images = [s["images"] for s in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        inputs["labels"] = labels
        return inputs

    return collate


def _current_schedule(step: int, total_steps: int, warmup_steps: int, cfg_custom: DictConfig) -> tuple[float, float]:
    """Linear tau anneal high→low; lambda 0 during warmup then linear ramp to target."""
    tau_start = cfg_custom.get("tau_start", 1.0)
    tau_end = cfg_custom.get("tau_end", 0.1)
    lambda_target = cfg_custom.get("lambda_target", 0.05)

    progress_total = step / max(1, total_steps)
    tau = max(tau_end, tau_start - (tau_start - tau_end) * progress_total)

    if step < warmup_steps:
        lam = 0.0
    else:
        denom = max(1, total_steps - warmup_steps)
        lam = min(1.0, (step - warmup_steps) / denom) * lambda_target

    return tau, lam


def custom_train(model, processor, train_ds, eval_ds, test_ds, cfg: DictConfig) -> None:
    dry_run: bool = cfg.get("dry_run", False)
    dry_run_samples: int = cfg.get("dry_run_samples", 10)
    output_dir: str = cfg.get("train_output_dir", "smolvlm-run") + ("-dryrun" if dry_run else "")

    cfg_custom: DictConfig = cfg.custom

    if dry_run:
        train_ds = train_ds[:dry_run_samples]
        eval_ds = eval_ds[:dry_run_samples]
        test_ds = test_ds[:dry_run_samples]

    print_trainable_breakdown(model)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg_custom.get("per_device_train_batch_size", 1),
        shuffle=True,
        collate_fn=_make_collate_fn(processor),
    )

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg_custom.get("learning_rate", 5e-5)),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg_custom.get("gradient_accumulation_steps", 1),
    )
    # Model is already on GPU via device_map="auto" (+ optional bnb 4-bit), so we don't
    # re-prepare it — accelerate only needs the optimizer + dataloader to wire grads
    # and auto-move batches to the active device.
    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

    trackio.init(
        project=cfg.tracking.get("project", "targe-chartqa"),
        name=cfg.tracking.get("run_name") or ("dryrun-custom" if dry_run else "custom"),
        space_id=cfg.tracking.get("space_id"),
        dataset_id=cfg.tracking.get("dataset_id"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    num_epochs: int = cfg_custom.get("num_epochs", 1)
    max_steps_cfg = cfg_custom.get("max_steps", None)
    max_steps: Optional[int] = 1 if dry_run else (int(max_steps_cfg) if max_steps_cfg is not None else None)
    total_steps = max_steps if max_steps is not None else max(1, len(train_dataloader) * num_epochs)
    warmup_steps: int = min(cfg_custom.get("warmup_steps", 0), total_steps)
    log_every: int = max(1, cfg_custom.get("log_every", 10))

    model.train()
    connector = model.model.connector
    current_step = 0
    should_stop = False

    print('starting training loop')
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                current_tau, current_lambda = _current_schedule(
                    current_step, total_steps, warmup_steps, cfg_custom
                )

                if hasattr(connector, "tau"):
                    connector.tau = current_tau

                outputs = model(**batch)
                downstream_loss = outputs.loss

                keep_probs = getattr(connector, "latest_keep_probs", None)
                if keep_probs is not None:
                    l1_loss = current_lambda * keep_probs.mean()
                else:
                    l1_loss = torch.zeros((), device=downstream_loss.device, dtype=downstream_loss.dtype)

                total_loss = downstream_loss + l1_loss

                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

                if current_step % log_every == 0:
                    log_payload = {
                        "train/total_loss": float(total_loss.detach()),
                        "train/downstream_loss": float(downstream_loss.detach()),
                        "train/l1_loss": float(l1_loss.detach()),
                        "train/tau": current_tau,
                        "train/lambda": current_lambda,
                        "train/epoch": epoch,
                        "train/step": current_step,
                    }
                    if keep_probs is not None:
                        log_payload["train/avg_keep_prob"] = float(keep_probs.detach().mean())
                    trackio.log(log_payload, step=current_step)
                    print(
                        f"Step {current_step} | Total: {log_payload['train/total_loss']:.4f} | "
                        f"Down: {log_payload['train/downstream_loss']:.4f} | "
                        f"L1: {log_payload['train/l1_loss']:.4f} | tau: {current_tau:.3f} | "
                        f"lambda: {current_lambda:.4f}"
                    )

                current_step += 1
                if max_steps is not None and current_step >= max_steps:
                    should_stop = True
                    break
        if should_stop:
            break

    accelerator.wait_for_everyone()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    metrics, records = evaluate_on_test(unwrapped, processor, test_ds)
    trackio.log({f"test/{k}": v for k, v in metrics.items()})

    results_path = Path(output_dir) / "test_results.json"
    with results_path.open("w") as f:
        json.dump({"metrics": metrics, "records": records}, f, indent=2)

    push_to_hub(cfg.hub, output_dir, results_path)

    trackio.finish()
