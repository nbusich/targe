from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from targe.data.ChartQA import load_chartqa
from targe.model.build import build_model


def make_training_args(output_dir: str = "smolvlm-instruct-trl-sft-ChartQA") -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=12,
        save_total_limit=2,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=True,
        report_to="none",
        max_length=None,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
    )


def make_peft_config() -> LoraConfig:
    """
    Trojan-horse PEFT config: r=1 q_proj LoRA so SFTTrainer doesn't crash,
    while modules_to_save unfreezes the custom connector for full training.
    """
    return LoraConfig(
        r=1,
        lora_alpha=1,
        target_modules=["q_proj"],
        modules_to_save=["connector"],
    )


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


def main() -> None:
    model, processor = build_model()
    train_ds, eval_ds, _ = load_chartqa()

    args = make_training_args()
    print_trainable_breakdown(model)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=make_peft_config(),
        processing_class=processor,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
