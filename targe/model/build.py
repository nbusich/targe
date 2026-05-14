import torch
from omegaconf import DictConfig
from hydra.utils import get_class, instantiate

from targe.model.vlm.smolvlm2 import load_smolvlm


def build_model(cfg: DictConfig):
    model_id = cfg.get("model_id", "HuggingFaceTB/SmolVLM-Instruct")
    quantize_4bit = cfg.get("quantize_4bit", True)
    attn_implementation = cfg.get("attn_implementation", "sdpa")

    model, processor = load_smolvlm(
        model_id,
        quantize_4bit=quantize_4bit,
        attn_implementation=attn_implementation,
    )

    print(f"Building custom connector: {cfg.connector._target_}")
    # Build the dataclass from yaml, then fill in `device` from the live model
    # (a runtime-only field). The connector class itself is resolved via _target_.
    custom_params = instantiate(cfg.connector.custom_params)
    custom_params.device = str(model.device)
    connector_cls = get_class(cfg.connector._target_)
    connector = connector_cls(model.config, custom_params)
    connector._init_weights()
    connector = torch.compile(connector)
    model.model.connector = connector.to(device=model.device, dtype=torch.bfloat16)

    if cfg.get("freeze_backbone", True):
        for name, param in model.named_parameters():
            param.requires_grad = "connector" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {model_id}")
    print(f"Trainable parameters: {trainable:,} ({trainable/total_params:.2%} of total)")
    return model, processor
