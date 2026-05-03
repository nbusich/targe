import torch

from targe.model.connector import (
    Idefics3SelectorConnector,
    SmolVLMInstructConnectorConfig,
)
from targe.model.vlm.smolvlm2 import load_smolvlm


def build_model(
    model_id: str | None = None,
    connector_config: SmolVLMInstructConnectorConfig | None = None,
    freeze_backbone: bool = True,
    quantize_4bit: bool = True,
):
    """Load the base VLM, swap in the custom selector connector, and freeze the backbone."""
    model, processor = (
        load_smolvlm(model_id, quantize_4bit=quantize_4bit)
        if model_id
        else load_smolvlm(quantize_4bit=quantize_4bit)
    )

    cfg = connector_config or SmolVLMInstructConnectorConfig(device=str(model.device))
    connector = Idefics3SelectorConnector(model.config, cfg)
    connector._init_weights()
    model.model.connector = connector.to(device=model.device, dtype=torch.bfloat16)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = "connector" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters in custom connector: {trainable:,}")
    return model, processor
