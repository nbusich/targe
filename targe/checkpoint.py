import os

from safetensors.torch import load_file


def load_connector_weights(model, adapter_path: str, strict: bool = True):
    """
    Load only the custom connector tensors from a PEFT-saved adapter checkpoint.

    The trojan-horse PEFT setup also writes a tiny q_proj LoRA; those keys are filtered
    out so we restore a clean state_dict onto the wrapped or unwrapped connector module.
    """
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    raw = load_file(safetensors_path)

    clean = {}
    for key, value in raw.items():
        if "connector." in key:
            clean[key.split("connector.")[-1]] = value

    target = model.model.connector
    if hasattr(target, "original_module"):
        target = target.original_module
    target.load_state_dict(clean, strict=strict)

    print(f"Loaded {len(clean)} connector tensors from {adapter_path}")
    return model
