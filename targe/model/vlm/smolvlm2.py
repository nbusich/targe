import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"


def load_smolvlm(
    model_id: str = DEFAULT_MODEL_ID,
    quantize_4bit: bool = True,
    attn_implementation: str = "sdpa",
):
    quantization_config = None
    if quantize_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        _attn_implementation=attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
