import time

import torch


def benchmark_models(model, processor, sample, iterations: int = 10, device: str = "cuda"):
    """Time generate() over `iterations` runs and report estimated projection FLOPs."""
    model.eval()
    inputs = processor(
        text="<image>Answer:",
        images=[sample["images"][0]],
        return_tensors="pt",
    ).to(device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model.generate(**inputs, max_new_tokens=5)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations

    std_flops = 10368 * 2048 * 81
    custom_flops = 1152 * 2048 * 92

    print("--- Benchmark Results ---")
    print(f"Custom Model Inf Time: {elapsed:.4f}s")
    print(f"Standard Model Est. FLOPs (Proj): {std_flops:,}")
    print(f"Custom Model Est. FLOPs (Proj):   {custom_flops:,}")
    print(f"Efficiency Gain: {std_flops / custom_flops:.2f}x fewer FLOPs in projection")
    return {
        "inference_time_s": elapsed,
        "std_flops": std_flops,
        "custom_flops": custom_flops,
    }
