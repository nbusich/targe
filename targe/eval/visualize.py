import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def visualize_isolated_patches(original_image, selected_indices, grid_size=(27, 27)):
    """Black out non-selected patches so only the high-density tokens remain visible."""
    img_np = np.array(original_image)
    H, W, _ = img_np.shape

    num_tokens = grid_size[0] * grid_size[1]
    mask = torch.zeros(num_tokens)
    mask[selected_indices.long()] = 1.0
    mask = mask.view(1, 1, grid_size[0], grid_size[1])

    upscaled = F.interpolate(mask, size=(H, W), mode="nearest").squeeze().numpy()
    masked_img = (img_np * upscaled[:, :, np.newaxis]).astype(np.uint8)

    plt.figure(figsize=(12, 12))
    plt.imshow(masked_img)
    plt.axis("off")
    plt.title(f"Isolated Vision Inputs (Top-{len(selected_indices)} Tokens)")
    plt.show()


def visualize_sample_with_selection(model, sample, grid_size=(27, 27)):
    """Pull the most recent selector indices off the model and render them over the sample image."""
    selected = model.model.connector.last_top_idx[0].cpu()
    visualize_isolated_patches(sample["images"][0], selected, grid_size=grid_size)
