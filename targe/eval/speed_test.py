"""
Compare FLOPS/input between models
"""
import torch
from torchvision.models import resnet50  # Example model
from thop import profile  # Import the profile function from THOP


def evaluate_model_speed(model, dummy_input):
    macs, params = profile(model, inputs=(dummy_input,))
    print(f"MACs: {macs}, Parameters: {params}")


