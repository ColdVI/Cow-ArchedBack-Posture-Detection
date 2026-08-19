from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def extract_resnet18_embeddings(
    image_paths: Sequence[str | Path],
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """Extract frozen 512-D ResNet18 features with official weight transforms."""
    try:
        import torch
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "Torch/TorchVision are required for embedding and fusion models. "
            "Install requirements.txt or train only the geometry model."
        ) from exc

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(device)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            embedding = model(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(embedding)
    if not outputs:
        return np.empty((0, 512), dtype=np.float32)
    return np.concatenate(outputs, axis=0)

