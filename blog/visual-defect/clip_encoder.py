from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
EMBED_DIM = 512


def _features(out) -> torch.Tensor:
    """
    transformers 4.x returns a Tensor from get_image_features and
    get_text_features. transformers 5.x returns a
    BaseModelOutputWithPooling whose pooler_output holds the
    projected vector. Handle both so this does not break on upgrade.
    """
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class ClipEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def encode_images(self, paths: Sequence[str | Path]) -> torch.Tensor:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        vecs = _features(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(vecs, dim=-1).cpu()

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        vecs = _features(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(vecs, dim=-1).cpu()