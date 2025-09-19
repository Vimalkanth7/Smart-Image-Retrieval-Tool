import torch
import numpy as np
from PIL import Image
from typing import List
import open_clip
from transformers import BlipForConditionalGeneration, AutoProcessor


class ImageTextEncoder:
    """
    OpenCLIP encoder (ViT-B/32, laion2b_s34b_b79k).
    Provides 512-dim normalized embeddings for images and text.
    """
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()

    @torch.no_grad()
    def embed_image(self, pil_img: Image.Image) -> np.ndarray:
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(x)  # [1, d]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(tokens)  # [1, d]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).float().cpu().numpy()


class BlipCaptioner:
    """
    BLIP captioner (Salesforce/blip-image-captioning-base)
    """
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.model.eval()

        
    @torch.no_grad()
    def caption(self, pil_img: Image.Image, max_new_tokens: int = 60) -> str:
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            repetition_penalty=1.2
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return _clean_caption(text)

    # @torch.no_grad()
    # def caption(self, pil_img: Image.Image, max_new_tokens: int = 20) -> str:
    #     inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
    #     out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3, repetition_penalty=1.2)
    #     text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
    #     return _clean_caption(text)


def _clean_caption(text: str) -> str:
    # simple tidy-up for BLIP quirks
    t = text.strip().lower()
    while "  " in t:
        t = t.replace("  ", " ")
    return t
