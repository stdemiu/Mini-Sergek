import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np

class HunyuanPlateOCR:
    def __init__(self, model_name="tencent/HunyuanOCR", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def recognize(self, img):
        """
        img: grayscale or BGR numpy array
        returns: (text, conf_stub)
        """
        if img.ndim == 2:
            pil = Image.fromarray(img)
        else:
            pil = Image.fromarray(img[..., ::-1]) 
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs, max_new_tokens=64)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return text, 0.0  