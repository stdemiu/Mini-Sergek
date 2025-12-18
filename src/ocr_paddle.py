import cv2
import numpy as np
from paddleocr import PaddleOCR

class PaddlePlateOCR:
    def __init__(self):
        print("[INFO] Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en")

    def recognize(self, img):
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return "", 0.0

        if len(img.shape) == 2:
            inp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            inp = img

        try:
            res = self.ocr.ocr(inp)
        except Exception as e:
            print(f"[OCR ERROR] {e}")
            return "", 0.0

        best_text, best_conf = "", 0.0
        if not res:
            return "", 0.0

        for line in res:
            for item in line:
                text, conf = "", 0.0

                if isinstance(item, dict):
                    text = str(item.get("rec_text", "")).strip()
                    conf = float(item.get("rec_score", 0.0))
                elif isinstance(item, (list, tuple)):
                    if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                        text = str(item[1][0]).strip() if len(item[1]) >= 1 else ""
                        conf = float(item[1][1]) if len(item[1]) >= 2 else 0.0

                if text and conf >= best_conf:
                    best_text, best_conf = text, conf

        return best_text, best_conf
