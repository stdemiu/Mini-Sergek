import cv2
import numpy as np

def preprocess_plate(img, profile="clahe_sharp"):
    if img is None or img.size == 0:
        return img

    if profile == "none":
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if profile == "gray":
        return gray

    if profile in ("clahe", "clahe_sharp"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)

        if profile == "clahe":
            return g

        # лёгкая резкость (unsharp mask)
        blur = cv2.GaussianBlur(g, (0, 0), 1.2)
        sharp = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
        return sharp

    return gray

def upscale(img, scale=2.5):
    if img is None or img.size == 0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
