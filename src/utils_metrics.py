import re

def normalize_plate(text: str) -> str:
    """
    Оставляем только A-Z и 0-9, верхний регистр.
    """
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text
