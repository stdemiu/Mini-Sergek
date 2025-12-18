import re

LETTERS = "ABCEHKMOPTXY"

def normalize_plate(text: str) -> str:
    if not text:
        return ""

    t = text.upper()


    t = re.sub(r"[^A-Z0-9]", "", t)


    t = t.replace("О", "O").replace("Q", "0")
    t = t.replace("I", "1").replace("Z", "2")
    t = t.replace("S", "5")
    t = t.replace("B", "8")


    m = re.search(rf"(\d{{2,3}})([{LETTERS}]{{3}})(\d{{2}})", t)
    if m:
        return "".join(m.groups())

    m = re.search(rf"(\d{{2,3}})([{LETTERS}]{{2}})(\d{{2}})", t)
    if m:
        return "".join(m.groups())

    return ""
