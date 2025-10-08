# utils/text.py
import re
TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

def tok(s: str): return [t.lower() for t in TOKEN_RE.findall(s or "")]
def normalize(s: str) -> str: 
  return re.sub(r"\s+", " ", (s or "").lower().strip())
def sentence_split(text: str):
  return re.split(r"(?<=[\.\!\?…])\s+", text or "")
