# utils/training.py
import json
from collections import Counter
from bot.utils.text import EMOJI_RE

def load_messages(path: str) -> list[dict]:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

def unique_names(records: list[dict]) -> list[str]:
  return sorted({(m.get("name") or "").strip() for m in records if m.get("name")})

def detect_persona_name(messages: list[dict], default="la persona") -> str:
  names = [m.get("name","") for m in messages if m.get("name")]
  return Counter(names).most_common(1)[0][0] if names else default

def top_emojis_from_messages(messages: list[dict], k: int=8) -> str:
  texts = [m.get("message","") for m in messages]
  from collections import Counter
  counts = Counter(ch for t in texts for ch in t if EMOJI_RE.match(ch))
  return "".join(e for e,_ in counts.most_common(k)) or "🙂"
