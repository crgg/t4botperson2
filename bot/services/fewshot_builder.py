# services/fewshot_builder.py
from collections import Counter
import json, re

def top_emojis(texts: list[str], k: int = 8, emoji_re=None) -> str:
  from collections import Counter
  counts = Counter(ch for t in texts for ch in t if emoji_re and emoji_re.match(ch))
  return "".join(e for e,_ in counts.most_common(k)) or "🙂"

def pair_examples(all_messages: list[dict], persona_name: str, max_pairs: int = 6):
  pairs = []
  for i in range(1, len(all_messages)):
    cur, prev = all_messages[i], all_messages[i - 1]
    if cur.get("name")==persona_name and prev.get("name")!=persona_name:
      if len(prev.get("message",""))<=220 and len(cur.get("message",""))<=220:
        pairs.append((prev["message"], cur["message"]))
  return pairs[-max_pairs:]

def build_fewshot_messages(pairs: list[tuple[str,str]]) -> list[dict]:
  msgs=[]
  for u,a in pairs:
    msgs += [{"role":"user","content":u},{"role":"assistant","content":a}]
  return msgs
