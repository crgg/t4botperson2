# utils/history.py
import json
def update_history(hist: list[dict], user: str, assistant: str) -> None:
  hist += [{"role":"user","content":user},{"role":"assistant","content":assistant}]

def save_conversation(hist: list[dict], filename: str="conversacion_guardada.json"):
  with open(filename, "w", encoding="utf-8") as f:
    json.dump(hist, f, ensure_ascii=False, indent=2)
