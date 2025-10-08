# services/mode_detector.py
class ModeDetector:
  def __init__(self, default_mode: str = "short"):
    self.default_mode = default_mode

  def detect(self, user_message: str) -> str:
    msg = user_message.lower().strip()
    if "modo corto" in msg: return "short"
    if "modo largo" in msg or "modo chatgpt" in msg: return "long"
    long_triggers = ["explica","detalla","por qué","porque","paso a paso","tutorial",
                     "lista","enumera","código","codigo","ejemplo","ejemplos","propuesta",
                     "arquitectura","plan","justifica","razona"]
    if len(msg) > 120 or any(k in msg for k in long_triggers): return "long"
    return self.default_mode
