# services/math_guard.py
import re
MATH_RE = re.compile(r"^\s*(?:cu[aá]nto\s+es|calcula|compute|evalu[aá])?\s*([-+/*xX()\d\s\.]+)\s*\??\s*$", re.IGNORECASE)

def try_eval(user_message: str) -> str | None:
  m = MATH_RE.match(user_message)
  if not m: return None
  expr = m.group(1).replace("X","*").replace("x","*")
  if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expr): return None
  try:
    res = eval(expr, {"__builtins__": {}}, {})
    if isinstance(res, float) and res.is_integer(): res = int(res)
    return str(res)
  except Exception:
    return None
