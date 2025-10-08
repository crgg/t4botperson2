# utils/patterns.py
import re
GENERAL_Q_RE = re.compile(r"\b(que\s+es|qué\s+es|quien\s+fue|quién\s+fue|definici[oó]n|como\s+funciona|cómo\s+funciona)\b", re.IGNORECASE)
FATIC_LINES = [
  r"^\s*hola(,|\s|$)", r"^\s*(buenas|buenos d[ií]as|buenas tardes|buenas noches)\b",
  r"^\s*c[oó]mo est[aá]s\b", r"^\s*bien(,)? gracias\b", r"^\s*disculpa por no contestar\b",
  r"^\s*estuve sin internet\b", r"^\s*espero que est[ée]s bien\b",
]
CALL_CENTER_PHRASES = []  # rellena con tus frases prohibidas
