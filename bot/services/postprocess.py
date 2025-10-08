# services/postprocess.py
import re
from bot.utils.text import sentence_split, normalize
from bot.utils.patterns import FATIC_LINES, CALL_CENTER_PHRASES

def compile_label_regex(names: list[str]) -> re.Pattern:
  base = ["Usuario", "User"]
  labels = base + [n for n in names if n]
  alt = "|".join(re.escape(x) for x in labels)
  return re.compile(rf"^(?:{alt})\s*:\s*.*$", re.MULTILINE)

class PostProcessor:
  def __init__(self, banned_phrases: list[str], banned_slang: list[str],
               participant_names: list[str], max_short_chars: int):
    self.banned_phrases = banned_phrases
    self.banned_slang = banned_slang
    self.participant_names = participant_names
    self.max_short_chars = max_short_chars
    self.label_regex = compile_label_regex(participant_names)

  def _user_greeted(self, user_message: str) -> bool:
    msg = normalize(user_message)
    tokens = ["hola","buenas","buenos dias","buenas tardes","buenas noches","¿como estas",
              "como estas","qué tal","que tal"]
    return any(t in msg for t in tokens)

  def _strip_phatic_prefix(self, text: str) -> str:
    pat = re.compile(r"^\s*(hola|buenas(?:\s+(tardes|noches|d[ií]as))?|buenos d[ií]as|que tal|qué tal|como estas|c[oó]mo est[aá]s|bien(,)? gracias!?)[,!\.\:]?\s*", re.IGNORECASE)
    if len(text) > 25:
      text = pat.sub("", text).strip()
    return text

  def _strip_phatic_sentences(self, text: str, user_message: str) -> str:
    if self._user_greeted(user_message): return text
    kept = [s for s in sentence_split(text) if not any(re.search(p, s, re.IGNORECASE) for p in FATIC_LINES)]
    out = " ".join(kept).strip()
    return out or text

  def _normalize_spanish_cl(self, text: str) -> str:
    for bad in self.banned_slang:
      text = re.sub(rf"\b{re.escape(bad)}\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()

  def _strip_action_offers(self, text: str) -> str:
    offer_pats = [
      r"\b(llamarte|te llamo|te marco|puedo llamarte|dame tu n[uú]mero|n[uú]mero de tel[eé]fono)\b",
      r"\b(agend[ae]mos|agenda una llamada|programar una llamada)\b",
      r"\b(te escribo por correo|te mando un mail|te env[ií]o un correo)\b",
      r"\b(dame|p[aá]same|env[ií]ame|m[aá]ndame) (tu )?correo( electr[oó]nico)?\b",
      r"\b(te\s*(mando|env[ií]o|paso)\s*(el|un|los)\s*(cv|curr[ií]culum|archivo|archivos))\b",
    ]
    sents = sentence_split(text)
    return " ".join([s for s in sents if not any(re.search(p, s, re.IGNORECASE) for p in offer_pats)]).strip()

  def _strip_logistics(self, text: str, user_message: str) -> str:
    low = user_message.lower()
    if any(k in low for k in ["hora","horario","agenda","disponible","martes","jueves","hoy","mañana"]):
      return text
    pat = re.compile(r"\b(horas?|horario|disponible|de\s+\d{1,2}\s*(am|pm)|\d{1,2}:\d{2})\b", re.IGNORECASE)
    kept = [s for s in sentence_split(text) if not pat.search(s)]
    out = " ".join(kept).strip()
    return out or text

  def _strip_name_if_not_present(self, user_message: str, text: str) -> str:
    msg = normalize(user_message)
    allowed = set()
    for n in self.participant_names:
      n_norm = normalize(n); short = n_norm.split()[0] if n_norm else ""
      if n_norm and n_norm in msg: allowed.add(n_norm)
      if short and short in msg: allowed.add(short)
    names = []
    for n in self.participant_names:
      names.append(n)
      if " " in n: names.append(n.split()[0])
    if not names: return text
    name_pat = re.compile(rf"^\s*(?:{'|'.join(map(re.escape, names))})\s*[:,]\s*", re.IGNORECASE)
    if not any(a in msg for a in allowed):
      text = name_pat.sub("", text).strip()
    return text

  def run(self, text: str, mode: str, user_message: str) -> str:
    text = self.label_regex.sub("", text).strip()
    text = re.sub(r"[\u2022\-\*]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for b in self.banned_phrases:
      if b.lower() in text.lower():
        text = re.sub(re.escape(b), "", text, flags=re.IGNORECASE).strip()
    text = self._normalize_spanish_cl(text)
    text = self._strip_action_offers(text)
    if not self._user_greeted(user_message):
      text = self._strip_phatic_prefix(text)
    text = self._strip_phatic_sentences(text, user_message)
    text = self._strip_logistics(text, user_message)
    text = self._strip_name_if_not_present(user_message, text)
    if mode == "short" and len(text) > self.max_short_chars:
      text = text[: self.max_short_chars - 3].rstrip() + "…"
    return text or "No cacho."
