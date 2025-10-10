from __future__ import annotations

"""
Persona card & memory helpers for your local chatbot.

Drop this file in: bot/services/persona_card.py

What you get:
- build_persona_card(chat_json_path, target_name, memory_dir="memory", save=True)
    -> returns (persona_card_text, memory_dict)
- load_persona_memory(target_name, memory_dir="memory")
- recall_memory(user_query, memory_dict)
- compose_pinned(persona_card_text)  # convenience to prepend to your prompt

Integration (minimal):
1) After you pick the persona in main.py (e.g., "Matias LOPEZ") and after you
   generate conversacion_completa.json, call build_persona_card(...). Keep the returned
   persona_card_text in memory and ALWAYS prepend it to the LLM prompt.
2) (Optional, recommended) Before answering, call recall_memory(user_query, memory) and
   if it returns a non-empty string, prepend that line to the assistant's answer or
   inject it as an instruction so the model starts with "Sí, recuerdo...".
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

# --------------------------
# Data models
# --------------------------

@dataclass
class MemoryHit:
    topic: str
    last_seen: str  # ISO date string
    sample: str

@dataclass
class PersonaMemory:
    name: str
    base_location: Optional[str]
    companies: List[str]
    style_notes: str
    emojis_top: List[str]
    topics: Dict[str, MemoryHit]  # canonical_topic -> MemoryHit
    last_updated: str

# --------------------------
# Public API
# --------------------------

def build_persona_card(
    chat_json_path: str | Path,
    target_name: str,
    memory_dir: str | Path = "memory",
    save: bool = True,
) -> Tuple[str, PersonaMemory]:
    """Build persona-card text and a structured memory object from WhatsApp export.

    Expects an array of messages like:
    {"date": "2/17/24", "time": "9:59:51 AM", "name": "Matias LOPEZ", "message": "..."}
    """
    chat_json_path = Path(chat_json_path)
    memory_dir = Path(memory_dir)
    data = _load_messages(chat_json_path)

    msgs = [m for m in data if (m.get("name") == target_name and isinstance(m.get("message"), str))]
    if not msgs:
        # fallback: accept case-insensitive name
        msgs = [m for m in data if (str(m.get("name", "")).lower() == target_name.lower() and isinstance(m.get("message"), str))]

    # Extract signals
    base_location = _infer_location(msgs)
    companies, topic_hits = _infer_companies_and_topics(msgs)
    emojis_top = _top_emojis(msgs, k=6)
    style_notes = _infer_style_notes(msgs)

    # Build memory map from topic hits
    topics_map: Dict[str, MemoryHit] = {}
    for canon, hit in topic_hits.items():
        topics_map[canon] = MemoryHit(topic=canon, last_seen=hit["last_seen"], sample=hit["sample"])  # type: ignore

    mem = PersonaMemory(
        name=target_name,
        base_location=base_location,
        companies=companies,
        style_notes=style_notes,
        emojis_top=emojis_top,
        topics=topics_map,
        last_updated=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    persona_card_text = _render_persona_card(mem)

    if save:
        _save_memory(mem, memory_dir)

    return persona_card_text, mem


def load_persona_memory(target_name: str, memory_dir: str | Path = "memory") -> Optional[PersonaMemory]:
    """Load previously saved memory JSON for a persona, if present."""
    p = _memory_path(target_name, memory_dir)
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    topics = {k: MemoryHit(**v) for k, v in obj.get("topics", {}).items()}
    return PersonaMemory(
        name=obj.get("name"),
        base_location=obj.get("base_location"),
        companies=obj.get("companies", []),
        style_notes=obj.get("style_notes", ""),
        emojis_top=obj.get("emojis_top", []),
        topics=topics,
        last_updated=obj.get("last_updated", ""),
    )

def recall_memory(user_query: str, memory: PersonaMemory, max_len: int = 120) -> str:
    """
    Devuelve una línea 'Sí, recuerdo…' SOLO si el usuario lo pide explícitamente
    (p. ej., 'te acuerdas', 'recuerdas'). No cita fechas ni texto literal.
    """
    q = user_query.lower()

    # Gatillos explícitos (evita que dispare por 'de dónde eres', etc.)
    explicit_triggers = [
        "te acuerdas", "¿te acuerdas", "recuerdas", "¿recuerdas",
        "te acordai", "te acordás", "te acordas", "acuerdas cuando", "recuerdas cuando",
    ]
    if not any(t in q for t in explicit_triggers):
        return ""

    # Intentamos identificar tema para personalizar la frase, sin citar texto/fechas
    topic = None

    # match por companies / topics
    for key in list(memory.topics.keys()) + [c.lower() for c in memory.companies]:
        if key in q:
            topic = key
            break

    # ubicación si el usuario menciona ciudad/país
    if not topic and memory.base_location:
        loc_tokens = _normalize_location(memory.base_location).split()
        if any(tok in q for tok in loc_tokens if len(tok) >= 3):
            topic = "ubicación"

    # Frases sin citas/fechas
    if topic == "ubicación" and memory.base_location:
        return f"Sí, recuerdo que te comenté que estoy en {memory.base_location}."
    if topic:
        # Capitaliza un poquito la etiqueta de tema
        pretty = topic.title() if topic.islower() else topic
        return f"Sí, recuerdo cuando hablamos de {pretty}."
    # fallback suave
    return "Sí, recuerdo que lo conversamos."

def compose_pinned(persona_card_text: str) -> str:
    """Return a block to prepend to your system prompt or top of the LLM input."""
    return (
        "[PERSONA]\n" + persona_card_text.strip() + "\n\n"
        "Instrucciones:\n"
        "- Mantén la identidad, tono y detalles anteriores.\n"
        "- Responde SOLO con base en el contexto y la memoria; si falta, dilo explícitamente.\n"
        "- Usa frases del tipo 'Sí, recuerdo…' **solo si el usuario lo pide explícitamente** (p. ej., '¿te acuerdas…?', '¿recuerdas…?').\n"
        "- No cites texto literal ni fechas del historial salvo que el usuario lo solicite.\n"
    )

# --------------------------
# Internal helpers
# --------------------------

_LOCATION_PATTERNS = [
    r"viña\s+del\s+mar",
    r"vina\s+del\s+mar",
    r"valpara[ií]so",
    r"g[oó]mez\s+carre[nñ]o",
    r"santiago",
    r"chile",
]

_COMPANY_HINTS = [
    r"suncast",
    r"llc\b", r"s\.a\.?\b", r"spa\b", r"corp\b", r"corporation\b", r"consulting\b", r"logistics\b",
    r"empresa\b", r"compa[nt][ií]a\b", r"trabajo\s+en\b", r"prueba\s+psicolog",
]

_EMOJI_BLOCKS = [
    (0x1F300, 0x1F5FF),  # symbols & pictographs
    (0x1F600, 0x1F64F),  # emoticons
    (0x1F680, 0x1F6FF),  # transport & map
    (0x2600, 0x26FF),    # misc symbols
    (0x2700, 0x27BF),    # dingbats
    (0x1F900, 0x1F9FF),  # supplemental symbols and pictographs
    (0x1FA70, 0x1FAFF),  # symbols & pictographs extended-A
]

_CHILEAN_MARKERS = [
    "po", "cachai", "wea", "bacán", "filete", "al tiro", "pega", "luca", "palta",
]


def _load_messages(path: Path) -> List[dict]:
    try:
        text = path.read_text(encoding="utf-8")
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    raise ValueError(f"No pude leer mensajes desde {path} (esperaba JSON array)")


def _infer_location(msgs: List[dict]) -> Optional[str]:
    # score by frequency of patterns
    scores = Counter()
    samples = defaultdict(list)
    for m in msgs:
        msg = str(m.get("message", ""))
        low = msg.lower()
        for pat in _LOCATION_PATTERNS:
            if re.search(pat, low):
                key = pat
                scores[key] += 1
                samples[key].append(msg)
    if not scores:
        return None
    best_pat, _ = scores.most_common(1)[0]
    # craft a human-friendly location from pattern
    mapping = {
        r"viña\s+del\s+mar": "Viña del Mar, Chile",
        r"vina\s+del\s+mar": "Viña del Mar, Chile",
        r"valpara[ií]so": "Valparaíso, Chile",
        r"g[oó]mez\s+carre[nñ]o": "Gómez Carreño, Viña del Mar, Chile",
        r"santiago": "Santiago, Chile",
        r"chile": "Chile",
    }
    return mapping.get(best_pat, "Chile")


def _infer_companies_and_topics(msgs: List[dict]) -> Tuple[List[str], Dict[str, dict]]:
    companies = Counter()
    topic_hits: Dict[str, dict] = {}

    for m in msgs:
        msg = str(m.get("message", ""))
        low = msg.lower()
        ts = _combine_dt(m.get("date"), m.get("time"))

        # company hints
        if any(re.search(p, low) for p in _COMPANY_HINTS):
            # crude proper-noun capture: words with initial caps or known brands
            caps = re.findall(r"\b([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑ0-9_-]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑ0-9_-]{2,})*)", msg)
            for c in caps:
                c_norm = c.strip().strip('.,;:!¿?"\'').lower()
                if len(c_norm) < 3:
                    continue
                if any(tok in c_norm for tok in ["hola", "ramon", "matias", "cliente", "direccion", "video", "whatsapp", "gmail"]):
                    continue
                companies[c_norm] += 1
                topic_hits.setdefault(c_norm, {"last_seen": ts, "sample": msg})
                # update last_seen if later
                if ts and topic_hits[c_norm]["last_seen"] and ts > topic_hits[c_norm]["last_seen"]:
                    topic_hits[c_norm]["last_seen"] = ts
                # keep a short sample
                if msg and (len(msg) < len(topic_hits[c_norm]["sample"]) or not topic_hits[c_norm]["sample"]):
                    topic_hits[c_norm]["sample"] = msg

        # explicit brand keywords (e.g., suncast)
        if "suncast" in low:
            topic_hits.setdefault("suncast", {"last_seen": ts, "sample": msg})
            companies["suncast"] += 1
            if ts and topic_hits["suncast"]["last_seen"] and ts > topic_hits["suncast"]["last_seen"]:
                topic_hits["suncast"]["last_seen"] = ts
            if msg and len(msg) < len(topic_hits["suncast"]["sample"]):
                topic_hits["suncast"]["sample"] = msg

    # pick top 5 companies by frequency
    top_companies = [c for c, _ in companies.most_common(5)]
    # ensure canonical lowercase -> pretty label
    top_companies_pretty = [c.title() if c.islower() else c for c in top_companies]

    return top_companies_pretty, topic_hits


def _top_emojis(msgs: List[dict], k: int = 6) -> List[str]:
    counts = Counter()
    for m in msgs:
        for ch in str(m.get("message", "")):
            code = ord(ch)
            if any(lo <= code <= hi for (lo, hi) in _EMOJI_BLOCKS):
                counts[ch] += 1
    return [e for e, _ in counts.most_common(k)]


def _infer_style_notes(msgs: List[dict]) -> str:
    text = "\n".join(str(m.get("message", "")) for m in msgs).lower()
    hints = []
    if any(w in text for w in _CHILEAN_MARKERS):
        hints.append("usa chilenismos de forma ligera (\"al tiro\", \"pega\", \"cachai\" si aplica)")
    if text.count("?") > text.count("."):
        hints.append("hace preguntas con frecuencia (mantén tono conversacional)")
    if any(x in text for x in ["gracias", "porfa", "por favor", "saludos"]):
        hints.append("amable y colaborativo, evita soniditos de call-center")
    if not hints:
        hints.append("tono coloquial y directo, sin exceso de formalidad")
    return "; ".join(hints)


def _render_persona_card(mem: PersonaMemory) -> str:
    lines = [
        f"Nombre: {mem.name}",
        f"Ubicación/Base: {mem.base_location or '—'}",
        f"Compañías/Temas: {', '.join(mem.companies) if mem.companies else '—'}",
        f"Estilo: {mem.style_notes}",
    ]
    if mem.emojis_top:
        lines.append(f"Emojis frecuentes: {' '.join(mem.emojis_top)}")

    # Add terse memory bullets from topics
    bullets = []
    for key, hit in mem.topics.items():
        dt = _try_parse_iso(hit.last_seen) or _try_parse_date_like(hit.last_seen)
        when = dt.strftime("%d/%m/%Y") if dt else hit.last_seen
        sample = hit.sample.replace("\n", " ")
        if len(sample) > 80:
            sample = sample[:79] + "…"
        bullets.append(f"- {key}: {sample} ({when})")
    if bullets:
        lines.append("Hechos recordados:")
        lines.extend(bullets[:6])

    return "\n".join(lines)


def _save_memory(mem: PersonaMemory, memory_dir: Path) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    p = _memory_path(mem.name, memory_dir)
    obj = asdict(mem)
    # dataclass MemoryHit -> dict
    obj["topics"] = {k: asdict(v) for k, v in mem.topics.items()}
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_path(name: str, memory_dir: str | Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return Path(memory_dir) / f"{slug}.json"


def _normalize_location(loc: str) -> str:
    return re.sub(r"\s+", " ", loc.strip()).lower()


def _combine_dt(date_str: Optional[str], time_str: Optional[str]) -> str:
    # Try to parse common WhatsApp-like formats, fall back to raw
    if not date_str:
        return time_str or ""
    s = f"{date_str} {time_str or ''}".strip()
    for fmt in ("%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M:%S", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except Exception:
            continue
    return s  # as-is


def _try_parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None


def _try_parse_date_like(s: str) -> Optional[datetime]:
    # Last-resort: dd/mm/yyyy
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if not m:
        return None
    d, mo, y = m.groups()
    y = ("20" + y) if len(y) == 2 else y
    try:
        return datetime(int(y), int(mo), int(d))
    except Exception:
        return None

# --------------------------
# Example usage (commented)
# --------------------------
# from bot.services.persona_card import build_persona_card, load_persona_memory, recall_memory, compose_pinned
# persona_text, memory = build_persona_card("conversacion_completa.json", target_name="Matias LOPEZ")
# pinned_block = compose_pinned(persona_text)
#
# # In your prompt assembly (pseudo):
# system_prompt = SYSTEM_BASE + "\n\n" + pinned_block
# user_query = input_text
# recall_line = recall_memory(user_query, memory)
# if recall_line:
#     # Option A: prepend to assistant reply after model returns (postprocess)
#     pass
#     # Option B: inject into prompt as instruction so the model starts with it.
#     # e.g., add to system: f"Si la consulta coincide con memoria, inicia con: '{recall_line.split(':')[0]}…'"
