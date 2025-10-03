# bot/local_chatbot.py
# ——————————————————————————————————————————————————————————
# Chat local con estilo WhatsApp + mejoras:
#  - Modo corto/largo (tipo WhatsApp vs. tipo ChatGPT)
#  - RAG híbrido con “gating” estricto (solo entra cuando aporta)
#  - Few-shot como mensajes (no incrustados en el prompt)
#  - Filtros anti “call-center”, anti regionalismos y anti small-talk
#  - Guard de matemáticas básicas (respuestas correctas)
#  - Limpieza dinámica de rótulos (Usuario:, Nombres:) y logística
# ——————————————————————————————————————————————————————————

import os
import re
import json
import time
import shutil
import subprocess
import numpy as np
from typing import List, Dict, Optional
from collections import Counter, defaultdict

import ollama
from rag.embed import ensure_rag_index, retrieve_robust, EMBED_MODEL

# Emojis comunes (rango Unicode amplio)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

# ——— utilidades de tokenización y patrones globales ———
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)
def _tok(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

# Preguntas generales que NO deberían disparar RAG (definiciones, “quién fue…”, etc.)
_GENERAL_Q_PAT = re.compile(
    r"\b(que\s+es|qué\s+es|quien\s+fue|quién\s+fue|definici[oó]n|como\s+funciona|cómo\s+funciona)\b",
    re.IGNORECASE,
)

# Frases fáticas que solemos querer borrar si el usuario no saludó
_FATIC_PATS = [
    r"^\s*hola(,|\s|$)",
    r"^\s*(buenas|buenos d[ií]as|buenas tardes|buenas noches)\b",
    r"^\s*c[oó]mo est[aá]s\b",
    r"^\s*bien(,)? gracias\b",
    r"^\s*disculpa por no contestar\b",
    r"^\s*estuve sin internet\b",
    r"^\s*espero que est[ée]s bien\b",
]


def _unique_names(records: List[Dict]) -> List[str]:
    """Devuelve una lista ordenada de nombres únicos presentes en los registros."""
    return sorted({(m.get("name") or "").strip() for m in records if m.get("name")})


class LocalChatbot:
    def __init__(self, messages_file: str, model_name: str = "mistral"):
        self.model_name = model_name
        self.messages_file = messages_file
        self.conversation_history: List[Dict] = []

        # ----------------------------
        # 1) Datos de entrenamiento
        # ----------------------------
        with open(messages_file, "r", encoding="utf-8") as f:
            self.training_messages = json.load(f)

        # Persona + emojis (definir ANTES del prompt)
        names = [m.get("name", "") for m in self.training_messages if m.get("name")]
        self.persona_name = Counter(names).most_common(1)[0][0] if names else "la persona"
        self.style_emojis = self._top_emojis([m.get("message", "") for m in self.training_messages])

        # Config de estilo/longitud (puedes cambiar el modo por defecto a "short")
        self.default_mode = "long"      # "short" (tipo WhatsApp) | "long" (tipo ChatGPT)
        self.max_short_chars = 180

        # ----------------------------
        # 2) Infra: Ollama + embeddings
        # ----------------------------
        self._check_ollama_and_models()
        self.corpus_texts, self.corpus_vecs, self.sparse_index = ensure_rag_index(self.training_messages)

        # Mapa texto→lista de índices (por si hay duplicados)
        self.text_to_indices = defaultdict(list)
        for i, t in enumerate(self.corpus_texts):
            self.text_to_indices[t].append(i)

        # ----------------------------
        # 3) Few-shot + nombres participantes
        # ----------------------------
        self.fewshot_pairs: List = []
        self.participant_names: List[str] = []
        try:
            with open("conversacion_completa.json", "r", encoding="utf-8") as f:
                all_msgs = json.load(f)
            self.fewshot_pairs = self._pair_examples(all_msgs, max_pairs=6)
            self.participant_names = _unique_names(all_msgs)
        except Exception:
            # Fallback si no existe el archivo
            self.participant_names = _unique_names(self.training_messages) or [self.persona_name]

        self.fewshot_messages = self._build_fewshot_messages()
        self.label_regex = self._compile_label_regex(self.participant_names)

        # Frases “call-center” a filtrar (puedes ampliar esta lista)
        self.banned_phrases = [
            "¿Cómo puedo ayudarte hoy?",
            "estoy aquí para ayudarte",
            "aquí hay algunos ejemplos",
        ]

        # Regionalismos a evitar (no presentes en tu dataset o no deseados)
        self.banned_slang = [
            "chido", "wey", "órale",         # MX
            "vale", "tío",                   # ES
            "che", "pibe",                   # AR
        ]

        # ----------------------------
        # 4) Prompt del sistema (sin ejemplos incrustados)
        # ----------------------------
        self.system_prompt = self._create_system_prompt()

    # ========= Helpers de inicialización =========

    def _top_emojis(self, texts: List[str], k: int = 8) -> str:
        counts = Counter(ch for t in texts for ch in t if EMOJI_RE.match(ch))
        return "".join(e for e, _ in counts.most_common(k)) or "🙂"

    def _check_ollama_and_models(self):
        # Asegura que el daemon está arriba
        try:
            ollama.list()
        except Exception:
            cand = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
            if not os.path.exists(cand):
                raise SystemExit("Ollama CLI no encontrado. Instálalo con Homebrew o el .pkg.")
            subprocess.Popen([cand, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    ollama.list()
                    break
                except Exception:
                    pass
            else:
                raise SystemExit("No pude conectar con el daemon de Ollama. Ejecuta 'ollama serve' en otra terminal.")
        # Asegura que existan los modelos (generación + embeddings)
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
        for m in (self.model_name, EMBED_MODEL):
            if m not in out:
                print(f"\nDescargando {m}...")
                r = subprocess.run(["ollama", "pull", m])
                if r.returncode != 0:
                    raise SystemExit(f"Fallo el pull de {m}. Revisa el daemon.")
        print(f"✓ Ollama y modelos listos: {self.model_name} + {EMBED_MODEL}")

    def _pair_examples(self, all_messages: List[Dict], max_pairs: int = 6):
        """Crea pares (otro → persona) consecutivos, acotados en longitud, para few-shot."""
        pairs = []
        for i in range(1, len(all_messages)):
            cur, prev = all_messages[i], all_messages[i - 1]
            if cur.get("name") == self.persona_name and prev.get("name") != self.persona_name:
                if len(prev.get("message", "")) <= 220 and len(cur.get("message", "")) <= 220:
                    pairs.append((prev["message"], cur["message"]))
        return pairs[-max_pairs:]

    def _build_fewshot_messages(self):
        """Convierte los pares en mensajes rolados (no texto dentro del prompt)."""
        msgs = []
        for u, a in getattr(self, "fewshot_pairs", []):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        return msgs

    def _compile_label_regex(self, names: List[str]) -> re.Pattern:
        """
        Regex para borrar líneas del tipo:
        Usuario: ..., User: ..., Matías LOPEZ: ..., Ramon Gajardo: ...
        """
        base = ["Usuario", "User"]
        labels = base + [n for n in names if n]
        alt = "|".join(re.escape(x) for x in labels)
        pat = rf"^(?:{alt})\s*:\s*.*$"
        return re.compile(pat, re.MULTILINE)

    # ========= Normalización y utilidades de texto =========

    def _normalize(self, s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    def _user_greeted(self, user_message: str) -> bool:
        msg = self._normalize(user_message)
        tokens = [
            "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
            "¿como estas", "como estas", "qué tal", "que tal",
        ]
        return any(t in msg for t in tokens)

    def _sentence_split(self, text: str):
        # Split simple por signos de puntuación fuertes
        return re.split(r"(?<=[\.\!\?…])\s+", text)

    def _strip_phatic(self, text: str) -> str:
        # Si el modelo arranca con saludos/cortesías y el usuario NO saludó, los quitamos.
        pat = re.compile(
            r"^\s*(hola|buenas(?:\s+(tardes|noches|d[ií]as))?|buenos d[ií]as|que tal|qué tal|como estas|c[oó]mo est[aá]s|bien(,)? gracias!?)[,!\.\:]?\s*",
            re.IGNORECASE,
        )
        # Quitamos solo el arranque fático; si el texto entero es saludo, lo dejamos.
        if len(text) > 25:
            text = pat.sub("", text).strip()
        return text

    def _strip_phatic_sentences(self, text: str, user_message: str) -> str:
        if self._user_greeted(user_message):
            return text  # si el usuario saludó, respeta
        sents = self._sentence_split(text)
        kept = []
        for s in sents:
            if any(re.search(p, s, re.IGNORECASE) for p in _FATIC_PATS):
                continue
            kept.append(s)
        out = " ".join(kept).strip()
        return out or text

    def _strip_name_if_not_present(self, user_message: str, text: str) -> str:
        # Si el usuario no usó el nombre en este turno, evitamos que el bot lo anteponga (“Hola Ramon,”)
        msg = self._normalize(user_message)
        allowed = set()
        for n in self.participant_names:
            n_norm = self._normalize(n)
            short = n_norm.split()[0] if n_norm else ""
            if n_norm and n_norm in msg:
                allowed.add(n_norm)
            if short and short in msg:
                allowed.add(short)

        # Patrón dinámico para “Nombre:” o “Nombre,” al inicio
        names_for_regex = []
        for n in self.participant_names:
            parts = [n]
            # también nombres cortos
            if " " in n:
                parts.append(n.split()[0])
            for p in parts:
                if p:
                    names_for_regex.append(re.escape(p))
        if not names_for_regex:
            return text

        name_pat = re.compile(rf"^\s*(?:{'|'.join(names_for_regex)})\s*[:,]\s*", re.IGNORECASE)
        # Solo lo removemos si NO está permitido (usuario no lo dijo)
        if not any(a in msg for a in allowed):
            text = name_pat.sub("", text).strip()
        return text

    def _strip_action_offers(self, text: str) -> str:
        # Remueve oraciones que ofrecen llamadas/acciones fuera del chat
        offer_pats = [
            r"\b(llamarte|te llamo|te marco|puedo llamarte|dame tu n[uú]mero|n[uú]mero de tel[eé]fono)\b",
            r"\b(agend[ae]mos|agenda una llamada|programar una llamada)\b",
            r"\b(te escribo por correo|te mando un mail|te envío un correo)\b",
        ]
        sentences = re.split(r"(?<=[\.\!\?…])\s+", text)
        kept = [s for s in sentences if not any(re.search(p, s, re.IGNORECASE) for p in offer_pats)]
        return " ".join(kept).strip()

    def _strip_logistics(self, text: str, user_message: str) -> str:
        # Evita horas/agenda si el usuario no lo pidió
        low = user_message.lower()
        if any(k in low for k in ["hora", "horario", "agenda", "disponible", "martes", "jueves", "hoy", "mañana"]):
            return text
        sents = self._sentence_split(text)
        pat = re.compile(r"\b(horas?|horario|disponible|de\s+\d{1,2}\s*(am|pm)|\d{1,2}:\d{2})\b", re.IGNORECASE)
        kept = [s for s in sents if not pat.search(s)]
        out = " ".join(kept).strip()
        return out or text

    # ========= Prompt de sistema =========

    def _create_system_prompt(self) -> str:
        # Reglas sin ejemplos incrustados
        prompt = (
            f"Eres {self.persona_name} chateando por WhatsApp. Hablas SIEMPRE en español chileno.\n\n"
            "REGLAS DURAS (incumplir = respuesta incorrecta):\n"
            "- Natural y preciso. Si no sabes, di: “no cacho / no sé”.\n"
            "- No inventes estados/acciones (llamadas, correos, adjuntos, estar offline). No ofrezcas llamarme ni pedir números.\n"
            "- Evita regionalismos ajenos a tu estilo (p.ej., “chido”, “wey”, “vale”, “tío”, “che”).\n"
        #    "- No menciones “contexto”, “posibles respuestas” ni analices la conversación.\n"
            "- No digas que eres un modelo/IA. Eres " + self.persona_name + ".\n"
            "- Nunca reveles ni cites lo que venga entre <privado>…</privado>.\n"
            "- Si el usuario pide explicación o código, puedes extenderte en párrafos claros.\n"
            f"- Emojis de tu estilo: {self.style_emojis} (con moderación).\n"
        )
        return prompt

    # ========= Detección de "modo" (corto/largo) =========

    def _detect_mode(self, user_message: str) -> str:
        """
        Heurística simple:
          - Si el usuario pide explicación/código o el mensaje es largo → 'long'
          - Si es saludo, pregunta corta, small talk → 'short'
          - Se permite forzar modo con palabras clave.
        """
        msg = user_message.lower().strip()

        if "modo corto" in msg:
            return "short"
        if "modo largo" in msg or "modo chatgpt" in msg:
            return "long"

        long_triggers = [
            "explica", "detalla", "por qué", "porque", "paso a paso",
            "tutorial", "lista", "enumera", "código", "codigo",
            "ejemplo", "ejemplos", "propuesta", "arquitectura",
            "plan", "justifica", "razona"
        ]
        if len(msg) > 120 or any(k in msg for k in long_triggers):
            return "long"

        # Por defecto:
        return self.default_mode

    # ========= Guard de matemáticas básicas =========

    _MATH_PATTERN = re.compile(
        r"^\s*(?:cu[aá]nto\s+es|calcula|compute|evalu[aá])?\s*([-+/*xX()\d\s\.]+)\s*\??\s*$",
        re.IGNORECASE,
    )

    def _math_guard(self, user_message: str) -> Optional[str]:
        """
        Si detecta una expresión aritmética simple, la evalúa de forma segura
        para evitar errores del modelo (p.ej., 5*5=25).
        """
        m = self._MATH_PATTERN.match(user_message)
        if not m:
            return None

        expr = m.group(1)
        # Normaliza 'x' a '*'
        expr = expr.replace("X", "*").replace("x", "*")
        # Permitir sólo dígitos, operadores básicos y paréntesis
        if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expr):
            return None

        try:
            # Eval “seguro” con entorno vacío
            res = eval(expr, {"__builtins__": {}}, {})
            # Evita floats “feos”
            if isinstance(res, float) and res.is_integer():
                res = int(res)
            return str(res)
        except Exception:
            return None

    # ========= Señales para decidir si usar RAG =========

    def _lexical_signal(self, user_message: str) -> float:
        """Señal lexical (IDF) máxima vs. corpus; si es baja, RAG no ayuda."""
        if not getattr(self, "sparse_index", None):
            return 0.0
        q_terms = set(_tok(user_message))
        idf = self.sparse_index.get("idf", {})
        max_score = 0.0
        for terms in self.sparse_index.get("docs", []):
            inter = q_terms & terms
            s = sum(idf.get(t, 0.0) for t in inter)
            if s > max_score:
                max_score = s
        return max_score

    def _should_use_rag(self, user_message: str) -> bool:
        """Heurística: no uses RAG en saludos, mates y definiciones generales."""
        msg = user_message.strip()
        low = msg.lower()
        if self._MATH_PATTERN.match(msg):                    # mates simples
            return False
        if self._user_greeted(msg) and len(msg) < 40:        # saludo corto
            return False
        if _GENERAL_Q_PAT.search(low):                       # “qué es”, “quién fue”… (genérico)
            return False
        # Señal lexical mínima para “traer del chat”
        return self._lexical_signal(msg) >= 1.2

    # ========= Gating de RAG + recuperación robusta =========

    def _build_private_context(self, user_message: str) -> str:
        """
        Devuelve el bloque <privado>…</privado> usando recuperación robusta (hybrid + MMR).
        Aplica 'gating' si la similitud densa del mejor candidato es baja.
        """
        # 0) ¿Conviene RAG para este turno?
        if not self._should_use_rag(user_message):
            return ""

        # 1) Recuperación robusta (textos)
        top_texts = retrieve_robust(
            user_message,
            self.corpus_texts,
            self.corpus_vecs,
            sparse_index=self.sparse_index,
            k=5,
            alpha=0.6,         # más alto = más peso a embeddings; más bajo = más peso lexical
            fetch_k=32,        # candidatos amplios para que MMR funcione bien
            use_multiquery=(len(user_message) > 80),  # sin multiquery para turnos cortos
            use_mmr=True,
            use_llm_rerank=False,
        )

        if not top_texts:
            return ""

        # 2) 'Gating' por similitud densa del MEJOR candidato (umbral estricto)
        try:
            qv = ollama.embeddings(model=EMBED_MODEL, input=user_message)["embedding"]
        except TypeError:
            qv = ollama.embeddings(model=EMBED_MODEL, prompt=user_message)["embedding"]
        qv = np.array(qv, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)

        best_sim = 0.0
        for txt in top_texts:
            for idx in self.text_to_indices.get(txt, []):
                best_sim = max(best_sim, float(self.corpus_vecs[idx] @ qv))

        if best_sim < 0.60:
            return ""  # si nada es suficientemente parecido, no metas RAG (evita ruido)

        # 3) Contexto comprimido
        contexto_privado = "\n".join(f"- {t}" for t in top_texts)
        return f"<privado>\n{contexto_privado}\n</privado>\n"

    # ========= Chat =========

    def chat(self, user_message: str) -> str:
        # 0) Guard matemático (mejora precisión frente a modelos chicos)
        math_ans = self._math_guard(user_message)
        if math_ans is not None:
            out = math_ans
            self._update_history(user_message, out)
            return out

        # 1) Modo (corto/largo)
        mode = self._detect_mode(user_message)

        # 2) Contexto privado (gating de RAG)
        private_ctx = self._build_private_context(user_message)

        # 3) Instrucción según modo (enfasis en “solo responde a la consulta”)
        if mode == "short":
            instr = (
                f"{private_ctx}"
                f"Responde SOLO a la consulta. No agregues saludos ni información no solicitada. "
                f"Habla como {self.persona_name} (es-CL), natural y breve.\n"
                f"{user_message}"
            )
            opts = {
                "seed": 7,
                "temperature": 0.15,
                "top_p": 0.85,
                "repeat_penalty": 1.15,
                "num_predict": 96,      # tope duro → respuestas cortas
                "num_ctx": 4096,
                "stop": [
                    "Usuario:", "User:", "EJEMPLOS", "Ejemplos",
                    "En cuanto a las conversaciones",
                    "Posibles respuestas", "Possible responses",
                    "¿Cómo estás", "como estas", "¿cómo estás", "Como estas",
                    "Disculpa por", "perdón por", "Perdón por", "Espero que estés bien",
                ],
            }
        else:
            instr = (
                f"{private_ctx}"
                f"Responde SOLO a la consulta con claridad (tipo ChatGPT) y sin divagar. "
                f"No agregues saludos ni asuntos personales si el usuario no los pidió. "
                f"Habla como {self.persona_name} (es-CL).\n"
                f"{user_message}"
            )
            opts = {
                "seed": 7,
                "temperature": 0.20,
                "top_p": 0.90,
                "repeat_penalty": 1.10,
                "num_predict": 512,     # más largo si se pidió
                "num_ctx": 4096,
                "stop": [
                    "Usuario:", "User:", "EJEMPLOS", "Ejemplos",
                    "¿Cómo estás", "como estas", "¿cómo estás", "Como estas",
                    "Disculpa por", "perdón por", "Perdón por", "Espero que estés bien",
                ],
            }

        # 4) Mensajes: system + fewshot + historial + turno actual
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.fewshot_messages,            # ejemplos como diálogo (no en el prompt)
            *self.conversation_history[-10:],  # menos historial = menos deriva
            {"role": "user", "content": instr},
        ]

        # 5) Llamada al modelo
        resp = ollama.chat(model=self.model_name, messages=messages, options=opts)
        out = resp["message"]["content"]

        # 6) Post-proceso (filtros de rótulos, fáticos, logística, etc.)
        out = self._postprocess(out, mode, user_message)

        # 7) Historial
        self._update_history(user_message, out)
        return out

    # ========= Post-proceso =========

    def _normalize_spanish_cl(self, text: str) -> str:
        """
        Limpia regionalismos no deseados. Por defecto, elimina términos.
        (Si prefieres reemplazar, puedes mapear a 'bacán', 'al tiro', etc.)
        """
        for bad in self.banned_slang:
            text = re.sub(rf"\b{re.escape(bad)}\b", "", text, flags=re.IGNORECASE)
        # Espacios dobles tras eliminación
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _postprocess(self, text: str, mode: str, user_message: str) -> str:
        # 1) Quitar rótulos de diálogo "falso"
        text = self.label_regex.sub("", text).strip()

        # 2) Colapsar bullets y espacios
        text = re.sub(r"[\u2022\-\*]\s+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 3) Frases “call-center” prohibidas
        for b in self.banned_phrases:
            if b.lower() in text.lower():
                text = re.sub(re.escape(b), "", text, flags=re.IGNORECASE).strip()

        # 4) Regionalismos no deseados (MX, ES, AR, etc.)
        text = self._normalize_spanish_cl(text)

        # 5) Eliminar ofertas de llamadas/correos
        text = self._strip_action_offers(text)

        # 6a) Si el usuario NO saludó, elimina saludos/fáticos al inicio
        if not self._user_greeted(user_message):
            text = self._strip_phatic(text)

        # 6b) Quita fáticos también en medio del texto
        text = self._strip_phatic_sentences(text, user_message)

        # 6c) Evita logística (horas/agenda) si no fue pedida
        text = self._strip_logistics(text, user_message)

        # 7) No anteponer nombres si el usuario no lo dijo en este turno
        text = self._strip_name_if_not_present(user_message, text)

        # 8) Limitar longitud solo en modo corto
        if mode == "short" and len(text) > self.max_short_chars:
            text = text[: self.max_short_chars - 3].rstrip() + "…"

        return text or "No cacho."

    # ========= Guardado =========

    def save_conversation(self, filename: str = "conversacion_guardada.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"✓ Conversación guardada en {filename}")

    # ========= Utils =========

    def _update_history(self, user_message: str, assistant_message: str):
        self.conversation_history += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
