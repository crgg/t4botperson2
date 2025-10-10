# bot/local_chatbot.py
from __future__ import annotations
from typing import List, Dict, Optional
import os
import re
import textwrap

from bot.config import build_config, ChatbotConfig
from bot.prompts import make_system_prompt, make_instruction

from bot.utils.training import load_messages, unique_names, detect_persona_name, top_emojis_from_messages
from bot.utils.history import update_history, save_conversation as save_hist
from bot.utils.text import EMOJI_RE

from bot.services.ollama_service import OllamaService
from bot.services.mode_detector import ModeDetector
from bot.services.math_guard import try_eval as math_guard
from bot.services.postprocess import PostProcessor
from bot.services.fewshot_builder import pair_examples, build_fewshot_messages
from bot.services.rag_context import RagContextBuilder

# Persona Card / Memoria
from bot.services.persona_card import recall_memory, PersonaMemory

# Import opcional del módulo de media
try:
    from media.ingest import ingest_media_dir, save_media_index
except Exception:
    ingest_media_dir = None
    save_media_index = None


def _safe_get(obj, key, default=None):
    """Soporta dict u objeto con atributos."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class LocalChatbot:
    def __init__(
        self,
        messages_file: str,
        model_name: str = "mistral",
        overrides: dict | None = None,
        persona_pinned: str | None = None,           # bloque anclado [PERSONA] para el prompt
        persona_memory: PersonaMemory | None = None  # memoria estructurada para “Sí, recuerdo…”
    ):
        self.cfg: ChatbotConfig = build_config(model_name=model_name, overrides=overrides)
        self.messages_file = messages_file
        self.conversation_history: List[Dict] = []

        # Persona (ancla + memoria)
        self.persona_pinned = persona_pinned
        self.persona_memory = persona_memory

        # Cooldown para no abusar de “Sí, recuerdo…”
        self._last_recall_turn = -999
        self._recall_cooldown = 10  # no repetir hasta 3 turnos después

        # Datos base (txt de WhatsApp ya procesado)
        self.training_messages = load_messages(messages_file)
        self.persona_name = detect_persona_name(self.training_messages)
        self.style_emojis = top_emojis_from_messages(self.training_messages, k=self.cfg.rag.top_emojis_k)

        # ================== Ingesta multimodal -> TEXTO ==================
        docs = []  # importante: inicializado para evitar NameError si algo falla
        media_cfg = getattr(self.cfg, "media", None)

        if media_cfg and ingest_media_dir:
            try:
                # Algunos ingest_media_dir aceptan ocr=..., otros no
                try:
                    docs = ingest_media_dir(
                        media_dir=media_cfg.media_dir,
                        vision_model=media_cfg.vision_model,        # puede ser None -> solo OCR/transcripción
                        sum_model=self.cfg.model.model_name,        # modelo textual para resumen
                        every_sec=media_cfg.frame_stride_sec,
                        max_frames=media_cfg.max_frames,
                        ocr=True,                                   # si tu función lo soporta, fuerza OCR en imágenes
                    ) or []
                except TypeError:
                    # Fallback si la firma no acepta ocr=...
                    docs = ingest_media_dir(
                        media_dir=media_cfg.media_dir,
                        vision_model=media_cfg.vision_model,
                        sum_model=self.cfg.model.model_name,
                        every_sec=media_cfg.frame_stride_sec,
                        max_frames=media_cfg.max_frames,
                    ) or []

                # Guardar índice de media para auditar
                if save_media_index and docs:
                    try:
                        os.makedirs(media_cfg.media_dir, exist_ok=True)
                        save_media_index(docs, os.path.join(media_cfg.media_dir, "media_index.json"))
                    except Exception:
                        pass

                # --- DEBUG: resumen de lo extraído ---
                print(f"[media] docs extraídos: {len(docs)}")

                # Dump de texto de IMÁGENES en consola
                if docs:
                    print("\n[media] === IMÁGENES extraídas (OCR/Caption) ===")
                    img_count = 0
                    for i, d in enumerate(docs, 1):
                        kind = (_safe_get(d, "kind", "") or "").lower()
                        path = _safe_get(d, "path", "") or ""
                        # Preferimos 'text'; si tu ingest separa en caption/ocr también los consideramos
                        text = _safe_get(d, "text", "") or _safe_get(d, "caption", "") or _safe_get(d, "ocr", "")
                        is_image = ("image" in kind) or path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
                        if is_image and text:
                            img_count += 1
                            name = os.path.basename(path) if path else f"img_{i}"
                            pretty = textwrap.shorten(" ".join(text.split()), width=1000, placeholder=" …")
                            print(f"  [{img_count}] {name} [{kind or 'image'}]:\n      {pretty}\n")
                    if img_count == 0:
                        print("  (no se detectó texto en imágenes)")

                # Inyectar TEXTO de imágenes/videos como mensajes de la persona (entra al RAG)
                added = 0
                for d in docs:
                    text = _safe_get(d, "text", None)
                    if text:
                        self.training_messages.append({"name": self.persona_name, "message": text})
                        added += 1
                print(f"[media] mensajes añadidos a training_messages: {added}")

                # (Opcional) Dump en TXT para revisar offline
                if added:
                    try:
                        with open(os.path.join(media_cfg.media_dir, "images_text_dump.txt"), "w", encoding="utf-8") as f:
                            for d in docs:
                                path = _safe_get(d, "path", "") or ""
                                text = _safe_get(d, "text", "")
                                kind = (_safe_get(d, "kind", "") or "")
                                if text:
                                    name = os.path.basename(path) if path else "doc"
                                    f.write(f"[{kind}] {name}\n{text}\n\n")
                        print(f"[media] dump guardado en {os.path.join(media_cfg.media_dir, 'images_text_dump.txt')}")
                    except Exception:
                        pass

            except Exception as e:
                print(f"[media] Ingesta falló: {e}")
        else:
            if not media_cfg:
                print("[media] Configuración de media no encontrada; se omite ingesta.")
            elif not ingest_media_dir:
                print("[media] Módulo media.ingest no disponible; se omite ingesta.")
        # ===================================================================

        # Infra
        self.ollama = OllamaService(self.cfg.model.model_name, self.cfg.model.embed_model)

        # Few-shot
        try:
            import json
            with open("conversacion_completa.json", "r", encoding="utf-8") as f:
                all_msgs = json.load(f)
            pairs = pair_examples(all_msgs, self.persona_name, max_pairs=6)
            participant_names = unique_names(all_msgs)
        except Exception:
            pairs = []
            participant_names = unique_names(self.training_messages) or [self.persona_name]

        self.fewshot_messages = build_fewshot_messages(pairs)
        self.participant_names = participant_names

        # Servicios de lógica
        self.mode = ModeDetector(self.cfg.style.default_mode)
        self.post = PostProcessor(
            banned_phrases=list(self.cfg.style.banned_phrases),
            banned_slang=list(self.cfg.style.banned_slang),
            participant_names=self.participant_names,
            max_short_chars=self.cfg.style.max_short_chars,
        )

        # RAG (ya incluye lo de media porque lo agregamos a training_messages)
        print(f"[rag] total mensajes para indexar: {len(self.training_messages)}")
        self.rag = RagContextBuilder(
            training_messages=self.training_messages,
            embed_fn=self.ollama.embed,
            alpha=self.cfg.rag.alpha,
            fetch_k=self.cfg.rag.fetch_k,
            gating_dense_sim_threshold=self.cfg.rag.gating_dense_sim_threshold,
            lexical_signal_threshold=self.cfg.rag.lexical_signal_threshold,
            multiquery_min_chars=self.cfg.rag.multiquery_min_chars,
        )

        # Prompt sistema (anclar Persona Card si viene)
        base_system = make_system_prompt(self.persona_name, self.style_emojis)
        if self.persona_pinned:
            # Ancla la identidad/tono al inicio del prompt del modelo
            self.system_prompt = f"{self.persona_pinned}\n{base_system}"
        else:
            self.system_prompt = base_system

    # --- API pública ---
    def chat(self, user_message: str) -> str:
        math_ans = math_guard(user_message)
        if math_ans is not None:
            update_history(self.conversation_history, user_message, math_ans)
            return math_ans

        mode = self.mode.detect(user_message)

        is_short_greet = len(user_message.strip()) < 40 and self._user_greeted(user_message)

        # Fuerza RAG si la pregunta es de identidad/ubicación/empresa (evita alucinaciones tipo "soy argentino")
        force_identity = self._force_identity_lookup(user_message)
        use_rag = self.rag.should_use_rag(user_message, is_math=False, is_short_greet=is_short_greet) or force_identity
        private_ctx = self.rag.build_private_context(user_message) if use_rag else ""

        instr = make_instruction(mode, self.persona_name, user_message, private_ctx)
        opts = self.cfg.gen.short if mode == "short" else self.cfg.gen.long

        messages = [{"role": "system", "content": self.system_prompt},
                    *self.fewshot_messages,
                    *self.conversation_history[-10:],
                    {"role": "user", "content": instr}]
        raw = self.ollama.chat(messages, options=opts)

        # Efecto memoria: SOLO si el usuario lo pide explícitamente (lo controla recall_memory)
        if self.persona_memory:
            mem_hint = recall_memory(user_message, self.persona_memory)
            turn = len(self.conversation_history)
            if mem_hint and not self._starts_with_recall(raw) and (turn - self._last_recall_turn >= self._recall_cooldown):
                raw = f"{mem_hint}\n\n{raw}"
                self._last_recall_turn = turn

        out = self.post.run(raw, mode, user_message)
        update_history(self.conversation_history, user_message, out)
        return out

    def save_conversation(self, filename: str = "conversacion_guardada.json"):
        save_hist(self.conversation_history, filename)

    def _user_greeted(self, s: str) -> bool:
        return self.post._user_greeted(s)

    def _force_identity_lookup(self, s: str) -> bool:
        """Dispara RAG para preguntas típicas de identidad/ubicación/empresa (no dispara 'sí, recuerdo')."""
        q = s.lower()
        triggers = [
            "de donde", "de dónde", "donde vives", "dónde vives",
            "ubicacion", "ubicación", "ciudad", "país", "pais",
            "de que ciudad", "de qué ciudad", "desde donde", "desde dónde",
            "empresa", "trabajas", "suncast",
            "de donde me escribes", "desde donde me escribes",
        ]
        return any(t in q for t in triggers)

    def _starts_with_recall(self, text: str) -> bool:
        """Detecta si la respuesta ya comienza con 'sí/si, recuerdo' (con o sin tilde)."""
        return bool(re.match(r"^(sí|si)\\s*,?\\s*recuerdo", text.strip(), flags=re.IGNORECASE))
