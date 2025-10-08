# bot/local_chatbot.py
from __future__ import annotations
from typing import List, Dict, Optional
import re

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

# NUEVO: import opcional del módulo de media
try:
    from media.ingest import ingest_media_dir, save_media_index
except Exception:
    ingest_media_dir = None
    save_media_index = None

class LocalChatbot:
  def __init__(self, messages_file: str, model_name: str = "mistral", overrides: dict | None = None):
    self.cfg: ChatbotConfig = build_config(model_name=model_name, overrides=overrides)
    self.messages_file = messages_file
    self.conversation_history: List[Dict] = []

    # Datos base (txt de WhatsApp ya procesado)
    self.training_messages = load_messages(messages_file)
    self.persona_name = detect_persona_name(self.training_messages)
    self.style_emojis = top_emojis_from_messages(self.training_messages, k=self.cfg.rag.top_emojis_k)

    # ================== NUEVO: ingesta multimodal -> TEXTO ==================
    media_cfg = getattr(self.cfg, "media", None)
    if media_cfg and ingest_media_dir:
        try:
            docs = ingest_media_dir(
                media_dir=media_cfg.media_dir,
                vision_model=media_cfg.vision_model,   # puede ser None
                sum_model=self.cfg.model.model_name,   # usa tu modelo textual para resumir
                every_sec=media_cfg.frame_stride_sec,
                max_frames=media_cfg.max_frames,
            )
            if save_media_index and docs:
                try:
                    save_media_index(docs, "media/media_index.json")  # opcional, para debug
                except Exception:
                    pass
            # Inyecta el TEXTO de imágenes/videos como mensajes de la persona (entra al RAG)
            for d in docs or []:
                if getattr(d, "text", None):
                    self.training_messages.append({"name": self.persona_name, "message": d.text})
        except Exception as e:
            print(f"[media] Ingesta falló: {e}")
    # =======================================================================

    # Infra
    self.ollama = OllamaService(self.cfg.model.model_name, self.cfg.model.embed_model)

    # Few-shot
    try:
      import json
      with open("conversacion_completa.json","r",encoding="utf-8") as f:
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
    self.rag = RagContextBuilder(
      training_messages=self.training_messages,   # <- ya incluye texto de .jpg/.mp4
      embed_fn=self.ollama.embed,
      alpha=self.cfg.rag.alpha,
      fetch_k=self.cfg.rag.fetch_k,
      gating_dense_sim_threshold=self.cfg.rag.gating_dense_sim_threshold,
      lexical_signal_threshold=self.cfg.rag.lexical_signal_threshold,
      multiquery_min_chars=self.cfg.rag.multiquery_min_chars,
    )

    # Prompt sistema
    self.system_prompt = make_system_prompt(self.persona_name, self.style_emojis)

  # --- API pública (SIN CAMBIOS) ---
  def chat(self, user_message: str) -> str:
    math_ans = math_guard(user_message)
    if math_ans is not None:
      update_history(self.conversation_history, user_message, math_ans)
      return math_ans

    mode = self.mode.detect(user_message)

    is_short_greet = len(user_message.strip()) < 40 and self._user_greeted(user_message)
    private_ctx = self.rag.build_private_context(user_message) if \
      self.rag.should_use_rag(user_message, is_math=False, is_short_greet=is_short_greet) else ""

    instr = make_instruction(mode, self.persona_name, user_message, private_ctx)
    opts = self.cfg.gen.short if mode == "short" else self.cfg.gen.long

    messages = [{"role":"system","content": self.system_prompt},
                *self.fewshot_messages,
                *self.conversation_history[-10:],
                {"role":"user","content": instr}]
    raw = self.ollama.chat(messages, options=opts)

    out = self.post.run(raw, mode, user_message)
    update_history(self.conversation_history, user_message, out)
    return out

  def save_conversation(self, filename: str = "conversacion_guardada.json"):
    save_hist(self.conversation_history, filename)

  def _user_greeted(self, s: str) -> bool:
    return self.post._user_greeted(s)
