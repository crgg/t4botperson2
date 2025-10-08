# services/rag_context.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Sequence
from rag.embed import ensure_rag_index, retrieve_robust
from bot.utils.text import tok
from bot.utils.patterns import GENERAL_Q_RE
from collections import defaultdict

class RagContextBuilder:
  def __init__(self, training_messages: List[Dict], embed_fn, alpha: float,
               fetch_k: int, gating_dense_sim_threshold: float,
               lexical_signal_threshold: float, multiquery_min_chars: int,
               sparse_index=None):
    self.training_messages = training_messages
    self.embed_fn = embed_fn
    self.alpha = alpha
    self.fetch_k = fetch_k
    self.gating = gating_dense_sim_threshold
    self.lex_th = lexical_signal_threshold
    self.multi_min = multiquery_min_chars

    self.corpus_texts, self.corpus_vecs, self.sparse_index = ensure_rag_index(training_messages)
    self.text_to_indices = defaultdict(list)
    for i, t in enumerate(self.corpus_texts):
      self.text_to_indices[t].append(i)

  # --- señales ---
  def _lexical_signal(self, q: str) -> float:
    if not self.sparse_index: return 0.0
    q_terms = set(tok(q))
    idf = self.sparse_index.get("idf", {})
    max_score = 0.0
    for terms in self.sparse_index.get("docs", []):
      inter = q_terms & terms
      s = sum(idf.get(t, 0.0) for t in inter)
      if s > max_score: max_score = s
    return max_score

  def should_use_rag(self, user_message: str, is_math: bool, is_short_greet: bool) -> bool:
    low = user_message.lower().strip()
    if is_math or is_short_greet:
        return False

    # NUEVO: si es muy corto y no es pregunta ni pide explicar → NO RAG
    is_question = ("?" in low) or any(k in low for k in ["qué", "como", "cómo", "por qué", "porque", "explícame", "dime"])
    if len(low) < 50 and not is_question:
        return False

    # ya tienes este filtro general:
    from bot.utils.patterns import GENERAL_Q_RE
    if GENERAL_Q_RE.search(low):
        return False

    # endurece un poco la señal léxica
    return self._lexical_signal(user_message) >= (self.lex_th + 0.4)
  # --- construcción del <privado> ---
  def build_private_context(self, user_message: str) -> str:
    top_texts = retrieve_robust(
      user_message, self.corpus_texts, self.corpus_vecs,
      sparse_index=self.sparse_index, k=5, alpha=self.alpha,
      fetch_k=self.fetch_k, use_multiquery=(len(user_message) > self.multi_min),
      use_mmr=True, use_llm_rerank=False
    )
    if not top_texts: return ""

    qv = np.array(self.embed_fn(user_message), dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    best_sim = 0.0
    for txt in top_texts:
      for idx in self.text_to_indices.get(txt, []):
        best_sim = max(best_sim, float(self.corpus_vecs[idx] @ qv))
    if best_sim < self.gating: return ""

    ctx = "\n".join(f"- {t}" for t in top_texts)
    return f"<privado>\n{ctx}\n</privado>\n"
