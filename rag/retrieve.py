# Paso 10: orquestación de recuperación robusta (consulta en tiempo real)
# Ejecuta el pipeline de RAG en cada turno de usuario:
# 1) (opcional) multi-query (multiquery.py) para variantes de la consulta.
# 2) híbrido (hybrid.py): mezcla denso+léxico; une candidatos de todas las variantes.
# 3) (opcional) MMR (mmr.py) para diversidad del conjunto final.
# 4) (opcional) re-ranking con LLM (rerank.py) para la última pasada de calidad.
# - retrieve_robust(query, corpus_texts, corpus_vecs, sparse_index, k, ...)
#   devuelve los k textos más útiles para meter en <privado>…</privado>.
# - No construye el índice; asume que index_io.ensure_rag_index ya lo dejó listo.

# rag/retrieve.py
import os, json, hashlib, math, re
from typing import List, Tuple
import numpy as np
import ollama

from .embeddings import embed_texts, EMBED_MODEL
from .sparse_index import build_sparse_index, dump_sparse_index, load_sparse_index

# --- utilidades locales ---
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)
def _tok(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

def _sparse_scores(query: str, sparse_index):
    q_terms = set(_tok(query))
    idf = sparse_index["idf"]
    docs = sparse_index["docs"]
    scores = []
    for terms in docs:
        inter = q_terms & terms
        s = sum(idf.get(t, 0.0) for t in inter)
        scores.append(s)
    return np.array(scores, dtype=np.float32)

def _minmax_norm(arr: np.ndarray):
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _hash_corpus(texts: List[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update((t or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()

# --- recuperación híbrida + MMR ---
def _hybrid_top_k(query, corpus_texts, corpus_vecs, sparse_index, k=8, alpha=0.6, fetch_k=32):
    """
    Mezcla similitud densa (embeddings) y dispersa (IDF).
    Devuelve (pre_idx, combo_scores) y deja a quien llama decidir top-k final.
    """
    qv = embed_texts([query])[0]
    dense = corpus_vecs @ qv                   # (N,)
    sparse = _sparse_scores(query, sparse_index)  # (N,)

    d = _minmax_norm(dense)
    s = _minmax_norm(sparse)
    combo = alpha * d + (1 - alpha) * s

    pre_idx = combo.argsort()[-fetch_k:][::-1]
    return pre_idx, combo, qv

def _mmr_rerank(query_vec: np.ndarray, candidate_idx, doc_vecs: np.ndarray, lambda_mmr=0.5, k=8):
    selected = []
    candidates = list(candidate_idx)
    sims_q = {i: float(doc_vecs[i] @ query_vec) for i in candidates}
    while candidates and len(selected) < k:
        best_i, best_score = None, -1e9
        for i in candidates:
            sim_to_q = sims_q[i]
            sim_to_sel = 0.0
            if selected:
                sim_to_sel = max(float(doc_vecs[i] @ doc_vecs[j]) for j in selected)
            score = lambda_mmr * sim_to_q - (1 - lambda_mmr) * sim_to_sel
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        candidates.remove(best_i)
    return selected

def _multi_query_expansion(query, n=3, model="mistral"):
    """
    Genera n variantes del query con el LLM vía Ollama. Si falla, devuelve [query].
    """
    try:
        prompt = (
            "Parafrasea la siguiente consulta en 3 variantes concisas y distintas, una por línea, "
            "sin numeración adicional:\n\n"
            f"Consulta: {query}\n\nVariantes:"
        )
        out = ollama.generate(model=model, prompt=prompt)
        lines = [l.strip("-• ").strip() for l in out["response"].splitlines() if l.strip()]
        seen, variants = set(), []
        for l in lines:
            if l and l.lower() != query.lower() and l not in seen:
                variants.append(l); seen.add(l)
            if len(variants) >= n:
                break
        return variants or [query]
    except Exception:
        return [query]

# --- API pública: retrieve_robust ---
def retrieve_robust(
    query: str,
    corpus_texts: List[str],
    corpus_vecs: np.ndarray,
    sparse_index=None,
    k: int = 8,
    alpha: float = 0.6,
    fetch_k: int = 32,
    use_multiquery: bool = True,
    use_mmr: bool = True,
    use_llm_rerank: bool = False,
    rerank_model: str = "mistral",
) -> List[str]:
    """
    1) (Opcional) multi-query → candidatos por híbrido (denso+léxico) y unión.
    2) (Opcional) MMR para diversidad.
    3) (Opcional) LLM re-ranking (caro).
    Devuelve: lista de textos top-k.
    """
    if sparse_index is None:
        # si no te pasan índice disperso, constrúyelo “on the fly”
        sparse_index = build_sparse_index(corpus_texts)

    queries = [query]
    if use_multiquery:
        queries += _multi_query_expansion(query, n=3, model=rerank_model)

    combo_scores = np.zeros(len(corpus_texts), dtype=np.float32)
    seen = set()
    last_qv = None
    for q in queries:
        pre_idx, combo, qv = _hybrid_top_k(q, corpus_texts, corpus_vecs, sparse_index,
                                           k=k, alpha=alpha, fetch_k=fetch_k)
        combo_scores[pre_idx] += combo[pre_idx]
        seen.update(pre_idx.tolist())
        last_qv = qv

    candidates = sorted(seen, key=lambda i: float(combo_scores[i]), reverse=True)[:max(fetch_k, k)]

    if use_mmr and len(candidates) > 1:
        qv = last_qv if last_qv is not None else embed_texts([query])[0]
        candidates = _mmr_rerank(qv, candidates, corpus_vecs, lambda_mmr=0.5, k=max(k, 10))

    top_texts = [corpus_texts[i] for i in candidates[:max(k, 10)]]

    if use_llm_rerank and len(top_texts) > k:
        try:
            numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(top_texts))
            prompt = (
                "Selecciona los fragmentos más útiles para responder la consulta. "
                "Devuelve SOLO una lista de índices entre corchetes separados por comas, por ejemplo: [0,2,3]\n\n"
                f"Consulta: {query}\n\nFragmentos:\n{numbered}\n\nÍndices:"
            )
            out = ollama.generate(model=rerank_model, prompt=prompt)
            text = out["response"]
            m = re.search(r"\[(.*?)\]", text)
            idxs = [int(x.strip()) for x in m.group(1).split(",")] if m else []
            idxs = [i for i in idxs if 0 <= i < len(top_texts)]
            if idxs:
                return [top_texts[i] for i in idxs[:k]]
        except Exception:
            pass

    return top_texts[:k]

# --- API pública: ensure_rag_index (con hash + mmap) ---
def ensure_rag_index(
    training_messages: List[dict],
    texts_path: str = "rag_texts.json",
    vecs_path: str = "rag_vecs.npy",
    sparse_path: str = "rag_sparse.json",
    meta_path: str = "rag_meta.json",
) -> Tuple[list, np.ndarray, dict]:
    """
    Crea/carga el índice RAG desde disco.
    Optimiza RAM con mmap y evita recomputar si el corpus no cambió.
    """
    corpus_texts = [m.get("message") for m in training_messages if m.get("message")]
    cur_hash = _hash_corpus(corpus_texts)

    need_rebuild = True
    if os.path.exists(meta_path) and os.path.exists(texts_path) and os.path.exists(vecs_path) and os.path.exists(sparse_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("hash") == cur_hash and meta.get("embed_model") == EMBED_MODEL:
                need_rebuild = False
        except Exception:
            need_rebuild = True

    if need_rebuild:
        # Embeddings nuevos
        vecs = embed_texts(corpus_texts)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(corpus_texts, f, ensure_ascii=False)
        np.save(vecs_path, vecs.astype(np.float32, copy=False))

        # Sparse nuevo
        si = build_sparse_index(corpus_texts)
        dumpable = dump_sparse_index(si)
        with open(sparse_path, "w", encoding="utf-8") as f:
            json.dump(dumpable, f, ensure_ascii=False)

        # Meta
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"hash": cur_hash, "embed_model": EMBED_MODEL}, f)

    # Carga con mmap (menos RAM)
    with open(texts_path, "r", encoding="utf-8") as f:
        corpus_texts = json.load(f)
    corpus_vecs = np.load(vecs_path, mmap_mode="r")  # <= mmap

    with open(sparse_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    sparse_index = load_sparse_index(loaded)

    return corpus_texts, corpus_vecs, sparse_index
