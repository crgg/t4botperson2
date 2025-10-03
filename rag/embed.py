# rag/embed.py
# ——————————————————————————————————————————————————————————
# Índice y recuperación híbrida para RAG:
#  - Embeddings con Ollama (denso)
#  - Índice disperso (IDF) lexical
#  - Hybrid scoring (mezcla denso+disperso)
#  - MMR (diversidad)
#  - Multi-query (paráfrasis) opcional
#  - Re-ranking con LLM (2ª pasada) opcional
#  - Persistencia de textos, vectores y sparse index
# ——————————————————————————————————————————————————————————

import os
import re
import json
import math
import time
import shutil
import subprocess
from collections import Counter, defaultdict  # defaultdict por si lo necesitas
from typing import List, Tuple

import numpy as np
import ollama

# Modelo de embeddings (Ollama)
EMBED_MODEL = "nomic-embed-text"  # alternativas: "mxbai-embed-large"

# ---------------------
# Infra Ollama
# ---------------------
def _ensure_ollama_running():
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

def _ensure_model(model_name: str):
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if model_name not in out:
        print(f"\nDescargando {model_name}...")
        r = subprocess.run(["ollama", "pull", model_name])
        if r.returncode != 0:
            raise SystemExit(f"Fallo el pull de {model_name}. Verifica el daemon de Ollama.")

# ---------------------
# Embeddings
# ---------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Devuelve array (n,d) de embeddings normalizados.
    Compatible con APIs viejas (prompt/embedding) y nuevas (input/embeddings).
    """
    _ensure_ollama_running()
    _ensure_model(EMBED_MODEL)

    if isinstance(texts, str):
        texts = [texts]

    vecs = []
    for t in texts:
        try:
            out = ollama.embeddings(model=EMBED_MODEL, input=t)  # API nueva
            v = out.get("embedding") or out["embeddings"][0]
        except TypeError:
            out = ollama.embeddings(model=EMBED_MODEL, prompt=t)  # API vieja
            v = out["embedding"]
        vecs.append(v)

    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms

def cosine_top_k(query: str, corpus_texts: List[str], corpus_vecs: np.ndarray, k: int = 16):
    """
    Recuperación densa simple (compatibilidad con código existente).
    """
    qv = embed_texts([query])[0]
    sims = corpus_vecs @ qv  # (N,)
    idx = sims.argsort()[-k:][::-1]
    return [corpus_texts[i] for i in idx], sims[idx]

# ---------------------
# Tokenización simple (para índice disperso)
# ---------------------
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def _tok(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

# ---------------------
# Índice disperso (IDF)
# ---------------------
def _build_sparse_index(corpus_texts: List[str]):
    """
    Retorna:
      {
        "idf": dict(term -> idf),
        "docs": [set(terms_en_documento), ...],
        "N": cantidad_documentos
      }
    """
    N = len(corpus_texts)
    doc_terms = []
    df = Counter()
    for txt in corpus_texts:
        terms = _tok(txt)
        uniq = set(terms)
        for t in uniq:
            df[t] += 1
        doc_terms.append(uniq)
    # IDF suavizado
    idf = {t: math.log(1.0 + (N / (1.0 + dfc))) for t, dfc in df.items()}
    return {"idf": idf, "docs": doc_terms, "N": N}

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

# ---------------------
# Hybrid (denso + disperso)
# ---------------------
def hybrid_top_k(query, corpus_texts, corpus_vecs, sparse_index, k=8, alpha=0.6, fetch_k=32):
    """
    Mezcla similitud densa (embeddings) y dispersa (IDF). Devuelve:
      texts_top_k, pre_idx (indices preseleccionados), combo_scores (tamaño N)
    """
    qv = embed_texts([query])[0]
    dense = corpus_vecs @ qv  # (N,)
    sparse = _sparse_scores(query, sparse_index)  # (N,)
    # normaliza y mezcla
    d = _minmax_norm(dense)
    s = _minmax_norm(sparse)
    combo = alpha * d + (1 - alpha) * s
    # preselección amplia
    pre_idx = combo.argsort()[-fetch_k:][::-1]
    return [corpus_texts[i] for i in pre_idx[:k]], pre_idx, combo

# ---------------------
# MMR (diversidad)
# ---------------------
def mmr_rerank(query_vec: np.ndarray, candidate_idx, doc_vecs: np.ndarray, lambda_mmr=0.5, k=8):
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

# ---------------------
# Multi-query expansion (paráfrasis)
# ---------------------
def multi_query_expansion(query, n=4, model="mistral"):
    """
    Genera reformulaciones del query con el LLM (vía Ollama).
    Si falla, devuelve [query].
    """
    try:
        _ensure_ollama_running()
        _ensure_model(model)
        prompt = (
            "Parafrasea la siguiente consulta en 4 variantes concisas y distintas, una por línea, "
            "sin numeración adicional:\n\n"
            f"Consulta: {query}\n\nVariantes:"
        )
        out = ollama.generate(model=model, prompt=prompt)
        lines = [l.strip("-• ").strip() for l in out["response"].splitlines() if l.strip()]
        # toma primeras n no vacías y distintas al original
        seen = set()
        variants = []
        for l in lines:
            if l and l.lower() != query.lower() and l not in seen:
                variants.append(l)
                seen.add(l)
            if len(variants) >= n:
                break
        return variants or [query]
    except Exception:
        return [query]

# ---------------------
# Re-ranking con LLM (segunda pasada)
# ---------------------
def llm_rerank(query, candidates, model="mistral", topn=5):
    """
    Pide al LLM que escoja los candidatos más útiles para responder la consulta.
    candidates: lista de strings cortos (frases o mensajes).
    """
    try:
        _ensure_ollama_running()
        _ensure_model(model)
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        prompt = (
            "Selecciona los fragmentos más útiles para responder la consulta. "
            "Devuelve SOLO una lista de índices entre corchetes separados por comas, por ejemplo: [0,2,3]\n\n"
            f"Consulta: {query}\n\nFragmentos:\n{numbered}\n\nÍndices:"
        )
        out = ollama.generate(model=model, prompt=prompt)
        text = out["response"]
        m = re.search(r"\[(.*?)\]", text)
        if not m:
            return candidates[:topn]
        idxs = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        idxs = [i for i in idxs if 0 <= i < len(candidates)]
        if not idxs:
            return candidates[:topn]
        return [candidates[i] for i in idxs[:topn]]
    except Exception:
        return candidates[:topn]

# ---------------------
# Pipeline de recuperación robusto
# ---------------------
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
    1) (Opcional) multi-query: genera variantes y recupera con cada una.
    2) Hybrid (denso+disperso) por variante; une candidatos.
    3) (Opcional) MMR para diversidad.
    4) (Opcional) LLM re-ranking.
    Devuelve: lista de textos top-k.
    """
    # 0) aseguramos sparse_index
    if sparse_index is None:
        sparse_index = _build_sparse_index(corpus_texts)

    # 1) variantes de query
    queries = [query]
    if use_multiquery:
        try:
            variants = multi_query_expansion(query, n=3)
            queries += variants
        except Exception:
            pass

    # 2) unión de candidatos por hybrid
    combo_scores = np.zeros(len(corpus_texts), dtype=np.float32)
    seen = set()
    for q in queries:
        _, pre_idx, combo = hybrid_top_k(q, corpus_texts, corpus_vecs, sparse_index, k=k, alpha=alpha, fetch_k=fetch_k)
        # suma de scores para priorizar candidatos que aparecen en múltiples variantes
        combo_scores[pre_idx] += combo[pre_idx]
        seen.update(pre_idx.tolist() if hasattr(pre_idx, "tolist") else list(pre_idx))

    candidates = sorted(list(seen), key=lambda i: float(combo_scores[i]), reverse=True)[:max(fetch_k, k)]

    # 3) MMR para diversidad
    if use_mmr and len(candidates) > 1:
        qv = embed_texts([query])[0]
        mmr_idx = mmr_rerank(qv, candidates, corpus_vecs, lambda_mmr=0.5, k=max(k, 10))
        candidates = mmr_idx

    # 4) LLM re-ranking (caro pero eficaz)
    top_texts = [corpus_texts[i] for i in candidates[:max(k, 10)]]
    if use_llm_rerank and len(top_texts) > k:
        top_texts = llm_rerank(query, top_texts, model=rerank_model, topn=k)
    else:
        top_texts = top_texts[:k]

    return top_texts

# ---------------------
# Persistencia del índice
# ---------------------
def ensure_rag_index(
    training_messages: List[dict],
    texts_path: str = "rag_texts.json",
    vecs_path: str = "rag_vecs.npy",
    sparse_path: str = "rag_sparse.json",
) -> Tuple[List[str], np.ndarray, dict]:
    """
    Crea/carga el índice RAG desde disco.
    Retorna (corpus_texts, corpus_vecs, sparse_index).
    """
    # Textos + vectores
    if os.path.exists(texts_path) and os.path.exists(vecs_path):
        with open(texts_path, "r", encoding="utf-8") as f:
            corpus_texts = json.load(f)
        corpus_vecs = np.load(vecs_path)
    else:
        corpus_texts = [m.get("message") for m in training_messages if m.get("message")]
        corpus_vecs = embed_texts(corpus_texts)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(corpus_texts, f, ensure_ascii=False)
        np.save(vecs_path, corpus_vecs)

    # Índice disperso (serializando sets como listas)
    if os.path.exists(sparse_path):
        with open(sparse_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        sparse_index = {
            "idf": loaded.get("idf", {}),
            "N": loaded.get("N", len(corpus_texts)),
            "docs": [set(d) for d in loaded.get("docs", [])],
        }
    else:
        sparse_index = _build_sparse_index(corpus_texts)
        dumpable = {"idf": sparse_index["idf"], "N": sparse_index["N"], "docs": [list(s) for s in sparse_index["docs"]]}
        with open(sparse_path, "w", encoding="utf-8") as f:
            json.dump(dumpable, f, ensure_ascii=False)

    return corpus_texts, corpus_vecs, sparse_index
