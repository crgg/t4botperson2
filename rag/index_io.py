# Paso 9: construcción/carga del índice persistente
# Punto de entrada para preparar el corpus antes de consultar:
# - ensure_rag_index(training_messages, paths...) -> (corpus_texts, corpus_vecs, sparse_index)
#   * Extrae textos, calcula embeddings (embeddings.py) y arma índice léxico (sparse_index.py).
#   * Guarda en disco: textos (JSON), vectores (NPY) e índice disperso (JSON).
#   * Usa hash del corpus + nombre de modelo para evitar recomputar si nada cambió.
#   * Carga vectores con mmap (np.load(..., mmap_mode="r")) para ahorrar RAM.
# - Se invoca una vez al iniciar el chatbot/local server, no por turno.

import os, json, hashlib
from typing import List, Tuple
import numpy as np

from .embeddings import embed_texts, EMBED_MODEL
from .sparse_index import build_sparse_index, dump_sparse_index, load_sparse_index

def _hash_corpus(texts):
    h = hashlib.sha1()
    for t in texts:
        h.update((t or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()

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
