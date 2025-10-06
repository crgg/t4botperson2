# Paso 3: generación de embeddings densos
# Crea vectores normalizados (float32) para textos usando Ollama.
# - EMBED_MODEL: nombre del modelo de embeddings por defecto (reexportado).
# - embed_texts(texts) -> np.ndarray (n, d): maneja API nueva (input=) y vieja (prompt=).
# - cosine_top_k(query, corpus_texts, corpus_vecs): utilitario simple de similitud.
# - Depende de ollama_utils para asegurar daemon y modelos.
# - Los vectores se devuelven ya normalizados (norma L2 = 1).

from functools import lru_cache
from typing import List
import numpy as np

from .ollama_utils import ensure_ollama_running, ensure_model, embeddings_call

# Constante exportada
EMBED_MODEL = "nomic-embed-text"  # (alternativas: "mxbai-embed-large")

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / n

@lru_cache(maxsize=256)
def _embed_one_cached(text: str) -> np.ndarray:
    return np.asarray(embeddings_call(EMBED_MODEL, text), dtype=np.float32)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Devuelve array (n,d) de embeddings normalizados.
    Usa caché LRU para strings repetidos (p.ej., queries frecuentes).
    """
    ensure_ollama_running()
    ensure_model(EMBED_MODEL)

    if isinstance(texts, str):
        texts = [texts]

    vecs = [_embed_one_cached(t) for t in texts]
    arr = np.vstack(vecs).astype(np.float32, copy=False)
    return _l2norm(arr)
