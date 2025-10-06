# Paso 5: puntuación híbrida (denso + léxico)
# Combina similitud de embeddings con señal lexical IDF.
# - hybrid_top_k(query, corpus_texts, corpus_vecs, sparse_index, k, alpha, fetch_k)
#   devuelve: (pre_idx, combo_scores, query_vec)
#   * Normaliza cada señal con min-max y mezcla por: alpha*denso + (1-alpha)*léxico.
#   * pre_idx: candidatos más prometedores (fetch_k) para segundas pasadas.
# - Mantiene el cómputo en NumPy; no invoca LLM (barato/rápido).

from typing import List, Tuple
import numpy as np

from .embeddings import embed_texts
from .sparse_index import sparse_scores, minmax_norm

def hybrid_top_k(query: str,
                 corpus_texts: List[str],
                 corpus_vecs: np.ndarray,
                 sparse_index: dict,
                 k: int = 8,
                 alpha: float = 0.6,
                 fetch_k: int = 32):
    """
    Mezcla similitud densa y léxica. Devuelve:
      texts_top_k, pre_idx (indices preseleccionados), combo_scores (tamaño N)
    """
    qv = embed_texts([query])[0]
    dense = corpus_vecs @ qv                     # (N,)
    sparse = sparse_scores(query, sparse_index, N=len(corpus_texts))  # (N,)

    d = minmax_norm(dense)
    s = minmax_norm(sparse)
    combo = alpha * d + (1 - alpha) * s

    pre_idx = combo.argsort()[-fetch_k:][::-1]
    texts = [corpus_texts[i] for i in pre_idx[:k]]
    return texts, pre_idx, combo
