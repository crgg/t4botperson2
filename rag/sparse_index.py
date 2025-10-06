# Paso 4: índice disperso (léxico) e IDF
# Construye y serializa un índice lexical ligero para señales por términos.
# - build_sparse_index(corpus_texts) -> dict con {"idf", "docs", "N"}.
# - dump_sparse_index(idx) / load_sparse_index(jsonable): ida y vuelta JSON-friendly.
# - Tokenización simple en minúsculas; IDF suavizado para estabilidad.
# - Complementa a los embeddings con una señal no semántica pero precisa.

import re, math, json
from collections import Counter, defaultdict
from typing import List, Dict, Set
import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)
def tok(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

def build_sparse_index(corpus_texts: List[str]) -> Dict:
    """
    Retorna dict con:
      - idf: dict(term->idf)
      - docs: List[Set[str]] términos por doc
      - inv: dict(term->List[int]) índice invertido
      - N: total documentos
    """
    N = len(corpus_texts)
    docs_terms: List[Set[str]] = []
    df = Counter()
    inv = defaultdict(list)

    for i, txt in enumerate(corpus_texts):
        terms = set(tok(txt))
        docs_terms.append(terms)
        for t in terms:
            df[t] += 1
            inv[t].append(i)

    idf = {t: math.log(1.0 + (N / (1.0 + dfc))) for t, dfc in df.items()}
    return {"idf": idf, "docs": docs_terms, "inv": dict(inv), "N": N}

def sparse_scores(query: str, sparse_index: Dict, N: int) -> np.ndarray:
    """
    Calcula score léxico SOLO en docs que contienen términos del query (via índice invertido).
    Devuelve vector (N,) con ceros en el resto.
    """
    q_terms = set(tok(query))
    idf = sparse_index["idf"]
    inv = sparse_index["inv"]
    scores = np.zeros(N, dtype=np.float32)

    candidates = set()
    for t in q_terms:
        for i in inv.get(t, []):
            candidates.add(i)

    for i in candidates:
        inter = q_terms & sparse_index["docs"][i]
        if inter:
            scores[i] = sum(idf.get(t, 0.0) for t in inter)
    return scores

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def dump_sparse_index(si: Dict) -> Dict:
    # serializable (sets->lists)
    return {
        "idf": si["idf"],
        "N": si["N"],
        "docs": [list(s) for s in si["docs"]],
        "inv": {t: ids for t, ids in si["inv"].items()},
    }

def load_sparse_index(d: Dict) -> Dict:
    return {
        "idf": d.get("idf", {}),
        "N": d.get("N", 0),
        "docs": [set(x) for x in d.get("docs", [])],
        "inv": d.get("inv", {}),
    }
