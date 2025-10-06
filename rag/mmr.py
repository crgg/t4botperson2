# Paso 6: reranking por diversidad (MMR)
# Aplica Maximal Marginal Relevance sobre candidatos para evitar duplicados.
# - mmr_rerank(query_vec, candidate_idx, doc_vecs, lambda_mmr, k) -> lista de índices
# - Equilibrio: similitud con la consulta vs. novedad respecto a lo ya elegido.
# - Útil cuando varios fragmentos dicen “lo mismo” de forma muy parecida.

from typing import List
import numpy as np

def mmr_rerank(query_vec: np.ndarray,
               candidate_idx: List[int],
               doc_vecs: np.ndarray,
               lambda_mmr: float = 0.5,
               k: int = 8):
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
