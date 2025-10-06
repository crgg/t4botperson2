# Paso 7: expansión de consulta (multi-query)
# Genera variantes para la consulta con un LLM y así cubrir más formas de preguntar.
# - multi_query_expansion(query, n, model) -> [q1, q2, ...]
# - Fallback robusto: si algo falla, devuelve [query] y sigue el pipeline.
# - Se usa antes del híbrido para acumular candidatos de varias redacciones.

import re
from typing import List
import ollama

from .ollama_utils import ensure_ollama_running, ensure_model

def multi_query_expansion(query: str, n: int = 4, model: str = "mistral") -> List[str]:
    try:
        ensure_ollama_running()
        ensure_model(model)
        prompt = (
            "Parafrasea la siguiente consulta en 4 variantes concisas y distintas, una por línea, "
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
