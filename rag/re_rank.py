# Paso 8: re-ranking con LLM (segunda pasada, opcional)
# Pide al LLM que elija los fragmentos más útiles entre candidatos ya filtrados.
# - llm_rerank(query, candidates, model, topn) -> subset ordenado
# - Devuelve índices parseados de una respuesta tipo "[0,2,3]"; si falla, deja top-n.
# - Es la parte más “cara”; úsala solo cuando necesites la última milla de calidad.

import re
from typing import List
import ollama

from .ollama_utils import ensure_ollama_running, ensure_model

def llm_rerank(query: str, candidates: List[str], model: str = "mistral", topn: int = 5) -> List[str]:
    try:
        ensure_ollama_running()
        ensure_model(model)
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
