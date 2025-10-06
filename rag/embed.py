# rag/embed.py  (shim fino)
from .embeddings import EMBED_MODEL
from .retrieve import ensure_rag_index, retrieve_robust

__all__ = ["EMBED_MODEL", "ensure_rag_index", "retrieve_robust"]
