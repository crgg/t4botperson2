# Paso 1: el proceso en la carpeta rag parte de aquí
# Fachada pública del paquete.
# - Expone los símbolos principales para consumo externo, p.ej.:
#   from rag import EMBED_MODEL, ensure_rag_index, retrieve_robust
# - No contiene lógica pesada; solo reexporta desde los submódulos.
# - Ventaja: desacopla a quien usa la librería de la estructura interna.
# - Mantén aquí importaciones livianas para evitar costos al importar.

# rag/__init__.py
from .embeddings import EMBED_MODEL
from .retrieve import ensure_rag_index, retrieve_robust

__all__ = ["EMBED_MODEL", "ensure_rag_index", "retrieve_robust"]
