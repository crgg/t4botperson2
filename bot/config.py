# bot/config.py
from dataclasses import dataclass, field
from typing import Dict, List
from rag.embed import EMBED_MODEL  # ya lo usas en tu proyecto

@dataclass
class GenerationOptions:
    """
    Parámetros del generador por modo. Ajusta estos knobs para performance/estilo.
    """
    short: Dict = field(default_factory=lambda: {
        "seed": 7,                 # Semilla de aleatoriedad. Mismo seed ⇒ respuestas más reproducibles.
        "temperature": 0.12,       # Aleatoriedad global. Más bajo = más determinista (ideal para WhatsApp corto).
        "top_p": 0.8,              # Nucleus sampling: recorta a los tokens que acumulan el 80% de prob.
        "repeat_penalty": 1.25,    # Penaliza repeticiones (eco). Más alto = menos repetición.
        "repeat_last_n": 128,      # Ventana (en tokens) sobre la que se calcula la penalización por repetición.
        "num_predict": 96,         # Máximo de tokens de salida. Mantiene respuestas breves.
        "num_ctx": 4096,           # Tamaño del contexto (entrada + historial + RAG + salida). Debe caber todo.
        "stop": [                  # Secuencias que, si aparecen, DETIENEN la generación (corta la respuesta).
            "Usuario:", "User:", "EJEMPLOS", "Ejemplos",
            "En cuanto a las conversaciones",
            "Posibles respuestas", "Possible responses",
            "¿Cómo estás", "como estas", "¿cómo estás", "Como estas",
            "Disculpa por", "perdón por", "Perdón por", "Espero que estés bien",
        ],
    })

    long: Dict = field(default_factory=lambda: {
        "seed": 7,                 # Igual que arriba: reproducibilidad.
        "temperature": 0.15,       # Un poco más libre que en short para explicaciones más naturales.
        "top_p": 0.85,             # Nucleus sampling algo más amplio (más variedad que en short).
        "repeat_penalty": 1.10,    # Menor penalización para no “cortar” fluidez en respuestas largas.
        "repeat_last_n": 128,      # Misma ventana de memoria para controlar repeticiones.
        "num_predict": 480,        # Límite mayor de tokens de salida (explicaciones tipo ChatGPT).
        "num_ctx": 4096,           # Contexto total disponible para el modelo.
        "stop": [                  # Mismas reglas de corte para evitar frases de call-center/saludos no pedidos.
            "Usuario:", "User:", "EJEMPLOS", "Ejemplos",
            "¿Cómo estás", "como estas", "¿cómo estás", "Como estas",
            "Disculpa por", "perdón por", "Perdón por", "Espero que estés bien",
        ],
    })

@dataclass
class RAGConfig:
    """
    Umbrales y parámetros de recuperación (impactan precisión/ruido).
    """
    alpha: float = 0.55               # peso denso vs. léxico (retrieve_robust)
    fetch_k: int = 32                  # candidatos para MMR
    use_mmr: bool = True
    gating_dense_sim_threshold: float = 0.68
    lexical_signal_threshold: float = 1.60
    multiquery_min_chars: int = 120     # usar multiquery si la consulta es más larga que esto
    top_emojis_k: int = 8              # para estilo

@dataclass
class StyleConfig:
    """
    Estilo/conducta del chat y filtros.
    """
    default_mode: str = "long"         # "short" | "long"
    max_short_chars: int = 180
    banned_phrases: List[str] = field(default_factory=lambda: [
        "¿Cómo puedo ayudarte hoy?",
        "estoy aquí para ayudarte",
        "aquí hay algunos ejemplos",
    ])
    banned_slang: List[str] = field(default_factory=lambda: [
        # MX / ES / AR (no deseados para es-CL)
        "chido", "wey", "órale", "vale", "tío", "che", "pibe",
    ])

@dataclass
class ModelConfig:
    """
    Modelos por defecto. Puedes cambiarlos centralizadamente.
    """
    model_name: str = "mistral"
    embed_model: str = EMBED_MODEL

@dataclass
class ChatbotConfig:
    """
    Config master que agrupa todo.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    gen: GenerationOptions = field(default_factory=GenerationOptions)
    rag: RAGConfig = field(default_factory=RAGConfig)
    style: StyleConfig = field(default_factory=StyleConfig)

# Helper para construir la config con overrides simples
def build_config(
    model_name: str | None = None,
    overrides: Dict | None = None
) -> ChatbotConfig:
    cfg = ChatbotConfig()
    if model_name:
        cfg.model.model_name = model_name
    # overrides opcionales por llave anidada, ej:
    # {"gen": {"long": {"temperature": 0.1}}, "rag": {"alpha": 0.7}}
    if overrides:
        for section, sub in overrides.items():
            node = getattr(cfg, section)
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if isinstance(getattr(node, k), dict) and isinstance(v, dict):
                        getattr(node, k).update(v)
                    else:
                        setattr(node, k, v)
    return cfg
