# Chatbot de Personalidad (Local) — README

> Chatbot local que **imita la personalidad** de una persona seleccionada a partir de sus conversaciones y **documentos multimedia**. Usa un **modelo de lenguaje pre‑entrenado** (vía Ollama), **aprendizaje de estilo** con *few‑shots* y **RAG híbrido** (denso + léxico) para razonar sobre **.txt, .jpg y .mp4** (texto directo, OCR y transcripción/resumen de video).

---

## Cómo funciona (muy breve)
1) **Ingesta y “learning” del estilo**: se lee un histórico (p. ej., export de WhatsApp) y se extraen pares *pregunta → respuesta* de la persona elegida para construir *few‑shots* que fijan tono y emojis.
2) **Indexación RAG**: se embeben textos y se arma un índice híbrido (denso + léxico) con MMR y re‑rank opcional.
3) **Chat**: en cada turno, se recupera contexto relevante (incluido lo derivado de imágenes y videos) y se responde **en el estilo** de la persona.

---

## Requisitos
- **Python 3.10+**
- **Ollama** en local (`ollama serve`)
- Modelos por defecto:
  - Generación: `mistral` (puedes escoger `llama3.2`, `llama3.2:1b`, `gemma2:2b`, etc.)
  - Embeddings: `nomic-embed-text`
  - Visión (opcional, para .jpg/.mp4): `llama3.2-vision`  
  Sugerido:  
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ollama pull llama3.2-vision   # opcional
  ```
- Librerías Python:
  ```bash
  pip install -U ollama numpy pillow pytesseract
  ```
- **Sistema** (para multimedia, opcional): `tesseract-ocr` y `ffmpeg` en PATH.

---

## Estructura del proyecto
```
chatbot/
├─ main.py
├─ _chat.txt                          # export de WhatsApp (ejemplo)
├─ conversacion_completa.json         # dataset completo (salida)
├─ mensajes_procesados.json           # dataset limpio p/ entrenamiento estilo (salida)
├─ rag_texts.json | rag_vecs.npy | rag_sparse.json | rag_meta.json   # índice RAG persistente
├─ bot/
│  ├─ local_chatbot.py
│  ├─ config.py
│  ├─ prompts.py
│  ├─ utils/ {training.py, history.py, text.py, patterns.py}
│  └─ services/ {ollama_service.py, rag_context.py, fewshot_builder.py, postprocess.py, mode_detector.py}
├─ chatio/
│  └─ whatsapp.py
├─ rag/ {embeddings.py, index_io.py, retrieve.py, hybrid.py, mmr.py, multiquery.py, re_rank.py, sparse_index.py, embed.py}
└─ media/
   └─ ingest.py                       # OCR/caption/transcripción de .jpg/.mp4
```

---

## 1) ¿Qué carpeta actúa primero respecto a `main.py`?
1. **`chatio/`** → `whatsapp.py` se usa **primero** desde `main.py` para **cargar y normalizar el histórico** (export `.txt`) y **producir** `mensajes_procesados.json` y `conversacion_completa.json`.
2. **`bot/`** → `local_chatbot.py` inicializa el motor del chat (**configura el modelo**, arma *few‑shots* de estilo, filtros y opciones de generación).
3. **`rag/`** → `index_io.py` construye o carga el **índice RAG** persistente; `retrieve.py` ejecuta la recuperación híbrida por turno.
4. **`media/`** (opcional) → `ingest.py` convierte **.jpg/.mp4** en texto (OCR, *captions*, transcripción de audio) y **añade ese texto** al corpus que usa RAG.

> Resumen del flujo en `main.py`: **parseo de WhatsApp (`chatio/`) → inicialización del chatbot (`bot/`) → preparación/carga del índice (`rag/`) → bucle de chat**.

---

## 2) ¿Qué hace cada archivo? (descripciones cortas)

### Raíz
- **`main.py`** — CLI. Orquesta: parseo de WhatsApp → creación de `LocalChatbot` → bucle de conversación.
- **`_chat.txt`** — Export estándar de WhatsApp (entrada).
- **`conversacion_completa.json`** — Conversación completa parseada (salida).
- **`mensajes_procesados.json`** — Mensajes limpios y balanceados para *few‑shots* (salida).
- **`rag_texts.json` / `rag_vecs.npy` / `rag_sparse.json` / `rag_meta.json`** — **Índice RAG** (textos, embeddings, índice léxico y metadatos).

### `chatio/` (I/O de conversaciones)
- **`whatsapp.py`** — Lee exports (`_chat.txt` o `.zip`), detecta participantes, crea datasets y estadísticas básicas.

### `rag/` (recuperación aumentada)
- **`embeddings.py`** — Embebe textos con `nomic-embed-text` (normaliza y cachea).
- **`index_io.py`** — Construye/carga índice persistente (textos, vectores, índice disperso).
- **`retrieve.py`** — Recuperación **robusta**: multi‑query, híbrido, MMR y *re‑rank* (opcional).
- **`hybrid.py`** — Combina similitud densa y léxica con normalización *min‑max*.
- **`mmr.py`** — *Maximal Marginal Relevance* para diversidad.
- **`multiquery.py`** — Amplía la consulta con variantes (si el texto del usuario es largo).
- **`re_rank.py`** — Segunda pasada de calidad con LLM (si se habilita).
- **`sparse_index.py`** — Índice léxico simplificado (tf‑idf/bolsa de palabras).
- **`embed.py`** — *Shim* de importaciones (reexporta funciones clave de RAG).

### `bot/` (núcleo del chat)
- **`local_chatbot.py`** — Clase principal. Construye *few‑shots*, decide modo (corto/largo), llama a RAG y genera la respuesta.
- **`config.py`** — Parámetros globales (modelos, opciones de generación, umbrales RAG, visión y estilo).
- **`prompts.py`** — `system prompt` y plantilla de instrucciones para preservar **personalidad**.
- **`utils/training.py`** — Carga mensajes, detecta nombre/persona y emojis dominantes.
- **`utils/history.py`** — Historial de la conversación y guardado de sesiones.
- **`utils/text.py`** — Normalización, *tokenizer* simple, utilidades de texto.
- **`services/ollama_service.py`** — Cliente a Ollama (asegura daemon/modelos; `chat` y `embed`).
- **`services/rag_context.py`** — Arma el contexto de RAG (híbrido denso+léxico, *gating*, MMR).
- **`services/fewshot_builder.py`** — Extrae pares *usuario→respuesta* para *few‑shots*.
- **`services/postprocess.py`** — Limpieza de salida (quita frases fáticas, etiquetas, *slang*).
- **`services/mode_detector.py`** — Heurísticas para responder corto/largo y activar cálculos.

### `media/` (ingesta de .jpg/.mp4 — opcional)
- **`ingest.py`** — Convierte imágenes y videos en texto utilizable por RAG:  
  - **Imágenes (`.jpg`)**: OCR con `pytesseract` o *caption* con modelo de visión.  
  - **Videos (`.mp4`)**: *key‑frames* con `ffmpeg`, OCR/caption por frame y resumen automático.

---

## Uso rápido

1. **Modelos** (una sola vez):
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ollama pull llama3.2-vision   # opcional, para imágenes/videos
   ```
2. **(Opcional) Ingesta de multimedia** en `chatbot/media`:
   ```bash
   # coloca .jpg/.mp4 en ./media y genera un índice de texto de apoyo
   python -c "from media.ingest import ingest_media_dir, save_media_index; docs=ingest_media_dir('media','llama3.2-vision','mistral'); save_media_index(docs,'rag_media.json')"
   ```
3. **Coloca tu export de WhatsApp** en `chatbot/_chat.txt`.
4. **Ejecuta**:
   ```bash
   cd chatbot
   python main.py
   ```
5. **Elige la persona** (participante detectado) y conversa.  
   El bot responderá **con su estilo** y apoyado por **RAG** sobre tus textos y lo derivado de **.jpg/.mp4**.

---

## Notas
- El proyecto **no hace fine‑tuning** del modelo; aprende **estilo** con *few‑shots* y usa **RAG** para conocimiento factual.
- Los archivos `__pycache__/`, `.DS_Store` y `__MACOSX/` se pueden ignorar.
- Puedes cambiar el modelo por defecto en `bot/config.py` o al iniciar el CLI.
