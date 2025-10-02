# rag/embed.py
import os, subprocess, shutil, time
import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"  # o "mxbai-embed-large"

def _ensure_ollama_running():
    try:
        ollama.list()
    except Exception:
        cand = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
        if not os.path.exists(cand):
            raise SystemExit("Ollama CLI no encontrado. Instálalo con Homebrew o el .pkg.")
        subprocess.Popen([cand, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for _ in range(20):
            time.sleep(0.5)
            try:
                ollama.list()
                break
            except Exception:
                pass
        else:
            raise SystemExit("No pude conectar con el daemon de Ollama. Ejecuta 'ollama serve' en otra terminal.")

def _ensure_model(model_name: str):
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if model_name not in out:
        print(f"\nDescargando {model_name}...")
        r = subprocess.run(["ollama", "pull", model_name])
        if r.returncode != 0:
            raise SystemExit(f"Fallo el pull de {model_name}. Verifica el daemon de Ollama.")

def embed_texts(texts):
    """
    Devuelve array (n,d) de embeddings normalizados.
    Compatible con APIs viejas (prompt/embedding) y nuevas (input/embeddings).
    """
    _ensure_ollama_running()
    _ensure_model(EMBED_MODEL)

    if isinstance(texts, str):
        texts = [texts]

    vecs = []
    for t in texts:
        try:
            out = ollama.embeddings(model=EMBED_MODEL, input=t)  # API nueva
            v = out.get("embedding") or out["embeddings"][0]
        except TypeError:
            out = ollama.embeddings(model=EMBED_MODEL, prompt=t)  # API vieja
            v = out["embedding"]
        vecs.append(v)

    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms

def cosine_top_k(query, corpus_texts, corpus_vecs, k=16):
    qv = embed_texts([query])[0]
    sims = corpus_vecs @ qv
    idx = sims.argsort()[-k:][::-1]
    return [corpus_texts[i] for i in idx], sims[idx]

def ensure_rag_index(training_messages, texts_path="rag_texts.json", vecs_path="rag_vecs.npy"):
    """
    Crea/carga el índice RAG (textos + embeddings) desde disco.
    Retorna (corpus_texts, corpus_vecs).
    """
    import json
    if os.path.exists(texts_path) and os.path.exists(vecs_path):
        with open(texts_path, "r", encoding="utf-8") as f:
            corpus_texts = json.load(f)
        corpus_vecs = np.load(vecs_path)
    else:
        corpus_texts = [m["message"] for m in training_messages if m.get("message")]
        corpus_vecs  = embed_texts(corpus_texts)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(corpus_texts, f, ensure_ascii=False)
        np.save(vecs_path, corpus_vecs)
    return corpus_texts, corpus_vecs
