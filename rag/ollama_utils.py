# Paso 2: utilidades de infraestructura (Ollama)
# Garantiza que el daemon de Ollama esté corriendo y que los modelos existan.
# - ensure_ollama_running(): levanta 'ollama serve' si no está activo.
# - ensure_model(model_name): hace 'ollama pull' si falta un modelo.
# - Centraliza interacción con el runtime; evita duplicar este código en otros módulos.
# - Se usa indirectamente por embeddings y por funciones que invocan LLM.


import os, time, shutil, subprocess
import ollama

def ensure_ollama_running():
    try:
        ollama.list()
        return
    except Exception:
        cand = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
        if not os.path.exists(cand):
            raise SystemExit("Ollama CLI no encontrado. Instálalo con Homebrew o el .pkg.")
        subprocess.Popen([cand, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for _ in range(20):
            time.sleep(0.5)
            try:
                ollama.list()
                return
            except Exception:
                pass
        raise SystemExit("No pude conectar con el daemon de Ollama. Ejecuta 'ollama serve' en otra terminal.")

def ensure_model(model_name: str):
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if model_name not in out:
        print(f"\nDescargando {model_name}...")
        r = subprocess.run(["ollama", "pull", model_name])
        if r.returncode != 0:
            raise SystemExit(f"Fallo el pull de {model_name}. Verifica el daemon de Ollama.")

def embeddings_call(model: str, text: str):
    """Compat con API nueva/vieja de ollama.embeddings()."""
    try:
        out = ollama.embeddings(model=model, input=text)  # API nueva
        v = out.get("embedding") or out["embeddings"][0]
    except TypeError:
        out = ollama.embeddings(model=model, prompt=text)  # API vieja
        v = out["embedding"]
    return v
