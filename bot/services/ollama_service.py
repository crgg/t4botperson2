# services/ollama_service.py
from __future__ import annotations
import os, time, shutil, subprocess
import ollama

class OllamaService:
  def __init__(self, gen_model: str, embed_model: str):
    self.gen_model = gen_model
    self.embed_model = embed_model
    self._ensure_daemon()
    self._ensure_models()

  def _ensure_daemon(self) -> None:
    try:
      ollama.list()
      return
    except Exception:
      cand = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
      if not os.path.exists(cand):
        raise SystemExit("Ollama CLI no encontrado. Instálalo.")
      subprocess.Popen([cand, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
      for _ in range(20):
        time.sleep(0.5)
        try:
          ollama.list(); return
        except Exception: pass
      raise SystemExit("No pude conectar con 'ollama serve'.")

  def _ensure_models(self) -> None:
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    for m in (self.gen_model, self.embed_model):
      if m not in out:
        r = subprocess.run(["ollama", "pull", m])
        if r.returncode != 0:
          raise SystemExit(f"Falló 'ollama pull {m}'.")

  # --- API pública ---
  def chat(self, messages: list[dict], options: dict) -> str:
    resp = ollama.chat(model=self.gen_model, messages=messages, options=options)
    return resp["message"]["content"]

  def embed(self, text: str) -> list[float]:
    try:
      return ollama.embeddings(model=self.embed_model, input=text)["embedding"]
    except TypeError:
      return ollama.embeddings(model=self.embed_model, prompt=text)["embedding"]
