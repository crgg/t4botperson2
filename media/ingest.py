# media/ingest.py
from __future__ import annotations
import os, re, json, tempfile, subprocess, uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

try:
  from PIL import Image
except Exception:
  Image = None

try:
  import pytesseract
except Exception:
  pytesseract = None

import ollama  # usamos directamente para captions si hay modelo de visión

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXT = {".mp4", ".mov", ".mkv", ".avi"}

@dataclass
class MediaDoc:
  id: str
  kind: str               # 'image' | 'video'
  path: str               # archivo original
  text: str               # texto final para RAG (caption/ocr/sumario)
  images: List[str]       # paths de imágenes relevantes (la misma imagen o keyframes)
  meta: Dict[str, Any]

def _has_cmd(name: str) -> bool:
  from shutil import which
  return which(name) is not None

def _caption_with_vision(img_path: str, vision_model: Optional[str]) -> Optional[str]:
  if not vision_model:
    return None
  try:
    resp = ollama.chat(
      model=vision_model,
      messages=[{
        "role": "user",
        "content": "Describe la imagen en una sola frase, estilo telegráfico, incluye personas/objetos/entorno.",
        "images": [img_path]
      }],
      options={"num_ctx": 4096, "temperature": 0.2}
    )
    return (resp["message"]["content"] or "").strip()
  except Exception:
    return None

def _ocr_image(img_path: str) -> Optional[str]:
  if pytesseract is None or Image is None:
    return None
  try:
    im = Image.open(img_path)
    txt = pytesseract.image_to_string(im, lang="spa+eng")
    txt = re.sub(r"\s+", " ", (txt or "").strip())
    return txt if txt else None
  except Exception:
    return None

def _sample_video_frames(video_path: str, out_dir: str, every_sec: int = 8, max_frames: int = 6) -> List[str]:
  if not _has_cmd("ffmpeg"):
    return []
  # dump frames cada N segundos hasta max_frames
  out = []
  # -vf fps=1/every_sec para samplear cada 'every_sec'
  cmd = [
    "ffmpeg", "-v", "error", "-i", video_path,
    "-vf", f"fps=1/{max_frames if every_sec<=0 else every_sec}",
    "-qscale:v", "2",
    os.path.join(out_dir, "kf_%03d.jpg")
  ]
  try:
    subprocess.run(cmd, check=True)
  except Exception:
    return []
  for f in sorted(os.listdir(out_dir)):
    if f.lower().endswith(".jpg") and f.startswith("kf_"):
      out.append(os.path.join(out_dir, f))
  # limita a max_frames
  return out[:max_frames]

def _transcribe_audio(video_path: str) -> Optional[str]:
  """
  Intenta transcribir con whisper CLI ('whisper archivo.mp4 --language Spanish').
  Si no está, intenta vía Python si el paquete 'whisper' está disponible.
  """
  # vía CLI
  if _has_cmd("whisper"):
    try:
      subprocess.run(["whisper", video_path, "--language", "Spanish", "--fp16", "False", "--model", "base"], check=True)
      # whisper genera archivo .txt junto al video
      base = os.path.splitext(video_path)[0]
      txt_path = base + ".txt"
      if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
          return re.sub(r"\s+", " ", f.read().strip())
    except Exception:
      pass
  # vía librería
  try:
    import whisper as w
    model = w.load_model("base")
    result = model.transcribe(video_path, language="es")
    t = result.get("text", "").strip()
    return re.sub(r"\s+", " ", t) if t else None
  except Exception:
    return None

def _summarize_text(text: str, sum_model: str) -> str:
  """
  Resumen corto con tu modelo de texto (Ollama).
  """
  if not text:
    return ""
  prompt = f"Resume en 5 viñetas max lo siguiente, estilo telegráfico y factual:\n\n{text[:7000]}"
  try:
    r = ollama.chat(model=sum_model, messages=[
      {"role": "user", "content": prompt}
    ], options={"temperature": 0.3, "num_ctx": 4096, "num_predict": 220})
    return (r["message"]["content"] or "").strip()
  except Exception:
    return text[:700]

def _ingest_image(path: str, vision_model: Optional[str]) -> MediaDoc:
  cap = _caption_with_vision(path, vision_model)
  ocr = _ocr_image(path)
  pieces = [p for p in [cap, ocr] if p]
  text = " | ".join(pieces) if pieces else "(imagen sin texto/caption)"
  return MediaDoc(
    id=str(uuid.uuid4()),
    kind="image",
    path=path,
    text=text,
    images=[path],
    meta={"caption_model": vision_model if cap else None, "has_ocr": bool(ocr)}
  )

def _ingest_video(path: str, vision_model: Optional[str], every_sec: int, max_frames: int, sum_model: str) -> MediaDoc:
  with tempfile.TemporaryDirectory() as td:
    frames = _sample_video_frames(path, td, every_sec=every_sec, max_frames=max_frames)
    captions = []
    for f in frames[:max_frames]:
      cap = _caption_with_vision(f, vision_model)
      if cap:
        captions.append(cap)
    transcript = _transcribe_audio(path)
    summary = _summarize_text(transcript or " ", sum_model) if transcript else ""
    text_parts = []
    if transcript: text_parts.append(f"[transcripción] {transcript}")
    if captions:   text_parts.append(f"[frames] {' | '.join(captions)}")
    if summary:    text_parts.append(f"[resumen] {summary}")
    text = " || ".join(text_parts) if text_parts else "(video sin señales textuales)"
    # guarda frames si quieres reutilizarlos fuera de tmp: opcional — aquí los perdemos al salir
  return MediaDoc(
    id=str(uuid.uuid4()),
    kind="video",
    path=path,
    text=text,
    images=[],  # si quieres, vuelve a extraer frames persistentes a un directorio fijo y ponlos acá
    meta={"frames_sampled": max_frames, "caption_model": vision_model, "has_transcript": bool(transcript)}
  )

def ingest_media_dir(
    media_dir: str,
    vision_model: Optional[str],
    sum_model: str,
    every_sec: int = 8,
    max_frames: int = 6,
) -> List[MediaDoc]:
  docs: List[MediaDoc] = []
  if not os.path.isdir(media_dir):
    return docs
  for root, _, files in os.walk(media_dir):
    for fn in files:
      ext = os.path.splitext(fn)[1].lower()
      p = os.path.join(root, fn)
      try:
        if ext in IMG_EXT:
          docs.append(_ingest_image(p, vision_model))
        elif ext in VID_EXT:
          docs.append(_ingest_video(p, vision_model, every_sec, max_frames, sum_model))
      except Exception as e:
        print(f"[media.ingest] Error con {p}: {e}")
  return docs

def save_media_index(docs: List[MediaDoc], out_path: str):
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump([asdict(d) for d in docs], f, ensure_ascii=False, indent=2)
