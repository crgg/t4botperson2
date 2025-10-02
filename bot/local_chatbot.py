# bot/local_chatbot.py
import os, subprocess, shutil, time, json
from typing import List, Dict
import ollama

from rag.embed import cosine_top_k, ensure_rag_index, EMBED_MODEL

class LocalChatbot:
    def __init__(self, messages_file: str, model_name: str = "mistral"):
        self.model_name = model_name
        self.messages_file = messages_file
        self.conversation_history: List[Dict] = []

        with open(messages_file, 'r', encoding='utf-8') as f:
            self.training_messages = json.load(f)

        self._check_ollama_and_models()
        self.corpus_texts, self.corpus_vecs = ensure_rag_index(self.training_messages)
        self.system_prompt = self._create_system_prompt()

    def _check_ollama_and_models(self):
        # 1) daemon corriendo
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
        # 2) modelos (generación + embeddings)
        for m in (self.model_name, EMBED_MODEL):
            out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
            if m not in out:
                print(f"\nDescargando {m}...")
                r = subprocess.run(["ollama", "pull", m])
                if r.returncode != 0:
                    raise SystemExit(f"Fallo el pull de {m}. Revisa el daemon.")

        print(f"✓ Ollama y modelos listos: {self.model_name} + {EMBED_MODEL}")

    def _create_system_prompt(self) -> str:
        sample_size = min(200, len(self.training_messages))
        sample = self.training_messages[:sample_size]
        messages_text = "\n".join([f"- {m['message']}" for m in sample])

        prompt = f"""Eres una recreación de una persona basada en sus conversaciones reales de WhatsApp.
Tu objetivo es responder exactamente como esta persona lo haría.

EJEMPLOS DE CÓMO SE COMUNICABA:
{messages_text}

INSTRUCCIONES:
- Usa el mismo tono, slang y emojis que en los ejemplos (tuteo chileno cuando corresponda).
- Mantén respuestas concisas, estilo WhatsApp.
- Si no está en el CONTEXTO PRIVADO, di con honestidad que no lo recuerdas.
- No menciones que eres un modelo o una IA.

Responde como si fueras esta persona en WhatsApp."""
        return prompt

    def chat(self, user_message: str) -> str:
        # RAG: busca contexto parecido a la consulta
        top_texts, sims = cosine_top_k(user_message, self.corpus_texts, self.corpus_vecs, k=12)
        filtered = [t for t, s in zip(top_texts, sims) if s > 0.15] or self.corpus_texts[:3]
        contexto = "CONTEXTO PRIVADO (no lo cites literal):\n" + "\n".join(f"- {t}" for t in filtered) + "\n\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history[-20:],
            {"role": "user", "content": f"{contexto}Usuario: {user_message}"}
        ]

        resp = ollama.chat(
            model=self.model_name,
            messages=messages,
            options={
                "seed": 7, #fija que la salida sea la misma si el input es el mismo
                "temperature": 0.3,    # sube/baja para más/menos “chispa”, mientras mas bajo mas pegado al estilo de la persona
                "top_p": 0.9, #en este caso estamos diciendo que tome el 90% del sample para "copiar" estilo de escritura
                "num_predict": -1,    # longitud aprox. de respuesta, en este caso -1 significa sin limite de longitud
                "num_ctx": 4096 #ram
            }
        )
        out = resp["message"]["content"]
        self.conversation_history += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": out},
        ]
        return out

    def save_conversation(self, filename: str = "conversacion_guardada.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"✓ Conversación guardada en {filename}")
