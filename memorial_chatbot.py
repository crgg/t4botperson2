"""
Sistema de Chat Personalizado LOCAL usando Ollama
100% privado, funciona sin internet después de la instalación inicial
Requiere: pip install ollama
"""

import re
import json
import os
from datetime import datetime
from typing import List, Dict
import subprocess

import zipfile
from collections import Counter

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"   # o "mxbai-embed-large"

def embed_texts(texts):
    """
    Compatible con clientes viejos (prompt + 'embedding') y nuevos (input + 'embeddings').
    Acepta str o list[str]. Devuelve matriz (n,d) normalizada.
    """
    import numpy as np
    if isinstance(texts, str):
        texts = [texts]

    vecs = []
    for t in texts:
        try:
            # Cliente nuevo (acepta 'input', puede devolver 'embeddings')
            out = ollama.embeddings(model=EMBED_MODEL, input=t)
            v = out.get("embedding")
            if v is None:
                v = out["embeddings"][0]
        except TypeError:
            # Cliente antiguo (usa 'prompt', devuelve 'embedding')
            out = ollama.embeddings(model=EMBED_MODEL, prompt=t)
            v = out["embedding"]
        vecs.append(v)

    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms

def cosine_top_k(query, corpus_texts, corpus_vecs, k=12):
    qv = embed_texts([query])[0]
    sims = corpus_vecs @ qv  # como están normalizados, es coseno
    idx = sims.argsort()[-k:][::-1]
    return [corpus_texts[i] for i in idx], sims[idx]

def load_whatsapp_export_text(path_or_txt: str) -> str:
    """
    Lee el contenido de un export de WhatsApp:
    - Si es .zip: abre y devuelve el contenido del primer .txt dentro.
    - Si es .txt: devuelve su contenido.
    Maneja varias codificaciones comunes.
    """
    def _decode(raw: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "iso-8859-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    path = path_or_txt
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            txts = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not txts:
                raise ValueError("El .zip no contiene ningún .txt de chat.")
            # Preferir el .txt más corto o que contenga 'chat'
            txts.sort(key=lambda n: (("chat" not in n.lower()), len(n)))
            with z.open(txts[0], "r") as f:
                raw = f.read()
        return _decode(raw)
    else:
        with open(path, "rb") as f:
            raw = f.read()
        return _decode(raw)

def detect_participants(content: str) -> Counter:
    content = normalize_whatsapp_text(content)

    # Componentes reutilizables
    DATE = r'\d{1,2}/\d{1,2}/\d{2,4}'
    WS   = r'[ \t\u00A0\u202F]'                # incluye NBSP y narrow NBSP
    TIME = r'\d{1,2}:\d{2}(?::\d{2})?'         # HH:MM[:SS]
    AMPM = r'(?:' + WS + r'?[AaPp]\.?M\.?)?'   # opcional AM/PM, con o sin puntos

    patterns = [
        # Android: "DD/MM/YYYY, HH:MM[:SS][ AM/PM] - Nombre: Mensaje"
        rf'({DATE}),?{WS}+({TIME}{AMPM}){WS}*-\s([^:]+):\s(.+?)(?=\n{DATE}|$)',
        # iOS:     "[DD/MM/YYYY, HH:MM[:SS][ AM/PM]] Nombre: Mensaje"
        rf'\[({DATE}),?{WS}+({TIME}{AMPM})\]{WS}+([^:]+):\s(.+?)(?=\n\[|$)',
    ]

    names = Counter()
    for pat in patterns:
        for m in re.findall(pat, content, re.DOTALL):
            # m = (date, time, name, message)
            names[m[2].strip()] += 1
        if names:
            break
    return names

def _tokens(s: str) -> set:
    return set(re.findall(r'\w+', s.lower()))

def retrieve_relevant_snippets(all_messages: List[Dict], query: str, k: int = 12) -> List[str]:
    """Jaccard súper simple: devuelve k mensajes del corpus más cercanos a la consulta."""
    q = _tokens(query)
    scored = []
    for m in all_messages:
        t = _tokens(m['message'])
        if not t: 
            continue
        inter = len(q & t)
        if inter == 0:
            continue
        score = inter / len(q | t)  # Jaccard
        scored.append((score, m['message']))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in scored[:k]]

class WhatsAppParser:
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.messages = []

    def parse_chat(self, target_name: str, content: str = None) -> List[Dict]:
        if content is None:
            if not self.file_path:
                raise ValueError("Proporciona 'file_path' o 'content'.")
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        content = normalize_whatsapp_text(content)

        DATE = r'\d{1,2}/\d{1,2}/\d{2,4}'
        WS   = r'[ \t\u00A0\u202F]'
        TIME = r'\d{1,2}:\d{2}(?::\d{2})?'
        AMPM = r'(?:' + WS + r'?[AaPp]\.?M\.?)?'

        patterns = [
            rf'({DATE}),?{WS}+({TIME}{AMPM}){WS}*-\s([^:]+):\s(.+?)(?=\n{DATE}|$)',
            rf'\[({DATE}),?{WS}+({TIME}{AMPM})\]{WS}+([^:]+):\s(.+?)(?=\n\[|$)',
        ]

        target_messages = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                for date, time, name, message in matches:
                    name_clean = name.strip()
                    msg = message.strip()

                    # ¿es la persona objetivo?
                    if target_name.lower() in name_clean.lower():
                        skip_tokens = [
                            '<multimedia omitido>', 'imagen omitida', 'archivo adjunto', 'sticker omitido',
                            'multimedia omitted', 'image omitted', 'attached file',
                            'missed voice call', 'missed video call'
                        ]
                        if not any(x in msg.lower() for x in skip_tokens):
                            target_messages.append({
                                'date': date, 'time': time, 'name': name_clean, 'message': msg
                            })
                break

        self.messages = target_messages
        return target_messages
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de los mensajes"""
        if not self.messages:
            return {}
        
        total = len(self.messages)
        total_words = sum(len(msg['message'].split()) for msg in self.messages)
        avg_length = total_words / total if total > 0 else 0
        
        # Palabras más comunes
        word_freq = {}
        for msg in self.messages:
            words = msg['message'].lower().split()
            for word in words:
                if len(word) > 3:  # Ignorar palabras muy cortas
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_messages': total,
            'total_words': total_words,
            'avg_message_length': round(avg_length, 2),
            'top_words': top_words
        }
    
    def export_for_training(self, output_file: str):
        """Exporta mensajes en formato JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        print(f"✓ {len(self.messages)} mensajes exportados a {output_file}")


class LocalChatbot:
    def __init__(self, messages_file: str, model_name: str = "mistral"):
        self.model_name = model_name
        self.messages_file = messages_file
        self.conversation_history = []

        with open(messages_file, 'r', encoding='utf-8') as f:
            self.training_messages = json.load(f)

        self._check_ollama()
        self._ensure_rag_index()     # <-- NUEVO
        self.system_prompt = self._create_system_prompt()

    def _ensure_rag_index(self):
        # Prepara el corpus y embeddings y los persiste
        self.corpus_texts_path = "rag_texts.json"
        self.corpus_vecs_path  = "rag_vecs.npy"

        if os.path.exists(self.corpus_texts_path) and os.path.exists(self.corpus_vecs_path):
            with open(self.corpus_texts_path, "r", encoding="utf-8") as f:
                self.corpus_texts = json.load(f)
            self.corpus_vecs = np.load(self.corpus_vecs_path)
        else:
            # usa cada mensaje como “chunk” (puedes agregar fecha/autor si quieres)
            self.corpus_texts = [m["message"] for m in self.training_messages if m.get("message")]
            self.corpus_vecs  = embed_texts(self.corpus_texts)
            with open(self.corpus_texts_path, "w", encoding="utf-8") as f:
                json.dump(self.corpus_texts, f, ensure_ascii=False)
            np.save(self.corpus_vecs_path, self.corpus_vecs)
    
    def _check_ollama(self):
        import shutil, time
        # 1) localizar CLI
        cand = shutil.which("ollama")
        if not cand:
            for p in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"):
                if os.path.exists(p):
                    cand = p; break
        if not cand:
            raise SystemExit("Ollama CLI no encontrado. Instálalo con 'brew install ollama' o el pkg.")

        # 2) comprobar conexión API (daemon)
        try:
            import ollama
            ollama.list()  # ping a http://localhost:11434
        except Exception:
            # intentar levantar daemon
            subprocess.Popen([cand, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    import ollama
                    ollama.list()
                    break
                except Exception:
                    pass
            else:
                raise SystemExit("No pude conectar con el daemon de Ollama. Ejecuta 'ollama serve' en otra terminal.")

        # 3) asegurar modelo descargado
        out = subprocess.run([cand, "list"], capture_output=True, text=True).stdout
        if self.model_name not in out:
            print(f"\nDescargando {self.model_name}...")
            r = subprocess.run([cand, "pull", self.model_name])
            if r.returncode != 0:
                raise SystemExit(f"Fallo el pull de {self.model_name}. Asegura que el daemon esté arriba.")
        print(f"✓ Ollama y modelo {self.model_name} listos")
    
    def _create_system_prompt(self) -> str:
        """Crea el prompt del sistema basado en los mensajes"""
        # Tomar muestra representativa de mensajes
        sample_size = min(200, len(self.training_messages))
        sample = self.training_messages[:sample_size]
        
        messages_text = "\n".join([f"- {msg['message']}" for msg in sample])
        
        prompt = f"""Eres una recreación de una persona basada en sus conversaciones reales de WhatsApp.
Tu objetivo es responder exactamente como esta persona lo haría.

EJEMPLOS DE CÓMO SE COMUNICABA:

{messages_text}

INSTRUCCIONES:
- Mantén el mismo tono, estilo y personalidad
- Usa las mismas expresiones, emojis y manera de escribir
- Responde con la misma calidez y forma de ser
- Si la persona usaba modismos o expresiones particulares, úsalas
- Mantén la longitud de respuesta similar al promedio de sus mensajes
- Responde en el mismo idioma que usaba (español)
- NO menciones que eres una IA
- NO uses formato formal si la persona era informal
- Sé auténtico a su forma de ser

Responde como si fueras esta persona en WhatsApp."""
        
        return prompt
    
    def chat(self, user_message: str) -> str:
        try:
            import ollama

            # Recuperación por embeddings (mejor que Jaccard)
            top_texts, sims = cosine_top_k(user_message, self.corpus_texts, self.corpus_vecs, k=12)
            # (opcional) filtra muy irrelevantes: p.ej. sims > 0.15
            filtered = [t for t, s in zip(top_texts, sims) if s > 0.15]
            if not filtered:  # como fallback, usa 3 últimos mensajes del corpus
                filtered = self.corpus_texts[:3]

            contexto = "CONTEXTO PRIVADO (no lo cites literal):\n" + "\n".join(f"- {t}" for t in filtered) + "\n\n"

            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history[-20:],
                {"role": "user", "content": f"{contexto}Usuario: {user_message}"}
            ]

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "seed": 7,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 256,   # <-- sube esto; ver nota abajo
                    "num_ctx": 4096
                }
            )
            assistant_message = response['message']['content']
            self.conversation_history += [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
            return assistant_message

        except ImportError:
            return "Error: Instala ollama con: pip install ollama"
        except Exception as e:
            return f"Error: {str(e)}"      
    
    def save_conversation(self, filename: str = "conversacion_guardada.json"):
        """Guarda la conversación actual"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Conversación guardada en {filename}")

def normalize_whatsapp_text(s: str) -> str:
    # Convierte espacios “raros” a espacio normal y unifica saltos de línea
    s = s.replace('\u202F', ' ').replace('\u00A0', ' ')
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    return s

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  Sistema de Chat Personalizado LOCAL            ║")
    print("║  100% Privado - Sin Internet                    ║")
    print("╚══════════════════════════════════════════════════╝\n")

    print("PASO 1: Procesando conversaciones de WhatsApp...\n")

    # >>> AQUI: usa tu ZIP directamente <<<
    whatsapp_file = "_chat.txt"
    messages_file = "mensajes_procesados.json"

    # Cargar contenido desde zip/txt
    try:
        raw_text = load_whatsapp_export_text(whatsapp_file)
    except Exception as e:
        print(f"❌ No se pudo leer el export de WhatsApp: {e}")
        return

    # Sugerir participantes detectados
    names = detect_participants(raw_text)
    if not names:
        print("⚠ No se detectaron remitentes. ¿Es el archivo correcto?")
        return

    print("\nParticipantes detectados (top 5):")
    for name, cnt in names.most_common(5):
        print(f"  - {name} ({cnt} msgs)")

    target_name = input("\nEscribe el nombre EXACTO a clonar (como aparece arriba): ").strip()

    parser = WhatsAppParser()
    messages = parser.parse_chat(target_name, content=raw_text)
    if not messages:
        print(f"\n⚠ No se encontraron mensajes de '{target_name}'.")
        return

    stats = parser.get_statistics()
    print(f"\n✓ Análisis completado:")
    print(f"  • Mensajes encontrados: {stats.get('total_messages', 0)}")
    print(f"  • Total de palabras: {stats.get('total_words', 0)}")
    print(f"  • Promedio por mensaje: {stats.get('avg_message_length', 0)} palabras")
    if stats.get('top_words'):
        print("\n  • Palabras más usadas:")
        for word, count in stats['top_words'][:5]:
            print(f"    - {word}: {count} veces")

    parser.export_for_training(messages_file)

    print("\n" + "="*50)
    print("\nPASO 2: Iniciando chatbot local...\n")

    print("Modelos recomendados:")
    print("1. llama3.2 (3B) - Rápido, ~2GB RAM")
    print("2. llama3.2 (1B) - Muy rápido, ~1GB RAM")
    print("3. mistral (7B)  - Mejor calidad, ~4GB RAM")
    print("4. gemma2 (2B)   - Balance, ~2GB RAM")

    modelo_input = input("\nSelecciona modelo (1-4) o Enter para llama3.2: ").strip()
    modelos = {'1': 'llama3.2', '2': 'llama3.2:1b', '3': 'mistral', '4': 'gemma2:2b'}
    modelo = modelos.get(modelo_input, 'llama3.2')

    print(f"\nInicializando chatbot con {modelo}...")
    try:
        chatbot = LocalChatbot(messages_file, modelo)
        print("\n╔══════════════════════════════════════════════════╗")
        print("║              CHAT INICIADO                       ║")
        print("╚══════════════════════════════════════════════════╝")
        print("\nComandos: 'salir', 'limpiar', 'guardar'\n")

        while True:
            user_input = input("\n\033[1;36mTú:\033[0m ")
            if user_input.lower() == 'salir':
                chatbot.save_conversation()
                print("\n👋 Hasta pronto\n")
                break
            if user_input.lower() == 'limpiar':
                chatbot.conversation_history = []
                print("✓ Historial limpiado"); continue
            if user_input.lower() == 'guardar':
                chatbot.save_conversation(); continue
            if not user_input.strip():
                continue

            print("\n\033[1;32mRespuesta:\033[0m ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nVerifica que Ollama esté instalado:")
        print("curl -fsSL https://ollama.com/install.sh | sh")

if __name__ == "__main__":
    main()
