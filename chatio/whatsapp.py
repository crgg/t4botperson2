# chatio/whatsapp.py
import re, json, zipfile, os
from collections import Counter
from typing import List, Dict

def normalize_whatsapp_text(s: str) -> str:
    s = s.replace('\u202F', ' ').replace('\u00A0', ' ')
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    return s

def load_whatsapp_export_text(path_or_txt: str) -> str:
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
    DATE = r'\d{1,2}/\d{1,2}/\d{2,4}'
    WS   = r'[ \t\u00A0\u202F]'
    TIME = r'\d{1,2}:\d{2}(?::\d{2})?'
    AMPM = r'(?:' + WS + r'?[AaPp]\.?M\.?)?'
    patterns = [
        rf'({DATE}),?{WS}+({TIME}{AMPM}){WS}*-\s([^:]+):\s(.+?)(?=\n{DATE}|$)',
        rf'\[({DATE}),?{WS}+({TIME}{AMPM})\]{WS}+([^:]+):\s(.+?)(?=\n\[|$)',
    ]
    names = Counter()
    for pat in patterns:
        for m in re.findall(pat, content, re.DOTALL):
            names[m[2].strip()] += 1
        if names:
            break
    return names

class WhatsAppParser:
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.messages: List[Dict] = []

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
        target_messages: List[Dict] = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                for date, time, name, message in matches:
                    name_clean = name.strip()
                    msg = message.strip()
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
        if not self.messages:
            return {}
        total = len(self.messages)
        total_words = sum(len(m['message'].split()) for m in self.messages)
        avg = total_words / total if total else 0
        word_freq = {}
        for m in self.messages:
            for w in m['message'].lower().split():
                if len(w) > 3:
                    word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            'total_messages': total,
            'total_words': total_words,
            'avg_message_length': round(avg, 2),
            'top_words': top_words
        }

    def export_for_training(self, output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        print(f"✓ {len(self.messages)} mensajes exportados a {output_file}")
