# main.py
from bot.local_chatbot import LocalChatbot
from chatio.whatsapp import (
    load_whatsapp_export_text,
    detect_participants,
    WhatsAppParser,
)

# ✨ Persona Card / Memoria
from bot.services.persona_card import (
    build_persona_card,
    compose_pinned,
    recall_memory,
)

def run_cli():
    print("╔══════════════════════════════════════════════════╗")
    print("║  Sistema de Chat Personalizado LOCAL            ║")
    print("║  100% Privado - Sin Internet                    ║")
    print("╚══════════════════════════════════════════════════╝\n")

    print("PASO 1: Procesando conversaciones de WhatsApp...\n")

    whatsapp_file = "_chat.txt"          # <- ajusta si usas .zip
    messages_file = "mensajes_procesados.json"

    try:
        raw_text = load_whatsapp_export_text(whatsapp_file)
    except Exception as e:
        print(f"❌ No se pudo leer el export de WhatsApp: {e}")
        return

    names = detect_participants(raw_text)
    if not names:
        print("⚠ No se detectaron remitentes. ¿Es el archivo correcto?")
        return

    print("\nParticipantes detectados (top 5):")
    for i, (name, cnt) in enumerate(names.most_common(5), 1):
        print(f"  {i}. {name} ({cnt} msgs)")

    target_name = input("\nEscribe el nombre EXACTO a clonar (como aparece arriba): ").strip()
    if not target_name:
        print("⚠ No ingresaste un nombre.")
        return

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

    # Exporta datasets
    parser.export_for_training(messages_file)
    parser.export_all("conversacion_completa.json")

    # ===== Persona Card & Memoria para el target elegido =====
    try:
        persona_text, memory = build_persona_card(
            "conversacion_completa.json",
            target_name=target_name,
            memory_dir="memory",
            save=True,  # guarda memory/<slug>.json
        )
        pinned_block = compose_pinned(persona_text)
        print("\n✓ Persona Card generada para:", target_name)
    except Exception as e:
        print(f"\n⚠ No se pudo construir la Persona Card: {e}")
        persona_text, memory, pinned_block = None, None, None

    print("\n" + "="*50)
    print("\nPASO 2: Iniciando chatbot local...\n")
    print("Modelos recomendados:")
    print("1. llama3.2 (3B) - Rápido, ~2GB RAM")
    print("2. llama3.2:1b   - Muy rápido, ~1GB RAM")
    print("3. mistral (7B)  - Mejor calidad, ~4GB RAM")
    print("4. gemma2:2b     - Balance, ~2GB RAM")

    opciones = {'1': 'llama3.2', '2': 'llama3.2:1b', '3': 'mistral', '4': 'gemma2:2b'}
    modelo = opciones.get(input("\nSelecciona modelo (1-4) o Enter para llama3.2: ").strip(), 'llama3.2')

    print(f"\nInicializando chatbot con {modelo}...")
    try:
        # Overrides recomendados (puedes ajustar a tu gusto)
        overrides = {
            "media": {
                "media_dir": "media",        # carpeta con .jpg/.png/.mp4
                "vision_model": None,        # None = sin caption por visión (usa OCR/Whisper)
                "frame_stride_sec": 8,
                "max_frames": 6,
                "use_images_in_chat": False  # no adjuntamos imágenes por turno
            },
            # Opcional: mejorar recall para preguntas cortas
            "rag": {
                "multiquery_min_chars": 0,
                "fetch_k": 96,
                "k": 12
            }
        }

        # Intento 1: pasar Persona Card/Memoria si tu LocalChatbot ya soporta estos kwargs
        use_postprocess_memory = True
        try:
            if pinned_block or memory:
                chatbot = LocalChatbot(
                    messages_file,
                    modelo,
                    overrides=overrides,
                    persona_pinned=pinned_block,  # si tu LocalChatbot lo soporta
                    persona_memory=memory         # si tu LocalChatbot lo soporta
                )
                use_postprocess_memory = False
            else:
                chatbot = LocalChatbot(messages_file, modelo, overrides=overrides)
        except TypeError:
            # Tu LocalChatbot aún no acepta persona_pinned/persona_memory
            chatbot = LocalChatbot(messages_file, modelo, overrides=overrides)
            use_postprocess_memory = True

        print("\n╔══════════════════════════════════════════════════╗")
        print("║              CHAT INICIADO                       ║")
        print("╚══════════════════════════════════════════════════╝")
        print("\nComandos: 'salir', 'limpiar', 'guardar'\n")

        while True:
            user_input = input("\n\033[1;36mTú:\033[0m ")
            if not user_input.strip():
                continue
            low = user_input.lower().strip()
            if low == 'salir':
                chatbot.save_conversation()
                print("\n👋 Hasta pronto\n")
                break
            if low == 'limpiar':
                chatbot.conversation_history = []
                print("✓ Historial limpiado")
                continue
            if low == 'guardar':
                chatbot.save_conversation()
                continue

            print("\n\033[1;32mRespuesta:\033[0m ", end="", flush=True)
            raw_answer = chatbot.chat(user_input)

            # Fallback: efecto “memoria” sin tocar LocalChatbot
            if use_postprocess_memory and memory:
                recall_line = recall_memory(user_input, memory)
                if recall_line and not raw_answer.lower().startswith("sí, recuerdo"):
                    raw_answer = f"{recall_line}\n\n{raw_answer}"

            print(raw_answer)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nVerifica que Ollama esté instalado y corriendo (ollama serve).")

if __name__ == "__main__":
    run_cli()
