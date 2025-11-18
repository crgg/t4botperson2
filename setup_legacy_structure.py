#!/usr/bin/env python3
# setup_legacy_structure.py
"""
Script para preparar la estructura de archivos para cada legado en T4ever
Ejecutar cuando se crea un nuevo legado en el sistema
"""

import os
import json
import shutil
from pathlib import Path

def setup_new_legacy(
    legacy_id: str,
    person_name: str,
    whatsapp_export_file: str,
    base_dir: str = "legacies"
):
    """
    Prepara todos los archivos necesarios para un nuevo legado
    
    Args:
        legacy_id: ID único del legado (ej: "juan-perez-123")
        person_name: Nombre de la persona (ej: "Juan Pérez")
        whatsapp_export_file: Path al export de WhatsApp
        base_dir: Directorio base donde guardar legados
    """
    
    print(f"🏗️  Configurando legado: {legacy_id} ({person_name})")
    
    # 1. Crear estructura de directorios
    legacy_path = Path(base_dir) / legacy_id
    legacy_path.mkdir(parents=True, exist_ok=True)
    
    subdirs = ["media", "documents", "memory"]
    for subdir in subdirs:
        (legacy_path / subdir).mkdir(exist_ok=True)
    
    print(f"✅ Estructura de directorios creada en: {legacy_path}")
    
    # 2. Copiar export de WhatsApp
    if os.path.exists(whatsapp_export_file):
        dest_chat = legacy_path / "_chat.txt"
        shutil.copy(whatsapp_export_file, dest_chat)
        print(f"✅ Export de WhatsApp copiado: {dest_chat}")
    else:
        print(f"⚠️  Export de WhatsApp no encontrado: {whatsapp_export_file}")
        return False
    
    # 3. Procesar export con tu parser existente
    from chatio.whatsapp import WhatsAppParser, load_whatsapp_export_text
    
    try:
        raw_text = load_whatsapp_export_text(str(dest_chat))
        parser = WhatsAppParser()
        messages = parser.parse_chat(person_name, content=raw_text)
        
        if not messages:
            print(f"❌ No se encontraron mensajes de '{person_name}'")
            return False
        
        # Guardar mensajes procesados
        messages_file = legacy_path / "mensajes_procesados.json"
        with open(messages_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        
        # Guardar conversación completa (todos los participantes)
        parser.export_all(str(legacy_path / "conversacion_completa.json"))
        
        stats = parser.get_statistics()
        print(f"✅ Mensajes procesados: {stats.get('total_messages', 0)}")
        
    except Exception as e:
        print(f"❌ Error al procesar WhatsApp: {e}")
        return False
    
    # 4. Crear metadatos del legado
    metadata = {
        "legacy_id": legacy_id,
        "person_name": person_name,
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "source": "whatsapp_export",
        "status": "active",
        "files": {
            "messages": "mensajes_procesados.json",
            "full_conversation": "conversacion_completa.json",
            "whatsapp_export": "_chat.txt",
        }
    }
    
    with open(legacy_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Metadata guardada")
    
    # 5. Crear índice RAG (usando tu sistema existente)
    print("🔍 Creando índice RAG...")
    try:
        from rag.retrieve import ensure_rag_index
        
        training_messages = messages
        
        # Guardar índice en el directorio del legado
        texts_path = str(legacy_path / "rag_texts.json")
        vecs_path = str(legacy_path / "rag_vecs.npy")
        sparse_path = str(legacy_path / "rag_sparse.json")
        meta_path = str(legacy_path / "rag_meta.json")
        
        corpus_texts, corpus_vecs, sparse_index = ensure_rag_index(
            training_messages,
            texts_path=texts_path,
            vecs_path=vecs_path,
            sparse_path=sparse_path,
            meta_path=meta_path
        )
        
        print(f"✅ Índice RAG creado: {len(corpus_texts)} documentos indexados")
        
    except Exception as e:
        print(f"⚠️  Error al crear índice RAG: {e}")
    
    # 6. Resumen final
    print("\n" + "="*60)
    print(f"✅ Legado '{legacy_id}' configurado exitosamente")
    print("="*60)
    print(f"📁 Ubicación: {legacy_path.absolute()}")
    print(f"📊 Estadísticas:")
    print(f"   - Mensajes: {stats.get('total_messages', 0)}")
    print(f"   - Palabras: {stats.get('total_words', 0)}")
    print(f"   - Promedio por mensaje: {stats.get('avg_message_length', 0)}")
    print("\n💡 Siguiente paso: Iniciar API con 'python t4ever_api.py'")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Uso: python setup_legacy_structure.py <legacy_id> <person_name> <whatsapp_export_file>")
        print("\nEjemplo:")
        print("  python setup_legacy_structure.py juan-perez-123 'Juan Pérez' _chat.txt")
        sys.exit(1)
    
    legacy_id = sys.argv[1]
    person_name = sys.argv[2]
    whatsapp_file = sys.argv[3]
    
    success = setup_new_legacy(legacy_id, person_name, whatsapp_file)
    sys.exit(0 if success else 1)