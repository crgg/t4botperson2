#!/usr/bin/env python3
# create_demo_legacy.py
"""
Crea un legado de ejemplo para probar la API sin necesidad de WhatsApp
"""

import os
import json
import numpy as np
from pathlib import Path

def create_demo_legacy(legacy_id="juan-perez-123", person_name="Juan Pérez"):
    """
    Crea un legado de ejemplo con mensajes ficticios
    """
    print(f"🏗️  Creando legado de ejemplo: {legacy_id}")
    
    # Crear estructura de directorios
    base_path = Path("legacies") / legacy_id
    base_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in ["media", "documents", "memory"]:
        (base_path / subdir).mkdir(exist_ok=True)
    
    print(f"✅ Estructura creada en: {base_path}")
    
    # Mensajes de ejemplo (conversación ficticia)
    example_messages = [
        {
            "date": "1/15/24",
            "time": "10:30:00 AM",
            "name": person_name,
            "message": "Buenos días! Cómo amaneciste hoy?"
        },
        {
            "date": "1/15/24",
            "time": "2:15:00 PM",
            "name": person_name,
            "message": "Estoy recordando cuando íbamos a la playa todos los veranos. Qué tiempos aquellos!"
        },
        {
            "date": "1/16/24",
            "time": "9:00:00 AM",
            "name": person_name,
            "message": "Te quiero mucho, nunca lo olvides. Siempre estaré aquí para ti."
        },
        {
            "date": "1/16/24",
            "time": "11:30:00 AM",
            "name": person_name,
            "message": "Recuerda que lo más importante en la vida es la familia y ser una buena persona."
        },
        {
            "date": "1/17/24",
            "time": "3:45:00 PM",
            "name": person_name,
            "message": "Mi consejo para ti: sigue tus sueños, trabaja duro, y nunca pierdas la fe."
        },
        {
            "date": "1/18/24",
            "time": "8:20:00 AM",
            "name": person_name,
            "message": "Me encanta el café de las mañanas y leer el periódico. Las pequeñas cosas son las que importan."
        },
        {
            "date": "1/19/24",
            "time": "5:00:00 PM",
            "name": person_name,
            "message": "Cuando era joven, soñaba con viajar por el mundo. Algunos sueños los cumplí, otros quedaron pendientes."
        },
        {
            "date": "1/20/24",
            "time": "7:30:00 PM",
            "name": person_name,
            "message": "La vida es corta, hay que vivirla con intensidad y sin arrepentimientos."
        },
        {
            "date": "1/21/24",
            "time": "10:00:00 AM",
            "name": person_name,
            "message": "Estoy orgulloso de ti y de todo lo que has logrado. Sigue así!"
        },
        {
            "date": "1/22/24",
            "time": "6:15:00 PM",
            "name": person_name,
            "message": "Recuerdo cuando eras pequeño y te enseñé a andar en bicicleta. Caíste varias veces pero nunca te rendiste."
        },
        {
            "date": "1/23/24",
            "time": "12:00:00 PM",
            "name": person_name,
            "message": "Mi comida favorita siempre fue el asado de los domingos con toda la familia reunida."
        },
        {
            "date": "1/24/24",
            "time": "4:30:00 PM",
            "name": person_name,
            "message": "Si pudiera darte un consejo: valora cada momento con las personas que amas."
        },
        {
            "date": "1/25/24",
            "time": "9:45:00 AM",
            "name": person_name,
            "message": "A veces me pongo nostálgico recordando los viejos tiempos, pero estoy feliz con la vida que viví."
        },
        {
            "date": "1/26/24",
            "time": "2:00:00 PM",
            "name": person_name,
            "message": "Te echo de menos cuando no estás. Espero verte pronto!"
        },
        {
            "date": "1/27/24",
            "time": "7:00:00 PM",
            "name": person_name,
            "message": "Nunca olvides de dónde vienes y quién eres. Eso es lo que te hace especial."
        }
    ]
    
    # Guardar mensajes procesados
    messages_file = base_path / "mensajes_procesados.json"
    with open(messages_file, "w", encoding="utf-8") as f:
        json.dump(example_messages, f, ensure_ascii=False, indent=2)
    print(f"✅ Mensajes guardados: {len(example_messages)} mensajes")
    
    # Crear conversación completa (igual que mensajes procesados para este ejemplo)
    with open(base_path / "conversacion_completa.json", "w", encoding="utf-8") as f:
        json.dump(example_messages, f, ensure_ascii=False, indent=2)
    
    # Metadata
    metadata = {
        "legacy_id": legacy_id,
        "person_name": person_name,
        "created_at": "2024-11-13T00:00:00Z",
        "source": "demo",
        "status": "active",
        "files": {
            "messages": "mensajes_procesados.json",
            "full_conversation": "conversacion_completa.json"
        }
    }
    
    with open(base_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("✅ Metadata guardada")
    
    # Crear índice RAG simple
    print("🔍 Creando índice RAG...")
    
    try:
        # Intentar usar tu sistema RAG existente
        from rag.retrieve import ensure_rag_index
        
        texts_path = str(base_path / "rag_texts.json")
        vecs_path = str(base_path / "rag_vecs.npy")
        sparse_path = str(base_path / "rag_sparse.json")
        meta_path = str(base_path / "rag_meta.json")
        
        corpus_texts, corpus_vecs, sparse_index = ensure_rag_index(
            example_messages,
            texts_path=texts_path,
            vecs_path=vecs_path,
            sparse_path=sparse_path,
            meta_path=meta_path
        )
        
        print(f"✅ Índice RAG creado: {len(corpus_texts)} documentos")
        
    except Exception as e:
        print(f"⚠️  No se pudo crear índice RAG automáticamente: {e}")
        print("   Creando índice básico manual...")
        
        # Crear índices básicos manualmente
        texts = [m["message"] for m in example_messages]
        
        # Guardar textos
        with open(base_path / "rag_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        
        # Crear embeddings dummy (reemplazar con reales en producción)
        n = len(texts)
        d = 768  # dimensión estándar
        dummy_vecs = np.random.randn(n, d).astype(np.float32)
        dummy_vecs = dummy_vecs / np.linalg.norm(dummy_vecs, axis=1, keepdims=True)
        np.save(base_path / "rag_vecs.npy", dummy_vecs)
        
        # Sparse index básico
        sparse = {
            "idf": {},
            "docs": [set() for _ in range(n)],
            "inv": {},
            "N": n
        }
        with open(base_path / "rag_sparse.json", "w", encoding="utf-8") as f:
            json.dump({
                "idf": {},
                "docs": [list(s) for s in sparse["docs"]],
                "inv": {},
                "N": n
            }, f)
        
        # Meta
        with open(base_path / "rag_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "hash": "demo",
                "embed_model": "demo"
            }, f)
        
        print("✅ Índice RAG básico creado")
    
    # Resumen
    print("\n" + "="*60)
    print(f"✅ Legado '{legacy_id}' creado exitosamente")
    print("="*60)
    print(f"📁 Ubicación: {base_path.absolute()}")
    print(f"📊 Mensajes: {len(example_messages)}")
    print("\n💡 Ahora puedes:")
    print(f"   1. Ejecutar tests: python3 test_api.py {legacy_id}")
    print(f"   2. Chatear: POST /api/chat/{legacy_id}")
    print(f"   3. Crear tu propio legado desde WhatsApp con setup_legacy_structure.py")
    print()

if __name__ == "__main__":
    import sys
    
    legacy_id = sys.argv[1] if len(sys.argv) > 1 else "juan-perez-123"
    person_name = sys.argv[2] if len(sys.argv) > 2 else "Juan Pérez"
    
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         🏗️  CREAR LEGADO DE EJEMPLO                       ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    try:
        create_demo_legacy(legacy_id, person_name)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)