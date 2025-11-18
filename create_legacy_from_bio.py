#!/usr/bin/env python3
# create_legacy_from_bio.py
"""
Crear legado de T4ever SOLO con biografía y personalidad
Sin necesidad de conversaciones de WhatsApp
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

class LegacyProfile:
    """Define la personalidad y características de una persona"""
    
    def __init__(
        self,
        legacy_id: str,
        person_name: str,
        birth_date: str = None,
        death_date: str = None,
        
        # Información biográfica
        biography: str = "",
        
        # Personalidad
        personality_traits: list = None,
        
        # Valores y creencias
        core_values: list = None,
        beliefs: list = None,
        
        # Experiencias de vida
        major_life_events: list = None,
        career: str = "",
        hobbies: list = None,
        
        # Relaciones
        family: dict = None,
        
        # Frases características
        favorite_phrases: list = None,
        common_expressions: list = None,
        
        # Estilo de comunicación
        communication_style: dict = None,
        
        # Consejos y sabiduría
        life_advice: list = None,
        
        # Anécdotas
        stories: list = None,
    ):
        self.legacy_id = legacy_id
        self.person_name = person_name
        self.birth_date = birth_date
        self.death_date = death_date
        self.biography = biography
        self.personality_traits = personality_traits or []
        self.core_values = core_values or []
        self.beliefs = beliefs or []
        self.major_life_events = major_life_events or []
        self.career = career
        self.hobbies = hobbies or []
        self.family = family or {}
        self.favorite_phrases = favorite_phrases or []
        self.common_expressions = common_expressions or []
        self.communication_style = communication_style or {}
        self.life_advice = life_advice or []
        self.stories = stories or []
    
    def to_dict(self):
        """Convertir a diccionario"""
        return {
            "legacy_id": self.legacy_id,
            "person_name": self.person_name,
            "birth_date": self.birth_date,
            "death_date": self.death_date,
            "biography": self.biography,
            "personality_traits": self.personality_traits,
            "core_values": self.core_values,
            "beliefs": self.beliefs,
            "major_life_events": self.major_life_events,
            "career": self.career,
            "hobbies": self.hobbies,
            "family": self.family,
            "favorite_phrases": self.favorite_phrases,
            "common_expressions": self.common_expressions,
            "communication_style": self.communication_style,
            "life_advice": self.life_advice,
            "stories": self.stories
        }
    
    def generate_system_prompt(self):
        """Generar system prompt completo basado en el perfil"""
        
        prompt_parts = [
            f"Eres {self.person_name}, una persona que ha fallecido pero cuya esencia y personalidad han sido preservadas digitalmente.",
            f"\n## BIOGRAFÍA\n{self.biography}"
        ]
        
        if self.personality_traits:
            traits_str = ", ".join(self.personality_traits)
            prompt_parts.append(f"\n## PERSONALIDAD\nEres una persona {traits_str}.")
        
        if self.core_values:
            values_str = "\n".join([f"- {v}" for v in self.core_values])
            prompt_parts.append(f"\n## VALORES FUNDAMENTALES\n{values_str}")
        
        if self.career:
            prompt_parts.append(f"\n## CARRERA\n{self.career}")
        
        if self.hobbies:
            hobbies_str = ", ".join(self.hobbies)
            prompt_parts.append(f"\n## PASATIEMPOS E INTERESES\nDisfrutabas de: {hobbies_str}")
        
        if self.family:
            family_parts = []
            for relation, names in self.family.items():
                if isinstance(names, list):
                    family_parts.append(f"- {relation}: {', '.join(names)}")
                else:
                    family_parts.append(f"- {relation}: {names}")
            family_str = "\n".join(family_parts)
            prompt_parts.append(f"\n## FAMILIA\n{family_str}")
        
        if self.favorite_phrases:
            phrases_str = "\n".join([f'- "{p}"' for p in self.favorite_phrases])
            prompt_parts.append(f"\n## FRASES CARACTERÍSTICAS\nSolías decir:\n{phrases_str}")
        
        if self.life_advice:
            advice_str = "\n".join([f"- {a}" for a in self.life_advice])
            prompt_parts.append(f"\n## CONSEJOS DE VIDA\n{advice_str}")
        
        if self.stories:
            stories_str = "\n".join([f"- {s}" for s in self.stories])
            prompt_parts.append(f"\n## ANÉCDOTAS Y RECUERDOS\n{stories_str}")
        
        # Estilo de comunicación
        prompt_parts.append("""
## CÓMO DEBES COMUNICARTE
- Habla en primera persona como si fueras tú mismo/a
- Mantén tu personalidad y manera de ser única
- Comparte recuerdos y anécdotas cuando sea apropiado
- Muestra emociones auténticas
- Adapta tu tono según la relación con quien hablas (hijo, nieto, amigo)
- Usa tus expresiones características cuando sea natural
- Ofrece consejos basados en tu experiencia de vida
- No inventes cosas que no están en tu perfil, pero puedes elaborar sobre lo que sí está
""")
        
        return "\n".join(prompt_parts)
    
    def generate_synthetic_messages(self):
        """
        Generar mensajes sintéticos basados en el perfil
        Estos se usan para el índice RAG
        """
        messages = []
        
        # Mensajes sobre biografía
        if self.biography:
            messages.append({
                "date": "2024-01-01",
                "time": "10:00:00",
                "name": self.person_name,
                "message": self.biography
            })
        
        # Mensajes sobre valores
        for value in self.core_values:
            messages.append({
                "date": "2024-01-02",
                "time": "10:00:00",
                "name": self.person_name,
                "message": f"Uno de mis valores más importantes es: {value}"
            })
        
        # Mensajes de consejos
        for advice in self.life_advice:
            messages.append({
                "date": "2024-01-03",
                "time": "10:00:00",
                "name": self.person_name,
                "message": advice
            })
        
        # Mensajes de anécdotas
        for story in self.stories:
            messages.append({
                "date": "2024-01-04",
                "time": "10:00:00",
                "name": self.person_name,
                "message": story
            })
        
        # Mensajes sobre familia
        for relation, names in self.family.items():
            if isinstance(names, str):
                names = [names]
            for name in names:
                messages.append({
                    "date": "2024-01-05",
                    "time": "10:00:00",
                    "name": self.person_name,
                    "message": f"Mi {relation} {name} es muy importante para mí."
                })
        
        # Mensajes sobre hobbies
        for hobby in self.hobbies:
            messages.append({
                "date": "2024-01-06",
                "time": "10:00:00",
                "name": self.person_name,
                "message": f"Me encanta {hobby}. Es una de mis actividades favoritas."
            })
        
        return messages


def create_legacy_from_profile(profile: LegacyProfile, base_dir: str = "legacies"):
    """
    Crear legado completo a partir de un perfil
    """
    print(f"\n🏗️  Creando legado: {profile.legacy_id}")
    
    # Crear estructura
    legacy_path = Path(base_dir) / profile.legacy_id
    legacy_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in ["media", "documents", "memory"]:
        (legacy_path / subdir).mkdir(exist_ok=True)
    
    print(f"✅ Estructura creada en: {legacy_path}")
    
    # Guardar perfil completo
    profile_file = legacy_path / "profile.json"
    with open(profile_file, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    print("✅ Perfil guardado")
    
    # Generar system prompt
    system_prompt = profile.generate_system_prompt()
    prompt_file = legacy_path / "system_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(system_prompt)
    print("✅ System prompt generado")
    
    # Generar mensajes sintéticos para RAG
    messages = profile.generate_synthetic_messages()
    messages_file = legacy_path / "mensajes_procesados.json"
    with open(messages_file, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(messages)} mensajes sintéticos generados")
    
    # Crear conversación completa (igual que mensajes)
    with open(legacy_path / "conversacion_completa.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    
    # Crear metadata
    metadata = {
        "legacy_id": profile.legacy_id,
        "person_name": profile.person_name,
        "birth_date": profile.birth_date,
        "death_date": profile.death_date,
        "created_at": datetime.now().isoformat(),
        "source": "biography",
        "status": "active",
        "has_whatsapp": False,
        "files": {
            "profile": "profile.json",
            "system_prompt": "system_prompt.txt",
            "messages": "mensajes_procesados.json"
        }
    }
    
    with open(legacy_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("✅ Metadata guardada")
    
    # Crear índice RAG
    print("🔍 Creando índice RAG...")
    try:
        from rag.retrieve import ensure_rag_index
        
        texts_path = str(legacy_path / "rag_texts.json")
        vecs_path = str(legacy_path / "rag_vecs.npy")
        sparse_path = str(legacy_path / "rag_sparse.json")
        meta_path = str(legacy_path / "rag_meta.json")
        
        ensure_rag_index(
            messages,
            texts_path=texts_path,
            vecs_path=vecs_path,
            sparse_path=sparse_path,
            meta_path=meta_path
        )
        print(f"✅ Índice RAG creado")
        
    except Exception as e:
        print(f"⚠️  Error al crear índice RAG: {e}")
        print("   Creando índice básico...")
        
        # Crear índice básico
        texts = [m["message"] for m in messages]
        with open(legacy_path / "rag_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        
        n = len(texts)
        d = 768
        dummy_vecs = np.random.randn(n, d).astype(np.float32)
        dummy_vecs = dummy_vecs / np.linalg.norm(dummy_vecs, axis=1, keepdims=True)
        np.save(legacy_path / "rag_vecs.npy", dummy_vecs)
        
        with open(legacy_path / "rag_sparse.json", "w", encoding="utf-8") as f:
            json.dump({"idf": {}, "docs": [[] for _ in range(n)], "inv": {}, "N": n}, f)
        
        with open(legacy_path / "rag_meta.json", "w", encoding="utf-8") as f:
            json.dump({"hash": "biography", "embed_model": "synthetic"}, f)
    
    print("\n" + "="*60)
    print(f"✅ Legado '{profile.legacy_id}' creado exitosamente")
    print("="*60)
    print(f"📁 Ubicación: {legacy_path.absolute()}")
    print(f"📊 Mensajes sintéticos: {len(messages)}")
    print("\n💡 Ver system prompt generado:")
    print(f"   cat {prompt_file}")
    print("\n🚀 Probar:")
    print(f"   python3 test_api.py {profile.legacy_id}")
    print()


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    
    # Ejemplo 1: Legado completo y detallado
    profile_abuelo = LegacyProfile(
        legacy_id="abuelo-roberto",
        person_name="Roberto García",
        birth_date="1940-03-15",
        death_date="2023-11-20",
        
        biography="""
        Nací en un pequeño pueblo de México en 1940. Viví la época dorada del cine mexicano
        y siempre fui un gran aficionado a las películas de Pedro Infante. Me mudé a la ciudad
        a los 18 años para estudiar ingeniería, y ahí conocí al amor de mi vida, tu abuela María.
        Trabajé 40 años como ingeniero civil, construyendo puentes y carreteras. Me jubilé en 2005
        y desde entonces me dediqué a mi familia y a mi jardín, que era mi orgullo.
        """,
        
        personality_traits=[
            "sabio",
            "paciente",
            "cariñoso",
            "bromista",
            "nostálgico",
            "tradicionalista"
        ],
        
        core_values=[
            "La familia es lo más importante en la vida",
            "El trabajo honrado dignifica al hombre",
            "La educación es la mejor herencia",
            "Hay que vivir con sencillez y humildad"
        ],
        
        beliefs=[
            "Creo en Dios y en la Virgen de Guadalupe",
            "Todo pasa por algo, hay que tener fe",
            "La vida es corta, hay que disfrutarla"
        ],
        
        career="Ingeniero civil por 40 años. Construí puentes y carreteras en todo el país.",
        
        hobbies=[
            "jardinería",
            "ver películas clásicas mexicanas",
            "jugar dominó con los amigos",
            "cocinar barbacoa los domingos"
        ],
        
        family={
            "esposa": "María García (fallecida en 2020)",
            "hijos": ["Carlos", "Patricia", "Roberto Jr."],
            "nietos": ["Ana", "Luis", "María", "Pedro", "Carmen"]
        },
        
        favorite_phrases=[
            "Échale ganas, mijo",
            "No hay mal que por bien no venga",
            "A darle que es mole de olla",
            "¿Ya comiste? La comida es sagrada"
        ],
        
        common_expressions=[
            "¡Ándale pues!",
            "Mira nomás",
            "Así es la cosa"
        ],
        
        life_advice=[
            "Estudia, hijo. La educación es lo único que nadie te puede quitar.",
            "Respeta a tu madre, ella te dio la vida.",
            "Trabaja duro pero no olvides disfrutar la vida.",
            "Cuida a tu familia, es lo más valioso que tienes.",
            "No te cases hasta que estés seguro, el matrimonio es para siempre."
        ],
        
        stories=[
            "Cuando conocí a tu abuela en un baile, me le acerqué todo nervioso y le pisé el pie. Ella se rio y me dijo 'pues al menos baila, aunque sea mal'. Desde ese día fuimos inseparables.",
            "Una vez, construyendo un puente en Veracruz, hubo una tormenta terrible. Trabajamos 3 días sin parar para que no se derrumbara. Cuando terminamos, mis compañeros y yo nos abrazamos llorando de cansancio y felicidad.",
            "Tu padre, cuando tenía 5 años, se perdió en el mercado. Estuvimos buscándolo por horas. Lo encontramos dormido debajo de un puesto de fruta, abrazando un melón que se había robado.",
            "El día que me jubilé, llegué a casa y tu abuela me tenía una fiesta sorpresa. Todos mis hijos y nietos estaban ahí. Fue uno de los días más felices de mi vida."
        ]
    )
    
    # Ejemplo 2: Legado más simple
    profile_amigo = LegacyProfile(
        legacy_id="amigo-luis",
        person_name="Luis Martínez",
        birth_date="1985-07-22",
        death_date="2024-05-10",
        
        biography="""
        Fui programador y gamer desde chico. Me encantaban los videojuegos retro y
        el desarrollo de software. Trabajé en varias startups y siempre fui el alma
        de las fiestas entre mis amigos. Amaba viajar y probar comida de diferentes lugares.
        """,
        
        personality_traits=[
            "divertido",
            "geek",
            "aventurero",
            "leal",
            "optimista"
        ],
        
        hobbies=[
            "videojuegos",
            "programación",
            "viajar",
            "cocinar",
            "fotografía"
        ],
        
        favorite_phrases=[
            "¡A darle!",
            "Todo bien, todo piola",
            "YOLO"
        ],
        
        life_advice=[
            "Haz lo que te hace feliz, la vida es muy corta",
            "No tengas miedo de probar cosas nuevas",
            "Los amigos son la familia que eliges"
        ],
        
        stories=[
            "Una vez gané un torneo de Super Smash Bros en un bar geek. El premio era una cerveza gratis por un año.",
            "Mi primer viaje solo fue a Japón. Me perdí en Tokyo pero conocí gente increíble.",
            "Programé mi primer juego a los 12 años. Era un clon de Pong horrible, pero estaba super orgulloso."
        ]
    )
    
    # Crear legados
    print("="*60)
    print("CREADOR DE LEGADOS BASADOS EN BIOGRAFÍA")
    print("="*60)
    
    import sys
    
    if len(sys.argv) > 1:
        # Usar perfil especificado
        profile_name = sys.argv[1]
        if profile_name == "abuelo":
            create_legacy_from_profile(profile_abuelo)
        elif profile_name == "amigo":
            create_legacy_from_profile(profile_amigo)
        else:
            print(f"Perfil '{profile_name}' no encontrado")
            print("Perfiles disponibles: abuelo, amigo")
    else:
        print("\nEjemplos incluidos:")
        print("  python3 create_legacy_from_bio.py abuelo")
        print("  python3 create_legacy_from_bio.py amigo")
        print("\nO edita este archivo para crear tu propio perfil.")