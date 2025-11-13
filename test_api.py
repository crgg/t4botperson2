#!/usr/bin/env python3
# test_api.py
"""
Script de prueba para verificar que la API T4ever funciona correctamente
Ejecutar después de levantar la API: python t4ever_api.py
"""

import requests
import json
import sys
import time

# Configuración
API_URL = "http://localhost:8000"
API_KEY = "t4ever_demo_key_123"
LEGACY_ID = "juan-perez-123"  # Cambiar al ID de tu legado

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def test_health_check():
    """Test 1: Verificar que la API esté corriendo"""
    print("\n🔍 Test 1: Health Check")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API está corriendo: {data['service']} v{data['version']}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que la API esté corriendo: python t4ever_api.py")
        return False

def test_create_session():
    """Test 2: Crear nueva sesión"""
    print("\n🔍 Test 2: Crear nueva sesión")
    try:
        response = requests.post(
            f"{API_URL}/api/session/new/{LEGACY_ID}",
            headers=HEADERS
        )
        if response.status_code == 200:
            data = response.json()
            session_id = data['session_id']
            print(f"✅ Sesión creada: {session_id[:30]}...")
            return session_id
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_get_legacy_info():
    """Test 3: Obtener info del legado"""
    print("\n🔍 Test 3: Obtener info del legado")
    try:
        response = requests.get(
            f"{API_URL}/api/legacy/{LEGACY_ID}/info",
            headers=HEADERS
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Info obtenida:")
            print(f"   Nombre: {data['person_name']}")
            print(f"   Legacy ID: {data['legacy_id']}")
            return True
        elif response.status_code == 404:
            print(f"❌ Legado no encontrado: {LEGACY_ID}")
            print(f"💡 Ejecuta: python setup_legacy_structure.py {LEGACY_ID} 'Nombre Persona' _chat.txt")
            return False
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chat_message(session_id):
    """Test 4: Enviar mensaje al chatbot"""
    print("\n🔍 Test 4: Enviar mensaje al chatbot")
    
    test_messages = [
        "Hola, ¿cómo estás?",
        "Cuéntame algo sobre ti",
        "¿Qué es lo que más te gustaba hacer?"
    ]
    
    try:
        for i, message in enumerate(test_messages, 1):
            print(f"\n   Mensaje {i}/{len(test_messages)}: '{message}'")
            
            payload = {
                "message": message,
                "session_id": session_id,
                "beneficiary_context": {
                    "name": "Tester",
                    "relationship": "friend"
                }
            }
            
            response = requests.post(
                f"{API_URL}/api/chat/{LEGACY_ID}",
                headers=HEADERS,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_message = data['message']
                print(f"   ✅ Respuesta ({len(bot_message)} chars):")
                
                # Mostrar primeros 150 caracteres
                preview = bot_message[:150] + "..." if len(bot_message) > 150 else bot_message
                print(f"      {preview}")
                
                # Pausa entre mensajes
                if i < len(test_messages):
                    time.sleep(1)
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"      Response: {response.text}")
                return False
        
        print(f"\n✅ Todos los mensajes enviados correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_history(session_id):
    """Test 5: Obtener historial de conversación"""
    print("\n🔍 Test 5: Obtener historial de conversación")
    try:
        response = requests.get(
            f"{API_URL}/api/session/{session_id}/history",
            headers=HEADERS
        )
        if response.status_code == 200:
            data = response.json()
            history = data['history']
            print(f"✅ Historial obtenido: {len(history)} mensajes")
            
            # Mostrar resumen
            user_msgs = sum(1 for msg in history if msg['role'] == 'user')
            bot_msgs = sum(1 for msg in history if msg['role'] == 'assistant')
            print(f"   Usuario: {user_msgs} mensajes")
            print(f"   Bot: {bot_msgs} mensajes")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_suggested_questions():
    """Test 6: Obtener preguntas sugeridas"""
    print("\n🔍 Test 6: Obtener preguntas sugeridas")
    try:
        response = requests.post(
            f"{API_URL}/api/legacy/{LEGACY_ID}/suggested-questions",
            headers=HEADERS
        )
        if response.status_code == 200:
            data = response.json()
            questions = data['questions']
            print(f"✅ Preguntas sugeridas obtenidas ({len(questions)}):")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. {q}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Ejecutar todos los tests"""
    print("="*60)
    print("🧪 T4EVER API - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health_check()))
    if not results[-1][1]:
        print("\n❌ API no está disponible. Deteniendo tests.")
        return
    
    # Test 2: Create Session
    session_id = test_create_session()
    results.append(("Create Session", session_id is not None))
    if not session_id:
        print("\n❌ No se pudo crear sesión. Deteniendo tests.")
        return
    
    # Test 3: Get Legacy Info
    results.append(("Get Legacy Info", test_get_legacy_info()))
    
    # Test 4: Chat Messages
    results.append(("Chat Messages", test_chat_message(session_id)))
    
    # Test 5: Get History
    results.append(("Get History", test_get_history(session_id)))
    
    # Test 6: Suggested Questions
    results.append(("Suggested Questions", test_suggested_questions()))
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE TESTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"Resultado: {passed}/{total} tests pasaron")
    print("="*60)
    
    if passed == total:
        print("\n🎉 ¡Todos los tests pasaron! La API está funcionando correctamente.")
        print("\n💡 Próximos pasos:")
        print("   1. Integrar con tu frontend web/Android/iOS")
        print("   2. Ver ejemplos en: frontend_integration_examples.py")
        print("   3. Desplegar en producción: DEPLOYMENT_GUIDE.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) fallaron. Revisa los errores arriba.")

if __name__ == "__main__":
    # Verificar argumentos
    if len(sys.argv) > 1:
        LEGACY_ID = sys.argv[1]
        print(f"💡 Usando Legacy ID: {LEGACY_ID}")
    else:
        print(f"💡 Usando Legacy ID por defecto: {LEGACY_ID}")
        print(f"   Para usar otro: python test_api.py tu-legacy-id\n")
    
    run_all_tests()