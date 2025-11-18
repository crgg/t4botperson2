# t4ever_api.py
"""
API REST para T4ever - Integración del chatbot de legado digital
Endpoints para que frontend web/mobile consuma el chatbot
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import json
import os

# Importar tu chatbot existente
from bot.local_chatbot import LocalChatbot

# ========================================
# Modelos de Datos (Request/Response)
# ========================================

class ChatMessage(BaseModel):
    """Mensaje individual del chat"""
    role: str = Field(..., description="'user' o 'assistant'")
    content: str = Field(..., description="Contenido del mensaje")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ChatRequest(BaseModel):
    """Request para enviar mensaje al chatbot"""
    message: str = Field(..., description="Mensaje del usuario/beneficiario")
    session_id: Optional[str] = Field(default=None, description="ID de sesión para mantener contexto")
    beneficiary_context: Optional[Dict] = Field(default=None, description="Contexto del beneficiario (nombre, relación)")

class ChatResponse(BaseModel):
    """Response del chatbot"""
    message: str = Field(..., description="Respuesta del bot")
    session_id: str = Field(..., description="ID de sesión")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    legacy_name: str = Field(..., description="Nombre de la persona cuyo legado se representa")

class LegacyInfo(BaseModel):
    """Información del legado"""
    legacy_id: str
    person_name: str
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    bio: Optional[str] = None
    photo_url: Optional[str] = None
    created_at: str
    
class BeneficiaryInfo(BaseModel):
    """Info del beneficiario que chatea"""
    beneficiary_id: str
    name: str
    relationship: str  # "child", "grandchild", "spouse", "friend", "other"
    access_level: str = "standard"  # "standard", "premium", "admin"

# ========================================
# Estado Global de la API
# ========================================

class ChatbotManager:
    """Gestiona instancias de chatbot (uno por legacy_id)"""
    
    def __init__(self):
        self.chatbots: Dict[str, LocalChatbot] = {}
        self.sessions: Dict[str, Dict] = {}  # session_id -> {"legacy_id": ..., "history": [...]}
        
    def get_or_create_chatbot(self, legacy_id: str, config: Dict) -> LocalChatbot:
        """Obtiene chatbot existente o crea uno nuevo"""
        if legacy_id not in self.chatbots:
            print(f"[ChatbotManager] Creando nuevo chatbot para legacy_id={legacy_id}")
            
            # Cargar configuración del legado desde archivos
            messages_file = config.get("messages_file", f"legacies/{legacy_id}/mensajes_procesados.json")
            model_name = config.get("model_name", "mistral")
            
            # Verificar que existan los archivos
            if not os.path.exists(messages_file):
                raise FileNotFoundError(f"No se encontró el archivo de mensajes: {messages_file}")
            
            # Crear instancia del chatbot
            self.chatbots[legacy_id] = LocalChatbot(
                messages_file=messages_file,
                model_name=model_name,
                overrides=config.get("overrides", {})
            )
            
        return self.chatbots[legacy_id]
    
    def create_session(self, legacy_id: str) -> str:
        """Crea nueva sesión de chat"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "legacy_id": legacy_id,
            "history": [],
            "created_at": datetime.now().isoformat()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Obtiene sesión existente"""
        return self.sessions.get(session_id)
    
    def update_session_history(self, session_id: str, user_msg: str, bot_msg: str):
        """Actualiza historial de la sesión"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].extend([
                {"role": "user", "content": user_msg, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": bot_msg, "timestamp": datetime.now().isoformat()}
            ])

# ========================================
# Inicialización de FastAPI
# ========================================

app = FastAPI(
    title="T4ever Legacy Chat API",
    description="API para interactuar con legados digitales",
    version="1.0.0"
)

# CORS para permitir requests desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tus dominios: ["https://t4ever.com", "http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manager global
chatbot_manager = ChatbotManager()

# ========================================
# Utilidades de Autenticación (básica)
# ========================================

async def verify_api_key(x_api_key: str = Header(...)):
    """Verifica API key del cliente (tu frontend)"""
    # TODO: En producción, implementar verificación real con DB
    # Por ahora, acepta cualquier key que empiece con "t4ever_"
    if not x_api_key.startswith("t4ever_"):
        raise HTTPException(status_code=401, detail="API Key inválida")
    return x_api_key

async def verify_beneficiary_access(legacy_id: str, beneficiary_id: str):
    """Verifica que el beneficiario tenga acceso a este legado"""
    # TODO: Verificar en DB que beneficiary_id tiene acceso a legacy_id
    # Por ahora, permitir todos los accesos
    return True

# ========================================
# ENDPOINTS PRINCIPALES
# ========================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "T4ever Legacy Chat API",
        "version": "1.0.0"
    }

@app.post("/api/chat/{legacy_id}", response_model=ChatResponse)
async def chat_with_legacy(
    legacy_id: str,
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Endpoint principal: enviar mensaje al legado y recibir respuesta
    
    Ejemplo de uso:
    ```
    POST /api/chat/juan-perez-123
    Headers: X-API-Key: t4ever_demo_key_123
    Body: {
        "message": "Papá, cuéntame sobre tu infancia",
        "session_id": "optional-session-uuid",
        "beneficiary_context": {
            "name": "María",
            "relationship": "child"
        }
    }
    ```
    """
    try:
        # 1. Obtener o crear chatbot para este legado
        config = {
            "messages_file": f"legacies/{legacy_id}/mensajes_procesados.json",
            "model_name": "mistral",  # Puedes parametrizar esto desde DB
        }
        
        chatbot = chatbot_manager.get_or_create_chatbot(legacy_id, config)
        
        # 2. Gestionar sesión
        session_id = request.session_id
        if not session_id:
            session_id = chatbot_manager.create_session(legacy_id)
        else:
            # Verificar que la sesión existe y corresponde a este legado
            session = chatbot_manager.get_session(session_id)
            if not session or session["legacy_id"] != legacy_id:
                raise HTTPException(status_code=400, detail="Sesión inválida")
        
        # 3. Obtener respuesta del chatbot
        user_message = request.message
        bot_response = chatbot.chat(user_message)
        
        # 4. Actualizar historial de sesión
        chatbot_manager.update_session_history(session_id, user_message, bot_response)
        
        # 5. Retornar respuesta
        return ChatResponse(
            message=bot_response,
            session_id=session_id,
            legacy_name=chatbot.persona_name,
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Legado no encontrado: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/api/legacy/{legacy_id}/info", response_model=LegacyInfo)
async def get_legacy_info(
    legacy_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Obtiene información básica del legado (nombre, foto, bio, fechas)
    """
    # TODO: Cargar desde DB real
    # Por ahora, retornar datos de ejemplo
    return LegacyInfo(
        legacy_id=legacy_id,
        person_name="Juan Pérez",
        birth_date="1950-05-15",
        death_date="2024-01-20",
        bio="Padre amoroso, ingeniero, amante del fútbol y la familia",
        photo_url=f"/media/{legacy_id}/profile.jpg",
        created_at="2024-02-01T10:00:00Z"
    )

@app.post("/api/session/new/{legacy_id}")
async def create_new_session(
    legacy_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Crea una nueva sesión de chat (útil para empezar conversación fresca)
    """
    session_id = chatbot_manager.create_session(legacy_id)
    return {
        "session_id": session_id,
        "legacy_id": legacy_id,
        "created_at": datetime.now().isoformat()
    }

@app.get("/api/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Obtiene el historial completo de una sesión
    """
    session = chatbot_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    return {
        "session_id": session_id,
        "legacy_id": session["legacy_id"],
        "history": session["history"],
        "created_at": session["created_at"]
    }

@app.post("/api/legacy/{legacy_id}/suggested-questions")
async def get_suggested_questions(
    legacy_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Genera preguntas sugeridas basadas en el contenido del legado
    (útil para botones de inicio rápido en el frontend)
    """
    # TODO: Analizar corpus y generar preguntas inteligentes
    # Por ahora, retornar preguntas genéricas
    return {
        "legacy_id": legacy_id,
        "questions": [
            "Cuéntame sobre tu infancia",
            "¿Cuál fue tu momento más feliz?",
            "¿Qué consejo me darías hoy?",
            "Recuérdame esa historia de cuando...",
            "¿Qué era lo que más te gustaba hacer?"
        ]
    }

# ========================================
# Endpoint para construir legado (persona viva)
# ========================================

@app.post("/api/legacy/build/{legacy_id}/question")
async def get_next_interview_question(
    legacy_id: str,
    previous_answer: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Para el modo "construcción de legado" (persona aún viva)
    Devuelve la siguiente pregunta de la entrevista guiada
    """
    # TODO: Implementar InterviewEngine completo
    # Por ahora, preguntas hardcodeadas
    questions = [
        {"id": 1, "category": "valores", "question": "¿Cuáles son los 3 valores más importantes que guiaron tu vida?"},
        {"id": 2, "category": "historias", "question": "Cuéntame la historia más divertida de tu infancia"},
        {"id": 3, "category": "sabiduría", "question": "¿Qué consejo le darías a tus nietos cuando tengan 20 años?"},
    ]
    
    # TODO: Guardar previous_answer en DB y determinar siguiente pregunta
    
    return {
        "legacy_id": legacy_id,
        "next_question": questions[0],  # Simplificado
        "progress": 33,  # Porcentaje de completitud
        "total_questions": 3
    }
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}
# ========================================
# MAIN: Ejecutar servidor
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuración de servidor
    uvicorn.run(
        "t4ever_api:app",
        host="0.0.0.0",  # Escuchar en todas las interfaces
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )