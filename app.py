from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import httpx
import os
import time
import json
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import asyncio
from contextlib import asynccontextmanager
import urllib.parse
import edge_tts
import tempfile
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
# Add file handler for production logs
os.makedirs("logs", exist_ok=True)
file_handler = RotatingFileHandler(
    "logs/app.log", maxBytes=10485760, backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# Load environment variables
print("Before loading .env file")
load_dotenv(verbose=True)
print("After loading .env file")

# API keys and configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Debug print
print(f"GEMINI_API_KEY value: {GEMINI_API_KEY}")
print(f"ELEVEN_LABS_API_KEY value: {ELEVEN_LABS_API_KEY}")

# Lifespan event handler for startup/shutdown tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validate API keys are present
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
    if not ELEVEN_LABS_API_KEY:
        logger.error("ELEVEN_LABS_API_KEY not configured")
    
    logger.info(f"Starting application in {ENVIRONMENT} environment")
    yield
    # Shutdown: cleanup resources
    logger.info("Shutting down application")

# Initialize FastAPI with lifespan and custom Swagger UI
app = FastAPI(
    title="AI Assistant API",
    description="Production-ready API for AI text generation and text-to-speech with multiple personas",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,  # Hide schemas section by default
        "displayRequestDuration": True,   # Show request duration
        "filter": True,                   # Enable filtering operations
        "syntaxHighlight.theme": "monokai",
        "tryItOutEnabled": True,          # Enable "Try it out" by default
        "persistAuthorization": True      # Remember authorization
    }
)

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    return response

# Define models with validation
class InputMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    messages: List[InputMessage] = Field(..., min_items=1)

class ChatResponse(BaseModel):
    message: str

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: str = Field(default="EXAVITQu4vr4xnSDxMaL")
    model_id: str = Field(default="eleven_multilingual_v2")
    stability: float = Field(default=0.5, ge=0, le=1)
    similarity_boost: float = Field(default=0.75, ge=0, le=1)
    
    model_config = {"protected_namespaces": ()}

# Dependency for API key validation
async def validate_gemini_api_key():
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        raise HTTPException(status_code=500, detail="API key not configured")
    return GEMINI_API_KEY

# Gemini API endpoint
@app.post("/api/query-gemini", response_model=ChatResponse, tags=["AI"])
async def query_gemini(
    request: ChatRequest, 
    api_key: str = Depends(validate_gemini_api_key)
):
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] POST /api/query-gemini")
    
    try:
        # Transform messages for Gemini API
        system_instruction = None
        gemini_contents = []
        
        # Extract system message if present
        system_message = next((msg for msg in request.messages if msg.role == "system"), None)
        if system_message:
            system_instruction = {"parts": [{"text": system_message.content}]}
        
        # Convert messages to Gemini format
        for msg in request.messages:
            if msg.role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": msg.content}]})
        
        # Construct request body
        request_body = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": 512,
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        # Add system instruction if present
        if system_instruction:
            request_body["systemInstruction"] = system_instruction
        
        # Make API call to Google Gemini with timeout
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
        logger.info(f"[{request_id}] Calling Gemini API")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=request_body,
                headers={"Content-Type": "application/json"}
            )
            
            # Handle error responses
            if response.status_code != 200:
                error_body = response.text
                logger.error(f"[{request_id}] Gemini API Error Status: {response.status_code}")
                logger.error(f"[{request_id}] Gemini API Error Body: {error_body}")
                
                error_message = f"Gemini API request failed with status {response.status_code}"
                try:
                    error_json = response.json()
                    if "error" in error_json and "message" in error_json["error"]:
                        error_message = error_json["error"]["message"]
                except:
                    pass
                
                raise HTTPException(status_code=response.status_code, detail=error_message)
            
            # Process successful response
            data = response.json()
            
            # Extract response text
            assistant_response = "No response available from Gemini."
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    assistant_response = candidate["content"]["parts"][0].get("text", assistant_response)
            
            logger.info(f"[{request_id}] Successfully processed Gemini request")
            return {"message": assistant_response}
            
    except httpx.RequestError as e:
        logger.error(f"[{request_id}] Error processing Gemini request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request: {str(e)}")
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Gemini API request timed out")
        raise HTTPException(status_code=504, detail="Request to Gemini API timed out")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Text-to-Speech endpoint with fallback
@app.post("/api/text-to-speech", tags=["Speech"])
async def text_to_speech(request: TextToSpeechRequest):
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] POST /api/text-to-speech")
    
    try:
        # Truncate long text logs to prevent log bloat
        log_text = request.text[:50] + "..." if len(request.text) > 50 else request.text
        logger.info(f"[{request_id}] Processing text: {log_text}")

        # Try Edge TTS first
        try:
            # Select a voice
            voice = "en-US-GuyNeural"  # Default voice
            
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_filename = temp_file.name
            
            # Use edge-tts to generate speech
            communicate = edge_tts.Communicate(request.text, voice)
            await communicate.save(temp_filename)
            
            # Read the generated audio file
            with open(temp_filename, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            
            logger.info(f"[{request_id}] Successfully generated speech using Edge TTS")
            
            # Return the audio data
            return Response(
                content=audio_data,
                media_type="audio/mpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Content-Disposition": "attachment; filename=speech.mp3"
                }
            )
                
        except Exception as e:
            logger.warning(f"[{request_id}] Edge TTS failed, using Google TTS fallback: {str(e)}")
            # Fall back to Google TTS
            
            # Limit text length to 200 characters (Google Translate TTS limit)
            text = request.text
            if len(text) > 200:
                text = text[:197] + "..."
                logger.warning(f"[{request_id}] Text truncated to 200 characters for Google TTS")
            
            # URL encode the text
            encoded_text = urllib.parse.quote(text)
            
            # Use Google Translate TTS service
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=en&q={encoded_text}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                
                if response.status_code != 200:
                    logger.error(f"[{request_id}] Google TTS returned status {response.status_code}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to fetch from Google TTS"
                    )
                
                logger.info(f"[{request_id}] Successfully generated speech using Google TTS fallback")
                return Response(
                    content=response.content,
                    media_type="audio/mpeg",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                        "Content-Disposition": "attachment; filename=speech.mp3"
                    }
                )
            
    except httpx.RequestError as e:
        logger.error(f"[{request_id}] Error connecting to TTS service: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error connecting to TTS service: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Add this new endpoint for Google Text-to-Speech
@app.post("/api/google-tts", tags=["Speech"])
async def google_text_to_speech(request: TextToSpeechRequest):
    """Use Google Text-to-Speech API as an alternative"""
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] POST /api/google-tts")
    
    try:
        # This uses the free Google Translate TTS endpoint
        text = request.text
        
        # Limit text length to 200 characters (Google Translate TTS limit)
        if len(text) > 200:
            text = text[:197] + "..."
            logger.warning(f"[{request_id}] Text truncated to 200 characters for Google TTS")
        
        # URL encode the text
        encoded_text = urllib.parse.quote(text)
        
        # Use Google Translate TTS service
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=en&q={encoded_text}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            
            if response.status_code != 200:
                logger.error(f"[{request_id}] Google TTS returned status {response.status_code}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch from Google TTS"
                )
            
            logger.info(f"[{request_id}] Successfully generated speech using Google TTS")
            return Response(
                content=response.content,
                media_type="audio/mpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Content-Disposition": "attachment; filename=speech.mp3"
                }
            )
    except httpx.RequestError as e:
        logger.error(f"[{request_id}] Error connecting to Google TTS: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error connecting to Google TTS: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Google TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint for Edge TTS
@app.post("/api/edge-tts", tags=["Speech"])
async def edge_text_to_speech(request: TextToSpeechRequest):
    """Use Microsoft Edge TTS for higher quality speech synthesis"""
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] POST /api/edge-tts")
    
    try:
        # Log the text (truncated for logs)
        log_text = request.text[:50] + "..." if len(request.text) > 50 else request.text
        logger.info(f"[{request_id}] Processing text: {log_text}")
        
        # Default voice if not specified
        voice = "en-US-ChristopherNeural"  # Male voice
        # Alternative voices: "en-US-AriaNeural" (Female), "en-GB-SoniaNeural" (British Female)
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
        
        # Use edge-tts to generate speech
        communicate = edge_tts.Communicate(request.text, voice)
        await communicate.save(temp_filename)
        
        # Read the generated audio file
        with open(temp_filename, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        logger.info(f"[{request_id}] Successfully generated speech using Edge TTS")
        
        # Return the audio data
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Content-Disposition": "attachment; filename=speech.mp3"
            }
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Edge TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "environment": ENVIRONMENT,
        "gemini_api": "configured" if GEMINI_API_KEY else "not configured",
        "services": {
            "edge_tts": "available",
            "google_tts": "available"
        }
    }

# Define personas that can be used with the chat-to-speech endpoint
PERSONAS = {
    "professional": """You are a professional AI assistant with expertise in various fields. 
    Your responses are well-researched, accurate, and presented in a formal, business-like manner. 
    You use precise language, cite sources when appropriate, and maintain a helpful but authoritative tone.
    You avoid casual language and always prioritize factual correctness.""",
    
    "empathetic": """You are an empathetic AI assistant focused on understanding and addressing emotional needs.
    Your responses are warm, supportive, and compassionate. You acknowledge feelings, validate concerns,
    and offer encouragement. You use a gentle tone and thoughtful language that conveys care and understanding.
    You're patient and never dismissive of someone's emotions.""",
    
    "reality_check": """You are a reality-checking AI assistant who helps people evaluate situations objectively.
    Your responses are grounded in facts, logic, and critical thinking. You gently challenge unfounded assumptions,
    point out cognitive biases, and encourage evidence-based reasoning.
    You're diplomatic but honest, helping users see different perspectives without being judgmental.""",
    
    "creative": """You are a creative AI assistant who thinks outside the box.
    Your responses are imaginative, original, and inspiring. You offer unique perspectives, 
    innovative solutions, and artistic ideas. You use colorful language, metaphors, and storytelling techniques.
    You encourage experimentation and playful thinking.""",
    
    "technical": """You are a technical AI assistant with deep knowledge of computer science, programming, and technology.
    Your responses are precise, detailed, and technically accurate. You use proper terminology,
    provide code examples when helpful, and explain complex concepts clearly.
    You focus on best practices and efficient solutions to technical problems."""
}

# Update the chat-to-speech endpoint with better voice selection and fallback
@app.post(
    "/api/chat-to-speech", 
    response_model=Dict[str, Any], 
    tags=["AI"],
    summary="Get both text and speech response with persona selection",
    description="""
    Send a chat message and receive both text and speech responses using a selected persona.
    
    To specify a persona, include a system message with content "persona: {persona_id}" where
    persona_id is one of: professional, empathetic, reality_check, creative, technical.
    
    Example request body:
    ```json
    {
      "messages": [
        {
          "role": "system",
          "content": "persona: technical"
        },
        {
          "role": "user",
          "content": "Explain how APIs work in simple terms"
        }
      ]
    }
    ```
    
    The response includes:
    - message: The text response from the AI
    - audio: Base64-encoded MP3 audio of the response
    - persona: The persona used for the response
    - voice: The voice used for text-to-speech
    """
)
async def chat_to_speech(
    request: ChatRequest,
    api_key: str = Depends(validate_gemini_api_key)
):
    """
    Get both text and speech response in one request using Edge TTS.
    
    This endpoint:
    1. Sends the chat messages to Gemini API with an optional persona
    2. Gets the text response
    3. Converts that response to speech using Edge TTS
    4. Returns both text and audio (base64 encoded)
    
    You can specify a persona by including a system message with the format:
    "persona: [professional|empathetic|reality_check|creative|technical]"
    """
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] POST /api/chat-to-speech")
    
    try:
        # Extract system message and check for persona
        system_instruction = None
        gemini_contents = []
        selected_persona = None
        
        # Process messages
        for msg in request.messages:
            if msg.role == "system":
                # Check for persona selection
                if msg.content.startswith("persona:"):
                    persona_id = msg.content.split(":", 1)[1].strip().lower()
                    if persona_id in PERSONAS:
                        selected_persona = persona_id
                        system_instruction = {"parts": [{"text": PERSONAS[persona_id]}]}
                        logger.info(f"[{request_id}] Using persona: {selected_persona}")
                    else:
                        # Custom system message
                        system_instruction = {"parts": [{"text": msg.content}]}
                else:
                    # Custom system message
                    system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": msg.content}]})
        
        # Construct request body
        request_body = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": 512,
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        # Add system instruction if present
        if system_instruction:
            request_body["systemInstruction"] = system_instruction
        
        # Make API call to Google Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
        logger.info(f"[{request_id}] Calling Gemini API")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                json=request_body,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"[{request_id}] Gemini API returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from Gemini API: {response.text}"
                )
            
            # Process successful response
            data = response.json()
            
            # Extract response text
            assistant_response = "No response available from Gemini."
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    assistant_response = candidate["content"]["parts"][0].get("text", assistant_response)
            
            logger.info(f"[{request_id}] Successfully processed Gemini request")
            
            # Define a list of voices to try in order of preference
            voices_to_try = []
            
            # Select voice based on persona
            if selected_persona == "empathetic":
                voices_to_try = ["en-US-JennyNeural", "en-US-SaraNeural", "en-GB-SoniaNeural"]
            elif selected_persona == "professional":
                voices_to_try = ["en-US-GuyNeural", "en-US-RogerNeural", "en-GB-RyanNeural"]
            elif selected_persona == "reality_check":
                voices_to_try = ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-LibbyNeural"]
            elif selected_persona == "creative":
                voices_to_try = ["en-US-DavisNeural", "en-US-JasonNeural", "en-GB-ThomasNeural"]
            elif selected_persona == "technical":
                voices_to_try = ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-RyanNeural"]
            else:
                # Default voices
                voices_to_try = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"]
            
            # Add a fallback voice that should work in most cases
            voices_to_try.append("en-US-EricNeural")
            
            # Try each voice until one works
            for voice_index, voice in enumerate(voices_to_try):
                try:
                    logger.info(f"[{request_id}] Attempting TTS with voice: {voice} (attempt {voice_index+1}/{len(voices_to_try)})")
                    
                    # Create a temporary file to store the audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        temp_filename = temp_file.name
                    
                    # Log the text length
                    logger.info(f"[{request_id}] Text length for TTS: {len(assistant_response)} characters")
                    
                    # Check if text is too long and truncate if necessary
                    max_tts_length = 3000  # Edge TTS has limitations on text length
                    truncated = False
                    tts_text = assistant_response
                    
                    if len(tts_text) > max_tts_length:
                        tts_text = tts_text[:max_tts_length] + "..."
                        truncated = True
                        logger.warning(f"[{request_id}] Text truncated for TTS from {len(assistant_response)} to {max_tts_length} characters")
                    
                    # Use edge-tts to generate speech
                    communicate = edge_tts.Communicate(tts_text, voice)
                    
                    # Set a timeout for the TTS operation
                    try:
                        # Use asyncio.wait_for to set a timeout
                        await asyncio.wait_for(communicate.save(temp_filename), timeout=15.0)
                        
                        # Check if the file was created and has content
                        if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                            # Read the generated audio file
                            with open(temp_filename, "rb") as audio_file:
                                audio_data = audio_file.read()
                            
                            # Clean up the temporary file
                            os.unlink(temp_filename)
                            
                            # Return both text and audio
                            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                            
                            logger.info(f"[{request_id}] Successfully generated chat and speech response with Edge TTS")
                            
                            return {
                                "message": assistant_response,
                                "audio": audio_base64,
                                "persona": selected_persona or "custom",
                                "voice": voice,
                                "truncated": truncated
                            }
                        else:
                            logger.error(f"[{request_id}] Edge TTS generated an empty audio file with voice {voice}")
                            # Continue to the next voice
                            continue
                            
                    except asyncio.TimeoutError:
                        logger.error(f"[{request_id}] Edge TTS operation timed out with voice {voice}")
                        # Continue to the next voice
                        continue
                        
                except edge_tts.exceptions.NoAudioReceived as e:
                    logger.error(f"[{request_id}] Edge TTS NoAudioReceived error with voice {voice}: {str(e)}")
                    # Continue to the next voice
                    continue
                except edge_tts.exceptions.EdgeTTSException as e:
                    logger.error(f"[{request_id}] Edge TTS specific error with voice {voice}: {str(e)}")
                    # Continue to the next voice
                    continue
                except Exception as e:
                    logger.error(f"[{request_id}] Unexpected error with voice {voice}: {str(e)}", exc_info=True)
                    # Continue to the next voice
                    continue
                finally:
                    # Clean up temporary file if it exists
                    if 'temp_filename' in locals() and os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
            
            # If all Edge TTS voices fail, fall back to Google TTS
            logger.warning(f"[{request_id}] All Edge TTS voices failed, falling back to Google TTS")
            
            try:
                # Limit text length to 200 characters (Google Translate TTS limit)
                fallback_text = assistant_response
                if len(fallback_text) > 200:
                    fallback_text = fallback_text[:197] + "..."
                    logger.warning(f"[{request_id}] Text truncated to 200 characters for Google TTS fallback")
                
                # URL encode the text
                encoded_text = urllib.parse.quote(fallback_text)
                
                # Use Google Translate TTS service
                url = f"https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=en&q={encoded_text}"
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        # Return both text and audio
                        audio_base64 = base64.b64encode(response.content).decode('utf-8')
                        
                        logger.info(f"[{request_id}] Successfully generated speech using Google TTS fallback")
                        
                        return {
                            "message": assistant_response,
                            "audio": audio_base64,
                            "persona": selected_persona or "custom",
                            "voice": "Google TTS (fallback)",
                            "truncated": len(fallback_text) < len(assistant_response),
                            "fallback": True
                        }
                    else:
                        logger.error(f"[{request_id}] Google TTS fallback failed with status {response.status_code}")
                        # If Google TTS also fails, return just the text response
                        return {
                            "message": assistant_response,
                            "audio": "",
                            "persona": selected_persona or "custom",
                            "voice": "Failed",
                            "truncated": False,
                            "error": "All TTS services failed. Please try again later."
                        }
            except Exception as fallback_error:
                logger.error(f"[{request_id}] Google TTS fallback error: {str(fallback_error)}", exc_info=True)
                # If all TTS options fail, return just the text response
                return {
                    "message": assistant_response,
                    "audio": "",
                    "persona": selected_persona or "custom",
                    "voice": "Failed",
                    "truncated": False,
                    "error": "All TTS services failed. Please try again later."
                }
            
    except httpx.RequestError as e:
        logger.error(f"[{request_id}] Error connecting to Gemini API: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error connecting to Gemini API: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Update the personas endpoint with better documentation
@app.get(
    "/api/personas", 
    tags=["AI"],
    summary="Get available personas",
    description="Returns a list of all available personas that can be used with the chat-to-speech endpoint."
)
async def get_personas():
    """
    Get a list of available personas that can be used with the chat-to-speech endpoint.
    
    Each persona has:
    - id: The identifier to use in API requests
    - name: A human-readable name
    - description: A brief description of the persona's characteristics
    - voice: The Edge TTS voice used for this persona
    
    To use a persona, include a system message with content "persona: {persona_id}" in your request.
    """
    # Create a simplified version of the personas for the response
    persona_descriptions = {
        "professional": "A formal, business-like assistant with precise language and factual correctness.",
        "empathetic": "A warm, supportive assistant focused on emotional needs and compassionate responses.",
        "reality_check": "An objective assistant who helps evaluate situations with facts, logic, and critical thinking.",
        "creative": "An imaginative assistant who offers unique perspectives and innovative solutions.",
        "technical": "A precise assistant with deep technical knowledge, providing detailed explanations and code examples."
    }
    
    # Map personas to their voice names
    persona_voices = {
        "professional": "Guy (Professional male voice)",
        "empathetic": "Jenny (Warm female voice)",
        "reality_check": "Aria (Clear, neutral female voice)",
        "creative": "Davis (Expressive male voice)",
        "technical": "Tony (Precise male voice)"
    }
    
    # Return the personas with their descriptions and voices
    return {
        "personas": [
            {
                "id": persona_id,
                "name": persona_id.replace("_", " ").title(),
                "description": description,
                "voice": persona_voices[persona_id]
            }
            for persona_id, description in persona_descriptions.items()
        ]
    }

# Add this new endpoint to list all available Edge TTS voices
@app.get(
    "/api/edge-tts-voices", 
    tags=["Speech"],
    summary="List all available Edge TTS voices",
    description="Returns a list of all available voices for Edge TTS"
)
async def list_edge_tts_voices():
    """
    Get a list of all available voices for Edge TTS.
    
    This endpoint returns all voices supported by the Edge TTS service,
    including their language, gender, and full name.
    """
    try:
        # Get all available voices
        voices = await edge_tts.list_voices()
        
        # Format the response
        formatted_voices = []
        for voice in voices:
            formatted_voices.append({
                "name": voice["ShortName"],
                "gender": voice["Gender"],
                "locale": voice["Locale"],
                "display_name": voice["DisplayName"],
                "local_name": voice["LocalName"],
                "status": voice.get("Status", "GA"),  # GA = Generally Available
                "suggested_for": "technical" if "Neural" in voice["ShortName"] and "US" in voice["ShortName"] else None
            })
        
        # Sort by locale and then by name
        formatted_voices.sort(key=lambda x: (x["locale"], x["name"]))
        
        return {
            "voices": formatted_voices,
            "count": len(formatted_voices)
        }
    except Exception as e:
        logger.error(f"Error listing Edge TTS voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")









