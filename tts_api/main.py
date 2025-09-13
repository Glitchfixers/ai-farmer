from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import tempfile
from TTS.api import TTS

app = FastAPI()

# Initialize TTS
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
except Exception as e:
    print(f"TTS initialization failed: {e}")
    tts = None

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    try:
        if tts is None:
            raise HTTPException(status_code=500, detail="TTS not initialized")
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            output_path = tmp.name
        
        # Generate speech
        tts.tts_to_file(text=request.text, file_path=output_path, language=request.language)
        
        # Read the generated audio file
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        # Clean up
        os.remove(output_path)
        
        return {
            "audio_data": audio_data.hex(),  # Convert to hex for JSON response
            "text": request.text,
            "language": request.language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Text-to-Speech API Running"}