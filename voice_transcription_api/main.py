from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import pipeline
import requests
import os
import tempfile

app = FastAPI()

# FFmpeg should be available in PATH for deployment

# Whisper for transcription (small model, supports Indic languages)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

# IndicTrans2 URL
INDIC_TRANS_URL = "http://localhost:8000/translate"

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name
        
        # Transcribe (Whisper should use the configured FFmpeg)
        result = whisper(audio_path)
        transcribed = result["text"]
        
        # Translate to English
        payload = {
            "text": transcribed,
            "src_lang": "hin_Deva",  # Default; adjust dynamically if needed
            "tgt_lang": "eng_Latn"
        }
        response = requests.post(INDIC_TRANS_URL, json=payload)
        if response.status_code != 200:
            raise ValueError("Translation failed")
        english_text = response.json().get("translation", transcribed)
        
        os.remove(audio_path)  # Cleanup
        return JSONResponse({
            "transcribed_text": transcribed,
            "english_text": english_text
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def home():
    return {"message": "Voice Transcription API Running"}