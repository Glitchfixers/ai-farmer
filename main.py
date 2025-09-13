from fastapi import FastAPI
from disease_detection_api.main import app as disease_app
from rag_api.main import app as rag_app
from tts_api.main import app as tts_app
from voice_transcription_api.main import app as vt_app

app = FastAPI(title="AI Farmer API")

app.mount("/disease", disease_app)
app.mount("/rag", rag_app)
app.mount("/tts", tts_app)
app.mount("/voice", vt_app)

@app.get("/")
def root():
    return {"message": "AI Farmer API is live"}