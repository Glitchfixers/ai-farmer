# AI Farmer API

A comprehensive AI-powered agricultural assistance API that provides disease detection, RAG-based Q&A, text-to-speech, and voice transcription services for farmers.

## 🌟 Features

### 1. Disease Detection API (`/disease`)
- **Plant Disease Classification**: Uses a hybrid CNN-ViT model to detect diseases in crop images
- **Multi-language Support**: Provides disease predictions in both English and local languages
- **Supported Crops**: Banana, Cardamom, Coconut, Coffee, Mango, Pepper, Potato, Rice, Rubber, Wheat
- **Real-time Translation**: Integrates with IndicTrans2 for local language support

### 2. RAG (Retrieval-Augmented Generation) API (`/rag`)
- **Knowledge Base**: Built on agricultural documents, government schemes, and crop calendars
- **Intelligent Q&A**: Uses Groq's Llama 3.3 70B model for accurate responses
- **Document Processing**: Supports PDF, DOCX, CSV, Excel, and text files
- **Vector Search**: FAISS-based semantic search for relevant information

### 3. Text-to-Speech API (`/tts`)
- **Multi-language TTS**: Supports multiple Indian languages
- **High-quality Audio**: Uses Coqui TTS for natural speech synthesis
- **Customizable Voice**: Multiple voice options available

### 4. Voice Transcription API (`/voice`)
- **Speech-to-Text**: Whisper-based transcription with high accuracy
- **Language Detection**: Automatic language detection and transcription
- **Translation Support**: Transcribes and translates to English

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-farmer-api.git
cd ai-farmer-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

5. **Download model files**
   - Place `all_crops_hybrid_best_model.pth` in `disease_detection_api/`
   - Place `classes.txt` in `disease_detection_api/`

6. **Run the API**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

### Disease Detection
```bash
POST /disease/predict
Content-Type: multipart/form-data

# Parameters:
# - file: Image file (jpg, jpeg, png)
# - tgt_lang: Target language code (default: hin_Deva)
```

**Response:**
```json
{
  "predicted_disease_en": "Coffee Leaf Rust",
  "predicted_disease_translated": "कॉफी पत्ती जंग",
  "target_language": "hin_Deva"
}
```

### RAG Query
```bash
POST /rag/rag
Content-Type: application/json

{
  "query": "What are the symptoms of coffee leaf rust?"
}
```

**Response:**
```json
{
  "answer": "Coffee leaf rust symptoms include...",
  "sources": ["path/to/relevant/document.pdf"]
}
```

### Text-to-Speech
```bash
POST /tts/synthesize
Content-Type: application/json

{
  "text": "Hello, this is a test",
  "language": "en"
}
```

### Voice Transcription
```bash
POST /voice/transcribe
Content-Type: multipart/form-data

# Parameters:
# - file: Audio file (wav, mp3, m4a)
```

## 🏗️ Architecture

```
ai_farmer_api/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── disease_detection_api/  # Disease detection service
│   ├── main.py
│   ├── model_def.py
│   ├── all_crops_hybrid_best_model.pth
│   └── classes.txt
├── rag_api/               # RAG service
│   └── main.py
├── tts_api/               # Text-to-speech service
│   └── main.py
└── voice_transcription_api/ # Voice transcription service
    └── main.py
```

## 🔧 Configuration

### Model Configuration
- **Disease Detection**: Hybrid CNN-ViT model with ResNet50 backbone
- **RAG**: Sentence transformers with FAISS vector store
- **TTS**: Coqui TTS with multiple language support
- **ASR**: OpenAI Whisper small model

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```

## 🚀 Deployment

### Render Deployment
This API is configured for easy deployment on Render:

1. **Connect GitHub Repository**
2. **Set Environment Variables** in Render dashboard
3. **Deploy** - Render will automatically build and deploy

### Docker Deployment
```bash
# Build image
docker build -t ai-farmer-api .

# Run container
docker run -p 8000:8000 ai-farmer-api
```

## 📊 Performance

- **Disease Detection**: ~2-3 seconds per image
- **RAG Query**: ~3-5 seconds per query
- **TTS**: ~1-2 seconds per sentence
- **Voice Transcription**: ~2-4 seconds per audio file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Groq** for fast LLM inference
- **Coqui TTS** for text-to-speech capabilities
- **OpenAI** for Whisper ASR model
- **FastAPI** for the web framework

## 📞 Support

For support and questions:
- Create an issue in this repository
- Contact: [your-email@example.com]

---

**Made with ❤️ for Indian Farmers**

