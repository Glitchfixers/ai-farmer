# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from PIL import Image
# from torchvision import transforms
# import torch
# import os

# from model_def import ImprovedCNNViTHybrid
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



# # ---- CONFIG ----
# MODEL_PATH = "all_crops_hybrid_best_model.pth"
# CLASSES_PATH = "classes.txt"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load class names
# with open(CLASSES_PATH, "r") as f:
#     CLASS_NAMES = [line.strip() for line in f]

# num_classes = len(CLASS_NAMES)

# # Load model
# model = ImprovedCNNViTHybrid(num_classes=num_classes, pretrained=False)
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# state_dict = checkpoint.get("model_state_dict", checkpoint)  # handle saved dict or direct weights
# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()

# # Define image transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# app = FastAPI()

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         img = Image.open(file.file).convert("RGB")
#         img_t = transform(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             outputs = model(img_t)
#             _, pred = torch.max(outputs, 1)
#             label = CLASS_NAMES[pred.item()]
#         return JSONResponse({"predicted_disease": label})
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)

# @app.get("/")
# def home():
#     return {"message": "Plant Disease Detection API Running"}



# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from PIL import Image
# from torchvision import transforms
# import torch
# import os

# from model_def import ImprovedCNNViTHybrid
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # ---- CONFIG ----
# MODEL_PATH = "all_crops_hybrid_best_model.pth"
# CLASSES_PATH = "classes.txt"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load class names
# with open(CLASSES_PATH, "r") as f:
#     CLASS_NAMES = [line.strip() for line in f]

# num_classes = len(CLASS_NAMES)

# # Load model
# model = ImprovedCNNViTHybrid(num_classes=num_classes, pretrained=False)
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# state_dict = checkpoint.get("model_state_dict", checkpoint)  # handle saved dict or direct weights
# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()

# # Define image transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # ---- Translation Model ----
# TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"  # English → Hindi
# tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
# translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)

# def translate_text(text, target_lang="hi"):
#     inputs = tokenizer(text, return_tensors="pt", padding=True)
#     outputs = translation_model.generate(**inputs, max_length=256)
#     translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return translated

# # ---- FastAPI ----
# app = FastAPI()

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         img = Image.open(file.file).convert("RGB")
#         img_t = transform(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             outputs = model(img_t)
#             _, pred = torch.max(outputs, 1)
#             label_en = CLASS_NAMES[pred.item()]

#         # Translate label
#         label_translated = translate_text(label_en)

#         return JSONResponse({
#             "predicted_disease_en": label_en,
#             "predicted_disease_translated": label_translated
#         })

#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)

# @app.get("/")
# def home():
#     return {"message": "Plant Disease Detection API Running"}

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
import torch
import os
import requests

from model_def import ImprovedCNNViTHybrid

# ---- CONFIG ----
MODEL_PATH = "all_crops_hybrid_best_model.pth"
CLASSES_PATH = "classes.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]

num_classes = len(CLASS_NAMES)
if num_classes == 0:
    raise ValueError("classes.txt is empty! Please provide class names.")

# Load model
model = ImprovedCNNViTHybrid(num_classes=num_classes, pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# IndicTrans2 URL
INDIC_TRANS_URL = "http://localhost:8000/translate"

def translate_text(text: str, tgt_lang: str = "hin_Deva"):
    payload = {"text": text, "src_lang": "eng_Latn", "tgt_lang": tgt_lang}
    response = requests.post(INDIC_TRANS_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("translation", text)
    return text  # Fallback if translation fails

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tgt_lang: str = Query("hin_Deva", description="Target language code (e.g., hin_Deva for Hindi)")
):
    try:
        img = Image.open(file.file).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            _, pred = torch.max(outputs, 1)
            label_en = CLASS_NAMES[pred.item()]

        # Translate if requested
        label_translated = translate_text(label_en, tgt_lang)

        return JSONResponse({
            "predicted_disease_en": label_en,
            "predicted_disease_translated": label_translated,
            "target_language": tgt_lang
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def home():
    return {"message": "Disease Detection API Running"}