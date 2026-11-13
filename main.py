from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Skin Triage AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Re-initialize the MobileNetV2 architecture structure
model = models.mobilenet_v2(weights=None)
num_classes = 3 # Matches the number of classes trained in Colab
model.classifier[1] = nn.Sequential(
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

# 2. Load your saved .pth weights into the model
try:
    model.load_state_dict(torch.load("skin_model.pth", map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
    print("Loaded skin_model.pth successfully!")
except Exception as e:
    print(f"Warning: Could not load model weights: {e}")

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"status": "Skin Triage AI Backend is running with custom model weights!"}

@app.post("/predict")
async def predict_skin_condition(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Run real inference using your loaded model
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
    
    # Map class index to human-readable text
    classes = ["Common Acne / Mild Inflammation", "Eczema / Dry Irritation", "General Rash / Contact Dermatitis"]
    condition_name = classes[predicted_class.item()]
    conf_score = f"{confidence.item() * 100:.1f}%"

    triage_result = {
        "condition": condition_name,
        "confidence": conf_score,
        "advice": "Keep the area clean, avoid harsh chemical soaps, and apply a gentle, fragrance-free moisturizer. If symptoms persist or worsen over 5 days, consult a physician.",
        "escalation": "Low urgency. Monitor for changes."
    }
    
    return triage_result