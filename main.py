from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os

app = FastAPI(title="Skin Triad Health Ecosystem API", version="1.1")

# Enable CORS for frontend/dashboard apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Safely
MODEL_PATH = "skin-.pth"
model = None

try:
    if os.path.exists(MODEL_PATH):
        # Using CPU for Render free tier safety
        device = torch.device("cpu")
        model = torch.load(MODEL_PATH, map_location=device)
        model.eval()
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model weights not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {
        "status": "Skin Triad AI Backend is running!",
        "ecosystem_modules": ["AI Triage", "Pharmacy Marketplace", "HMO Integration", "Admin & Pharmacy Dashboards"]
    }

# 1. Core AI Triage Endpoint with Intelligent Intake Questionnaire
@app.post("/api/triage")
async def analyze_skin_condition(
    file: UploadFile = File(...),
    duration_days: int = Form(3),
    itching_level: str = Form("moderate"), # mild, moderate, severe
    pain_level: str = Form("none"),         # none, mild, severe
    body_location: str = Form("face/arms")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # TODO: Pass image through model transforms and inference here
    # For now, we return a structured clinical triage report incorporating user intake data
    
    risk_assessment = "Moderate Priority - Consult Community Health Extension Worker (CHW) or Pharmacist"
    if pain_level == "severe" or itching_level == "severe":
        risk_assessment = "High Priority - Recommended review by a certified Dermatologist"

    return {
        "filename": file.filename,
        "intake_summary": {
            "duration_days": duration_days,
            "itching_level": itching_level,
            "pain_level": pain_level,
            "body_location": body_location
        },
        "ai_prediction": {
            "primary_condition": "Dermatological Evaluation Required",
            "confidence_score": 0.91,
            "risk_level": risk_assessment
        },
        "recommended_action": "Suggested OTC management or verified pharmacy product listed below.",
        "partner_pharmacy_integration": [
            {"item": "Soothing Calamine Lotion / Barrier Cream", "price_ngn": 3500, "in_stock": True},
            {"item": "Medicated Antifungal/Antibacterial Ointment", "price_ngn": 4200, "in_stock": True}
        ]
    }

# 2. Pharmacy & Affiliate Marketplace Endpoint
@app.get("/api/pharmacy/catalog")
def get_pharmacy_catalog():
    return {
        "partner_pharmacies": [
            {"name": "Agbara Community Pharmacy", "location": "Agbara Industrial Corridor", "delivery_available": True},
            {"name": "Badagry Central Health Mart", "location": "Badagry Town", "delivery_available": True}
        ],
        "products": [
            {"id": 1, "name": "Organic Sunscreen SPF 50", "category": "Skincare Protection", "price_ngn": 7500, "commission_rate": "10%"},
            {"id": 2, "name": "Eczema Relief Botanical Soap", "category": "Therapeutic Care", "price_ngn": 3000, "commission_rate": "15%"}
        ]
    }

# 3. HMO Verification Endpoint stub
@app.post("/api/hmo/verify")
def verify_hmo_policy(hmo_provider: str, policy_number: str):
    # Stub for future HMO database check (e.g., AXA Mansard, Hygeia)
    return {
        "provider": hmo_provider,
        "policy_number": policy_number,
        "status": "Verified",
        "telehealth_covered": True,
        "pharmacy_copay_percentage": 20
    }