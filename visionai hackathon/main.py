import uvicorn
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import *
import io

# --- Add these imports for the fix ---
import pathlib
import sys

# --- 1. SETUP THE FASTAPI APP ---

app = FastAPI(
    title="Plant Disease Detection API",
    description="API to predict plant diseases from images using a trained fastai model."
)

# --- 2. ADD CORS MIDDLEWARE ---
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. LOAD MODEL AND METADATA ON STARTUP ---

# --- BUG FIX 1: Corrected model filename ---
MODEL_PATH = Path("plant_disease_model_resnet34.pkl")
DISEASE_INFO_PATH = Path("disease_info.json")

learn = None
disease_info = None

@app.on_event("startup")
async def startup_event():
    """
    On startup, load the ML model and the disease information JSON.
    """
    global learn, disease_info
    
    # Load Disease Info
    if not DISEASE_INFO_PATH.exists():
        raise RuntimeError(f"Disease info file not found at {DISEASE_INFO_PATH}")
    with open(DISEASE_INFO_PATH) as f:
        disease_info = json.load(f)

    # Load Model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    
    # --- Platform-specific patch for Windows ---
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        learn = load_learner(MODEL_PATH)
        print("--- Model loaded successfully ---")
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {e}")


# --- 4. DEFINE API ENDPOINTS ---

@app.get("/")
def read_root():
    """Welcome message for the API."""
    return {"message": "Welcome to the Plant Disease Detection API!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the disease from an uploaded image.
    """
    if learn is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again in a moment.")

    try:
        # Read image file
        image_bytes = await file.read()
        img = PILImage.create(io.BytesIO(image_bytes))

        # Perform prediction
        pred_class, pred_idx, outputs = await run_in_threadpool(learn.predict, img)
        
        confidence = outputs[pred_idx].item()
        print(confidence)
        # --- IMPROVEMENT: Handle low-confidence predictions ---
        if confidence < 0.7:  # 50% threshold
            return {
                "predicted_class": "Uncertain",
                "confidence": f"{confidence:.4f}",
                "details": {
                    "name_of_species": "Unknown",
                    "diseased_or_healthy": "Diseased",
                    "disease_name": "Could not determine with high confidence.",
                    "cause": "The model is uncertain about the prediction. This could be due to a poor quality image or an unfamiliar plant/disease.",
                    "prevention": "Please try again with a clearer image, ensuring the affected area is well-lit and in focus.",
                    "treatment": "N/A"
                }
            }
        
        # If confidence is high, proceed as normal
        details = disease_info.get(pred_class)

        # Handle cases where the predicted class is not in our JSON file
        if details is None:
            details = {
                # --- IMPROVEMENT: Use consistent keys -- -
                "name_of_species": "Unknown",
                "diseased_or_healthy": "Diseased",
                "disease_name": pred_class.replace("_", " "),
                "cause": "Information for this specific disease is not yet available in our database.",
                "prevention": "General advice: Isolate the plant and consult a local agricultural expert.",
                "treatment": "General advice: Isolate the plant, remove affected leaves, and consult a local agricultural expert for further guidance."
            }
        
        print(f"Predicted: {pred_class} with confidence {confidence:.4f}")
        return {
            "predicted_class": pred_class,
            "confidence": f"{confidence:.4f}",
            "details": details
        }

    except Exception as e:
        # Log the full error to the console for debugging
        print(f"--- DETAILED PREDICTION ERROR ---: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")


# --- 5. MAKE THE APP RUNNABLE ---
# --- BUG FIX 2: Corrected dunder name ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)