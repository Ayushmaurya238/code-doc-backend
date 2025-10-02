import uvicorn
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
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
        raise FileNotFoundError(f"'{DISEASE_INFO_PATH}' not found. Please make sure the file is in the same directory.")
    with open(DISEASE_INFO_PATH) as f:
        disease_info = json.load(f)
    print("Disease info loaded successfully.")

    # --- FIX FOR WINDOWS USERS LOADING A LINUX-TRAINED MODEL---
    temp_path_patch = None
    if sys.platform == "win32":
        temp_path_patch = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    # ---------------------------------------------------------
    
    try:
        # Load Fastai Learner Model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"'{MODEL_PATH}' not found. Please place your exported .pkl model file in the same directory.")
        
        learn = load_learner(MODEL_PATH, cpu=True)
        print("Fastai learner loaded successfully.")

    finally:
        # --- IMPORTANT: Revert the patch after loading the model ---
        if sys.platform == "win32" and temp_path_patch is not None:
            pathlib.PosixPath = temp_path_patch
        # ---------------------------------------------------------


# --- 4. DEFINE API ENDPOINTS ---

@app.get("/")
async def root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Plant Disease Prediction API! Go to /docs for more info."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, predicts the disease, and returns detailed information.
    """
    if learn is None or disease_info is None:
        raise HTTPException(status_code=503, detail="Server is not ready: Model or disease data is not loaded.")

    try:
        # Read image file bytes
        image_bytes = await file.read()
        
        # Create a PILImage from bytes
        img = PILImage.create(io.BytesIO(image_bytes))

        # --- MORE ROBUST PREDICTION METHOD ---
        # Create a dataloader with the single image to apply all necessary transforms
        dl = learn.dls.test_dl([img])
        
        # Get predictions
        preds, _ = learn.get_preds(dl=dl)
        
        # The prediction is the index of the highest probability
        pred_idx = preds[0].argmax().item()
        pred_class = learn.dls.vocab[pred_idx]
        confidence = float(preds[0][pred_idx])
        # ------------------------------------
        
        # Check if confidence is below the threshold
        if confidence < 0.70:
            print(f"Low confidence prediction ({confidence:.4f}). Returning 'Unidentified'.")
            return {
                "predicted_class": "Unidentified",
                "confidence": f"{confidence:.4f}",
                "details": {
                    "status": "Unknown",
                    "diseaseName": "Unidentified",
                    "cause": "The model could not identify the disease with high confidence.",
                    "treatment": "Please provide a clearer, well-lit photograph of the affected area. Ensure the leaf is the main subject and is in focus."
                }
            }
        
        # If confidence is high, proceed as normal
        details = disease_info.get(pred_class)

        # Handle cases where the predicted class is not in our JSON file
        if details is None:
            details = {
                "status": "Diseased",
                "diseaseName": pred_class.replace("_", " "),
                "cause": "Information for this specific disease is not yet available in our database.",
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
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

