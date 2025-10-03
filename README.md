Plant Disease Detection API
This repository contains the backend server for the Crop Doc application, an intelligent plant health monitoring system. The API uses a trained deep learning model to classify plant leaf images, identify diseases, and provide treatment recommendations.

This service is built with FastAPI and uses a fastai model for predictions.

Features
Fast Predictions: Leverages FastAPI for high-performance, asynchronous request handling.

Model Integration: Loads a pre-trained fastai (.pkl) model on startup for efficient inference.

Rich Responses: Provides detailed information about the predicted disease, including cause and treatment, from a supplementary JSON file.

Error Handling: Gracefully handles low-confidence predictions and cases where a disease is not in the database.

CORS Enabled: Configured to accept requests from any origin, making frontend integration seamless.

Interactive Docs: Automatically generates API documentation at the /docs endpoint.

Tech Stack
Framework: FastAPI

ML Library: fastai / PyTorch

Server: Uvicorn

Language: Python 3.9+

Setup and Installation
Follow these steps to get the backend server running on your local machine.

1. Prerequisites
Python 3.9 or newer installed on your system.

pip (Python package installer).

2. Clone the Repository
git clone <your-repository-url>
cd <your-repository-folder>

3. Install Dependencies
It's highly recommended to use a virtual environment to keep dependencies isolated.

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

4. Add Required Files
Place the following files in the root directory of the backend project (the same folder as main.py):

plant_disease_model_resnet34.pkl: The exported fastai learner model file.

disease_info.json: The JSON file containing detailed information about each disease class.

The server will fail to start if these files are not present.

Running the Server
Once the setup is complete, you can start the API server with the following command:

python main.py

Alternatively, you can use Uvicorn directly, which is great for development as it auto-reloads on code changes:

uvicorn main:app --reload

The server will start on http://localhost:8000.

API Endpoints
The API provides the following endpoints.

Root
Endpoint: /

Method: GET

Description: A simple health check endpoint to confirm that the API is running.

Success Response:

{
  "message": "Welcome to the Plant Disease Prediction API! Go to /docs for more info."
}

Predict Disease
Endpoint: /predict

Method: POST

Description: The main endpoint that accepts an image of a plant leaf and returns a prediction. The image must be sent as multipart/form-data.

Request Body:

file: The image file (UploadFile).

Success Response (Code 200):

{
  "predicted_class": "Tomato___Late_blight",
  "confidence": "0.9876",
  "details": {
    "status": "Diseased",
    "diseaseName": "Tomato Late Blight",
    "cause": "Caused by the fungus-like organism Phytophthora infestans.",
    "treatment": "Apply fungicides containing mancozeb, chlorothalonil, or copper-based compounds."
  }
}

Error Response (Code 500): If an internal error occurs during prediction.

Interactive Documentation
For easy testing, navigate to http://localhost:8000/docs in your browser after starting the server. This will open the interactive Swagger UI where you can upload an image and test the /predict endpoint directly.
