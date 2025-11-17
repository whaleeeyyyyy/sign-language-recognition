from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from utils.mediapipe_processor import MediaPipeProcessor
import tensorflow as tf
import os

app = FastAPI(title="Sign Language Recognition API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe processor
mp_processor = MediaPipeProcessor()

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gesture_model.h5")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
        print("Please train the model first using the training scripts.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define gesture labels (should match training order)
GESTURE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                  'U', 'V', 'W', 'X', 'Y', 'Z']

class ImageRequest(BaseModel):
    image: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
async def root():
    return {
        "message": "Sign Language Recognition API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = request.image.split(",")[1] if "," in request.image else request.image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Extract keypoints using MediaPipe
        keypoints = mp_processor.extract_keypoints(image)
        
        if keypoints is None:
            return PredictionResponse(
                prediction="No hand detected",
                confidence=0.0
            )
        
        # Check if model is loaded
        if model is None:
            return PredictionResponse(
                prediction="Model not loaded",
                confidence=0.0
            )
        
        # Reshape keypoints for model input
        keypoints_array = np.array(keypoints).reshape(1, -1)
        
        # Make prediction
        predictions = model.predict(keypoints_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Get predicted gesture label
        predicted_gesture = GESTURE_LABELS[predicted_index] if predicted_index < len(GESTURE_LABELS) else "Unknown"
        
        return PredictionResponse(
            prediction=predicted_gesture,
            confidence=confidence
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mediapipe_ready": mp_processor is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)