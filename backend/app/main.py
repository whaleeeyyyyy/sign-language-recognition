# FastAPI - Framework for building APIs (like Flask but faster)
from fastapi import FastAPI, HTTPException

# CORS - Allows frontend (different port) to talk to backend
from fastapi.middleware.cors import CORSMiddleware

# Pydantic - Validates data types (type checking)
from pydantic import BaseModel

# Standard Python libraries
import base64  # Convert base64 strings to images
import cv2     # OpenCV - computer vision library
import numpy as np  # NumPy - for arrays and math
import os      # Operating system functions (file paths, etc.)

# Our custom MediaPipe processor
from utils.mediapipe_processor import MediaPipeProcessor

# TensorFlow - Machine learning library
import tensorflow as tf

# CREATE APP - Initialize FastAPI application
# title: Shows up in API documentation
app = FastAPI(title="Sign Language Recognition API")

# CORS CONFIGURATION - Allow frontend to access backend
# Think: Security guard allowing specific visitors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (websites)
    allow_credentials=True,    # Allow cookies/authentication
    allow_methods=["*"],       # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],       # Allow all headers
)

# INITIALIZE MEDIAPIPE - Create hand detector
mp_processor = MediaPipeProcessor()

# LOAD TRAINED MODEL - Load the AI brain we trained
# __file__ = this file's path
# dirname = get directory of this file
# join = combine paths correctly for your OS
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gesture_model.h5")
model = None  # Start with no model

try:
    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        # Load the trained neural network
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        # Model doesn't exist - need to train first
        print(f"Warning: Model not found at {MODEL_PATH}")
        print("Please train the model first using the training scripts.")
except Exception as e:
    # Catch any loading errors
    print(f"Error loading model: {e}")

# GESTURE LABELS - What each output number means
# Index 0 = 'A', Index 1 = 'B', etc.
# Must match the order used during training!
GESTURE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                  'U', 'V', 'W', 'X', 'Y', 'Z']

# REQUEST MODEL - Define what data frontend sends
# BaseModel = Pydantic class for data validation
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image string

# RESPONSE MODEL - Define what backend sends back
class PredictionResponse(BaseModel):
    prediction: str  # e.g., "A"
    confidence: float  # e.g., 0.95

# ROOT ENDPOINT - Test if server is running
# @app.get = Handle GET requests to "/"
@app.get("/")
async def root():
    # Return JSON response
    return {
        "message": "Sign Language Recognition API",
        "status": "running",
        "model_loaded": model is not None  # True if model exists
    }

# PREDICT ENDPOINT - Main endpoint for predictions
# @app.post = Handle POST requests to "/predict"
# response_model = Tells FastAPI what format to return
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    try:
        # STEP 1: DECODE IMAGE
        # Base64 images look like: "data:image/jpeg;base64,/9j/4AAQ..."
        # We only want the part after the comma
        
        # Split by comma, take second part if comma exists
        image_data = request.image.split(",")[1] if "," in request.image else request.image
        
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode array to OpenCV image (BGR format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if decoding worked
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # STEP 2: EXTRACT KEYPOINTS
        # Use MediaPipe to find hand landmarks
        keypoints = mp_processor.extract_keypoints(image)
        
        # If no hand detected, return early
        if keypoints is None:
            return PredictionResponse(
                prediction="No hand detected",
                confidence=0.0
            )
        
        # STEP 3: CHECK MODEL
        if model is None:
            return PredictionResponse(
                prediction="Model not loaded",
                confidence=0.0
            )
        
        # STEP 4: PREPARE DATA FOR MODEL
        # Convert list to numpy array
        # reshape(1, -1) = Make it 2D with 1 row, automatic columns
        # Model expects input shape: (batch_size, 63)
        keypoints_array = np.array(keypoints).reshape(1, -1)
        
        # STEP 5: MAKE PREDICTION
        # model.predict returns probabilities for each class
        # Shape: (1, 26) for 26 letters
        # verbose=0 = Don't print progress
        predictions = model.predict(keypoints_array, verbose=0)
        
        # Get index of highest probability
        # argmax = "argument of maximum" (index with biggest value)
        predicted_index = np.argmax(predictions[0])
        
        # Get confidence (probability of predicted class)
        # float() = Convert numpy float to Python float
        confidence = float(predictions[0][predicted_index])
        
        # STEP 6: GET LABEL
        # Convert index to letter
        predicted_gesture = GESTURE_LABELS[predicted_index] if predicted_index < len(GESTURE_LABELS) else "Unknown"
        
        # STEP 7: RETURN RESULT
        return PredictionResponse(
            prediction=predicted_gesture,
            confidence=confidence
        )
        
    except Exception as e:
        # Catch any errors and return them
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# HEALTH CHECK ENDPOINT - Check if everything is working
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mediapipe_ready": mp_processor is not None
    }

# RUN SERVER - Only runs if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    # Start server on all network interfaces, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)