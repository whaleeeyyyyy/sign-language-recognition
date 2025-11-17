# Sign Language Recognition Web App

A real-time sign language recognition system that uses computer vision and deep learning to translate hand gestures into text.

## Features

- 🎥 Real-time webcam capture and processing
- 🤖 MediaPipe-based hand landmark detection
- 🧠 Deep learning gesture classification
- 💬 Interactive sentence builder
- 🎨 Modern, responsive UI with React and Tailwind CSS

## Architecture

Camera Feed → FastAPI Backend → MediaPipe Hands → Keypoint Extraction →
Neural Network Classifier → Predicted Gesture → Frontend Display

## Tech Stack

### Frontend

- React 18
- Vite
- Tailwind CSS
- react-webcam
- Axios

### Backend

- FastAPI
- MediaPipe
- TensorFlow/Keras
- OpenCV
- NumPy

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- Webcam

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd sign-language-recognition
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### 4. Training Setup (Optional - for model training)

```bash
# Navigate to training directory
cd training

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Training Your Model

### Step 1: Collect Training Data

```bash
cd training
python collect_data.py
```

**Instructions:**

1. Press A-Z to select which letter/gesture to collect
2. Position your hand clearly in front of the camera
3. Press SPACE to capture images
4. Collect at least 100 images per gesture for best results
5. Press 'q' to quit

**Tips for better data:**

- Use various hand positions and angles
- Ensure good lighting
- Keep background simple
- Vary distance from camera slightly

### Step 2: Train the Model

```bash
python train_model.py
```

This will:

- Load all collected images
- Extract hand keypoints using MediaPipe
- Train a neural network classifier
- Save the model to `../backend/app/models/gesture_model.h5`
- Generate training visualization graphs

**Training typically takes 5-15 minutes depending on dataset size.**

## Running the Application

### Step 1: Start the Backend

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Step 2: Start the Frontend

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:3000`

## Usage

1. **Start Recognition**: Click "Start Recognition" button
2. **Perform Gestures**: Show hand signs to the camera
3. **View Predictions**: See real-time predictions with confidence scores
4. **Build Sentences**: Click "Add to Sentence" to build words
5. **Manage Text**: Use "Add Space" and "Clear" buttons

## Project Structure

sign-language-recognition/
├── frontend/ # React frontend application
│ ├── src/
│ │ ├── components/ # React components
│ │ ├── utils/ # API utilities
│ │ └── App.jsx # Main app component
│ └── package.json
│
├── backend/ # FastAPI backend
│ ├── app/
│ │ ├── main.py # API endpoints
│ │ ├── models/ # Trained models
│ │ └── utils/ # MediaPipe processor
│ └── requirements.txt
│
└── training/ # Model training scripts
├── collect_data.py # Data collection tool
├── train_model.py # Model training script
└── dataset/ # Collected training data

## API Endpoints

### `POST /predict`

Predicts gesture from webcam image.

**Request:**

```json
{
  "image": "base64_encoded_image"
}
```

**Response:**

```json
{
  "prediction": "A",
  "confidence": 0.95
}
```

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "mediapipe_ready": true
}
```

## Troubleshooting

### Model Not Loading

- Ensure you've trained the model first using `train_model.py`
- Check that `gesture_model.h5` exists in `backend/app/models/`

### Poor Prediction Accuracy

- Collect more training data (aim for 100+ images per gesture)
- Ensure consistent lighting conditions
- Use clear, distinct hand gestures
- Retrain the model with more diverse data

### Webcam Not Detected

- Check browser permissions
- Ensure no other application is using the webcam
- Try a different browser (Chrome recommended)

### CORS Errors

- Ensure backend is running on port 8000
- Check frontend API_BASE_URL in `src/utils/api.js`

## Performance Tips

- Use a well-lit environment
- Position hand clearly in frame
- Keep background simple and contrasting
- Maintain consistent distance from camera
- Allow the model to stabilize (wait 1-2 seconds per gesture)

## Future Enhancements

- [ ] Support for full words and phrases
- [ ] Multiple hand gesture recognition
- [ ] Real-time video feedback with landmark visualization
- [ ] Export sentence history
- [ ] Voice output for predicted text
- [ ] Mobile app version
- [ ] Expanded gesture vocabulary

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MediaPipe by Google for hand landmark detection
- TensorFlow team for the ML framework
- React and FastAPI communities
