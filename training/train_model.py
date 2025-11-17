# === IMPORTS - Libraries we need ===

# File and system operations
import os

# Computer vision
import cv2

# Arrays and math
import numpy as np

# Hand detection
import mediapipe as mp

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder

# Deep learning library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Plotting graphs
import matplotlib.pyplot as plt


class SignLanguageTrainer:
    """
    This class handles training the AI to recognize hand gestures.
    
    Think: A teacher that learns from examples and creates
    a smart pattern matcher (neural network)
    """
    
    def __init__(self, dataset_path="dataset", model_path="../backend/app/models"):
        """
        Initialize the trainer
        
        Args:
            dataset_path: Where training images are stored
            model_path: Where to save the trained model
        """
        # Where images are
        self.dataset_path = dataset_path
        
        # Where to save trained model
        self.model_path = model_path
        
        # SETUP MEDIAPIPE FOR HAND DETECTION
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # True = Image mode (not video)
            max_num_hands=1,         # Only detect 1 hand
            min_detection_confidence=0.5  # 50% confidence threshold
        )
        
        # LABEL ENCODER
        # Converts labels like "A", "B", "C" to numbers 0, 1, 2
        # Neural networks work with numbers, not letters!
        self.label_encoder = LabelEncoder()
        
        # Create folder for saving model
        os.makedirs(self.model_path, exist_ok=True)
    
    def extract_keypoints(self, image_path):
        """
        Load image and extract hand keypoints
        
        Args:
            image_path: Path to image file
            
        Returns:
            Numpy array of 63 values (21 landmarks × 3 coordinates)
            or None if no hand detected
        """
        # LOAD IMAGE
        image = cv2.imread(image_path)
        
        # Check if image loaded successfully
        if image is None:
            return None
        
        # CONVERT COLOR
        # OpenCV uses BGR, MediaPipe uses RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # DETECT HAND
        results = self.hands.process(image_rgb)
        
        # CHECK IF HAND FOUND
        if not results.multi_hand_landmarks:
            return None  # No hand in image
        
        # EXTRACT KEYPOINTS
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        
        # Get x, y, z for each of 21 landmarks
        for landmark in hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        # Return as numpy array
        return np.array(keypoints)
    
    def load_dataset(self):
        """
        Load all images and extract keypoints for training
        
        This is like preparing flashcards for studying:
        - Read all images
        - Extract hand positions
        - Remember which gesture each is
        
        Returns:
            X: Array of keypoints (features) - shape: (num_samples, 63)
            y: Array of labels (targets) - shape: (num_samples,)
        """
        print("Loading dataset...")
        
        # Lists to store data
        X = []  # Features (keypoints)
        y = []  # Labels (which gesture)
        
        # GET ALL GESTURE FOLDERS
        # os.listdir = list all files/folders in directory
        # We only want folders (not files)
        gesture_folders = [
            f for f in os.listdir(self.dataset_path) 
            if os.path.isdir(os.path.join(self.dataset_path, f))
        ]
        
        # Check if any folders found
        if not gesture_folders:
            raise ValueError(f"No gesture folders found in {self.dataset_path}")
        
        print(f"Found {len(gesture_folders)} gesture classes: {sorted(gesture_folders)}")
        
        # PROCESS EACH GESTURE FOLDER
        for gesture in gesture_folders:
            # Full path to gesture folder
            gesture_path = os.path.join(self.dataset_path, gesture)
            
            # Get all image files in folder
            image_files = [
                f for f in os.listdir(gesture_path) 
                if f.endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            print(f"Processing {gesture}: {len(image_files)} images")
            
            # PROCESS EACH IMAGE
            for img_file in image_files:
                # Full path to image
                img_path = os.path.join(gesture_path, img_file)
                
                # Extract keypoints from image
                keypoints = self.extract_keypoints(img_path)
                
                # Only use images where hand was detected
                if keypoints is not None:
                    X.append(keypoints)  # Add features
                    y.append(gesture)     # Add label
        
        print(f"Successfully loaded {len(X)} samples")
        
        # CONVERT TO NUMPY ARRAYS
        # Lists → Arrays (faster for math operations)
        X = np.array(X)
        y = np.array(y)
        
        # ENCODE LABELS TO NUMBERS
        # "A", "B", "C" → 0, 1, 2
        # fit_transform = learn the mapping and convert
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded
    
    def build_model(self, input_shape, num_classes):
        """
        Build the neural network architecture
        
        Think: Designing the brain structure
        - Input layer: Receives 63 keypoint values
        - Hidden layers: Learn patterns
        - Output layer: Predicts which gesture (26 letters)
        
        Args:
            input_shape: Number of input features (63)
            num_classes: Number of output classes (26 letters)
            
        Returns:
            Compiled Keras model ready for training
        """
        # CREATE MODEL - Sequential = layers stacked one after another
        model = keras.Sequential([
            
            # INPUT LAYER
            # Receives 63 values (hand keypoints)
            layers.Input(shape=(input_shape,)),
            
            # === FIRST DENSE BLOCK ===
            # Dense = Fully connected layer (every input connects to every output)
            layers.Dense(256, activation='relu'),
            # 256 neurons, relu = rectified linear unit (makes learning easier)
            
            # BatchNormalization = Normalizes data (speeds up training)
            layers.BatchNormalization(),
            
            # Dropout = Randomly turns off 30% of neurons
            # Why? Prevents overfitting (memorizing instead of learning)
            layers.Dropout(0.3),
            
            # === SECOND DENSE BLOCK ===
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # === THIRD DENSE BLOCK ===
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),  # Less dropout in deeper layers
            
            # === OUTPUT LAYER ===
            # num_classes neurons (26 for A-Z)
            # softmax = converts outputs to probabilities (all add up to 1.0)
            # Example output: [0.1, 0.7, 0.05, ...] means 70% sure it's "B"
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # COMPILE MODEL
        # Think: Setting up the learning rules
        model.compile(
            optimizer='adam',  # Adam = smart learning algorithm
            loss='sparse_categorical_crossentropy',  # Loss function for multi-class
            metrics=['accuracy']  # Track accuracy during training
        )
        
        return model
    
    def plot_training_history(self, history):
        """
        Create graphs showing training progress
        
        Args:
            history: Training history from model.fit()
        """
        # CREATE FIGURE WITH 2 SUBPLOTS
        # 1 row, 2 columns, figure size 12x4 inches
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # === LEFT PLOT: ACCURACY ===
        # Plot training accuracy (how well model does on training data)
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        
        # Plot validation accuracy (how well model does on unseen data)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')  # Epoch = one complete pass through data
        ax1.set_ylabel('Accuracy')
        ax1.legend()  # Show legend
        ax1.grid(True)  # Add grid lines
        
        # === RIGHT PLOT: LOSS ===
        # Loss = how wrong the model is (lower is better)
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # SAVE FIGURE
        plt.tight_layout()  # Adjust spacing
        plt.savefig(os.path.join(self.model_path, 'training_history.png'))
        print(f"Training history plot saved to {self.model_path}/training_history.png")
    
    def train(self, epochs=50, batch_size=32):
        """
        Main training function - teaches the AI
        
        Process:
        1. Load dataset (images → keypoints)
        2. Split into training and testing
        3. Build neural network
        4. Train the model
        5. Save trained model
        
        Args:
            epochs: How many times to go through all training data
            batch_size: How many samples to process at once
            
        Returns:
            model: Trained Keras model
            history: Training history (for plotting)
        """
        
        # === STEP 1: LOAD DATASET ===
        X, y = self.load_dataset()
        
        print(f"\nDataset shape: {X.shape}")
        # Shape example: (1000, 63) = 1000 samples, 63 features each
        
        print(f"Number of classes: {len(np.unique(y))}")
        # unique = get distinct values (count how many different gestures)
        
        # === STEP 2: SPLIT DATASET ===
        # Training set: Model learns from this (80%)
        # Testing set: Model evaluated on this (20%)
        # stratify=y: Keep same proportion of each class in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,      # 20% for testing
            random_state=42,    # Random seed (makes results reproducible)
            stratify=y          # Balance classes
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # === STEP 3: BUILD MODEL ===
        model = self.build_model(
            X.shape[1],           # Input shape = 63 features
            len(np.unique(y))     # Number of classes (26 letters)
        )
        
        print("\nModel architecture:")
        model.summary()  # Print model structure
        
        # === STEP 4: DEFINE CALLBACKS ===
        # Callbacks = functions that run during training
        
        # Early Stopping: Stop training if not improving
        # Why? Saves time and prevents overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',        # Watch validation loss
            patience=10,               # Stop after 10 epochs without improvement
            restore_best_weights=True  # Go back to best model
        )
        
        # Reduce Learning Rate: Make learning slower when stuck
        # Think: Taking smaller steps when close to goal
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Watch validation loss
            factor=0.5,          # Reduce by 50%
            patience=5,          # Wait 5 epochs before reducing
            min_lr=0.00001       # Don't go below this
        )
        
        # === STEP 5: TRAIN MODEL ===
        print("\nTraining model...")
        
        # fit = train the model
        # This is where the learning happens!
        history = model.fit(
            X_train, y_train,              # Training data
            validation_data=(X_test, y_test),  # Testing data
            epochs=epochs,                 # How many times through data
            batch_size=batch_size,         # Samples per update
            callbacks=[early_stopping, reduce_lr],  # Use our callbacks
            verbose=1                      # Print progress (1 = progress bar)
        )
        
        # === STEP 6: EVALUATE MODEL ===
        # Test the final model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        # Example: 0.9500 = 95% correct
        
        print(f"Test loss: {test_loss:.4f}")
        # Lower loss = better predictions
        
        # === STEP 7: SAVE MODEL ===
        # Save to .h5 file (HDF5 format)
        model_file = os.path.join(self.model_path, "gesture_model.h5")
        model.save(model_file)
        print(f"\nModel saved to: {model_file}")
        
        # === STEP 8: SAVE LABEL MAPPINGS ===
        # Save which number corresponds to which letter
        # Example: 0→A, 1→B, 2→C, ...
        labels_file = os.path.join(self.model_path, "labels.npy")
        np.save(labels_file, self.label_encoder.classes_)
        print(f"Labels saved to: {labels_file}")
        
        # === STEP 9: PLOT TRAINING HISTORY ===
        self.plot_training_history(history)
        
        return model, history


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Create trainer instance
    trainer = SignLanguageTrainer()
    
    # Train the model
    # This takes 5-15 minutes depending on:
    # - How many images you collected
    # - Your computer's speed
    # - Whether you have a GPU
    model, history = trainer.train(epochs=50, batch_size=32)
    
    print("\nTraining complete!")
    print("\nNext steps:")
    print("1. Check training_history.png to see if model learned well")
    print("2. Start the backend server")
    print("3. Start the frontend")
    print("4. Test your sign language recognition!")