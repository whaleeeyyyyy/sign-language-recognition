import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class SignLanguageTrainer:
    def __init__(self, dataset_path="dataset", model_path="../backend/app/models"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.label_encoder = LabelEncoder()
        
        # Create model directory
        os.makedirs(self.model_path, exist_ok=True)
    
    def extract_keypoints(self, image_path):
        """Extract hand keypoints from an image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Check if hand landmarks were detected
        if not results.multi_hand_landmarks:
            return None
        
        # Extract keypoints
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for landmark in hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(keypoints)
    
    def load_dataset(self):
        """Load and process the dataset"""
        print("Loading dataset...")
        X = []
        y = []
        
        # Get all gesture folders
        gesture_folders = [f for f in os.listdir(self.dataset_path) 
                          if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        if not gesture_folders:
            raise ValueError(f"No gesture folders found in {self.dataset_path}")
        
        print(f"Found {len(gesture_folders)} gesture classes: {sorted(gesture_folders)}")
        
        # Process each gesture folder
        for gesture in gesture_folders:
            gesture_path = os.path.join(self.dataset_path, gesture)
            image_files = [f for f in os.listdir(gesture_path) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {gesture}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(gesture_path, img_file)
                keypoints = self.extract_keypoints(img_path)
                
                if keypoints is not None:
                    X.append(keypoints)
                    y.append(gesture)
        
        print(f"Successfully loaded {len(X)} samples")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded
    
    def build_model(self, input_shape, num_classes):
        """Build a neural network model"""
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            
            # First dense block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second dense block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third dense block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_history.png'))
        print(f"Training history plot saved to {self.model_path}/training_history.png")
    
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        # Load dataset
        X, y = self.load_dataset()
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=26, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Build model
        model = self.build_model(X.shape[1], len(np.unique(y)))
        print("\nModel architecture:")
        model.summary()
        
        # Define callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train model
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Save model
        model_file = os.path.join(self.model_path, "gesture_model.h5")
        model.save(model_file)
        print(f"\nModel saved to: {model_file}")
        
        # Save label encoder classes
        labels_file = os.path.join(self.model_path, "labels.npy")
        np.save(labels_file, self.label_encoder.classes_)
        print(f"Labels saved to: {labels_file}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history

if __name__ == "__main__":
    trainer = SignLanguageTrainer()
    model, history = trainer.train(epochs=50, batch_size=32)
    print("\nTraining complete!")