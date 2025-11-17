import cv2
import os
import time
from datetime import datetime

class DataCollector:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.current_gesture = None
        self.image_count = 0
        self.is_collecting = False
        
        # Create dataset directory if it doesn't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        
    def create_gesture_folder(self, gesture_name):
        """Create a folder for the gesture if it doesn't exist"""
        gesture_path = os.path.join(self.dataset_path, gesture_name)
        os.makedirs(gesture_path, exist_ok=True)
        return gesture_path
    
    def save_image(self, frame, gesture_name):
        """Save an image to the gesture folder"""
        gesture_path = self.create_gesture_folder(gesture_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{gesture_name}_{timestamp}.jpg"
        filepath = os.path.join(gesture_path, filename)
        cv2.imwrite(filepath, frame)
        self.image_count += 1
        print(f"Saved: {filename} (Total: {self.image_count})")
    
    def display_instructions(self, frame):
        """Display instructions on the frame"""
        instructions = [
            "Press A-Z to start collecting that gesture",
            "Press SPACE to save image",
            "Press 'q' to quit",
            f"Current Gesture: {self.current_gesture or 'None'}",
            f"Images Saved: {self.image_count}"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n=== Sign Language Data Collection ===")
        print("Instructions:")
        print("1. Press A-Z to select a gesture to collect")
        print("2. Position your hand in front of the camera")
        print("3. Press SPACE to capture images")
        print("4. Collect at least 100 images per gesture")
        print("5. Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Add instructions overlay
            display_frame = self.display_instructions(frame.copy())
            
            # Show the frame
            cv2.imshow('Data Collection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q'):
                break
            
            # Save image
            elif key == ord(' ') and self.current_gesture:
                key = cv2.waitKey(1) & 0xFF
                print("Key pressed:", key)
                self.save_image(frame, self.current_gesture)
            
            # Select gesture (A-Z)
            elif key >= ord('a') and key <= ord('z'):
                self.current_gesture = chr(key).upper()
                self.image_count = 0
                print(f"\nNow collecting: {self.current_gesture}")
                print("Press SPACE to capture images")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nData collection complete!")
        print(f"Dataset saved to: {self.dataset_path}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()