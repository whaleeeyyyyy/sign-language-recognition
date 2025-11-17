# OpenCV - Computer vision library for camera access
import cv2

# os - Operating system functions (create folders, etc.)
import os

# time - For delays (not used here, but useful for future)
import time

# datetime - For timestamps in filenames
from datetime import datetime

class DataCollector:
    """
    Tool for collecting training images from webcam.
    
    Think: A camera app that saves photos to organized folders
    """
    
    def __init__(self, dataset_path="dataset"):
        """
        Initialize the data collector
        
        Args:
            dataset_path: Where to save collected images
        """
        # Where to save images
        self.dataset_path = dataset_path
        
        # Which gesture we're currently collecting (e.g., "A")
        self.current_gesture = None
        
        # How many images saved so far
        self.image_count = 0
        
        # Are we currently in "collecting mode"?
        self.is_collecting = False
        
        # CREATE DATASET FOLDER
        # exist_ok=True = Don't error if folder already exists
        os.makedirs(self.dataset_path, exist_ok=True)
        
    def create_gesture_folder(self, gesture_name):
        """
        Create a folder for specific gesture if it doesn't exist
        
        Example: Creates "dataset/A/" folder
        
        Args:
            gesture_name: Letter/gesture name (e.g., "A")
            
        Returns:
            Full path to gesture folder
        """
        # Combine paths: dataset + gesture_name
        # e.g., "dataset" + "A" = "dataset/A"
        gesture_path = os.path.join(self.dataset_path, gesture_name)
        
        # Create folder (won't error if exists)
        os.makedirs(gesture_path, exist_ok=True)
        
        return gesture_path
    
    def save_image(self, frame, gesture_name):
        """
        Save current camera frame as image file
        
        Args:
            frame: OpenCV image from webcam
            gesture_name: Which gesture this is (e.g., "A")
        """
        # Get the folder for this gesture
        gesture_path = self.create_gesture_folder(gesture_name)
        
        # CREATE UNIQUE FILENAME
        # Format: A_20231215_143052_123456.jpg
        # Includes: gesture, date, time, microseconds (ensures uniqueness)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{gesture_name}_{timestamp}.jpg"
        
        # Full path: folder + filename
        filepath = os.path.join(gesture_path, filename)
        
        # SAVE IMAGE
        # cv2.imwrite = save OpenCV image to file
        cv2.imwrite(filepath, frame)
        
        # Increment counter
        self.image_count += 1
        
        # Print confirmation
        print(f"Saved: {filename} (Total: {self.image_count})")
    
    def display_instructions(self, frame):
        """
        Add text overlay with instructions
        
        Args:
            frame: OpenCV image to draw on
            
        Returns:
            Frame with text overlay
        """
        # List of instructions to show
        instructions = [
            "Press A-Z to start collecting that gesture",
            "Press SPACE to save image",
            "Press 'q' to quit",
            f"Current Gesture: {self.current_gesture or 'None'}",
            f"Images Saved: {self.image_count}"
        ]
        
        # Starting Y position for text
        y_offset = 30
        
        # Loop through each instruction
        for i, text in enumerate(instructions):
            # Draw text on image
            cv2.putText(
                frame,                    # Image to draw on
                text,                     # Text string
                (10, y_offset + i * 30), # Position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX, # Font
                0.6,                      # Font scale
                (0, 255, 0),             # Green color (BGR)
                2                         # Thickness
            )
        
        return frame
    
    def run(self):
        """
        Main loop - runs the data collection program
        """
        # OPEN WEBCAM
        # 0 = first camera (usually built-in webcam)
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # PRINT STARTUP INSTRUCTIONS
        print("\n=== Sign Language Data Collection ===")
        print("Instructions:")
        print("1. Press A-Z to select a gesture to collect")
        print("2. Position your hand in front of the camera")
        print("3. Press SPACE to capture images")
        print("4. Collect at least 100 images per gesture")
        print("5. Press 'q' to quit\n")
        
        # MAIN LOOP - Runs continuously until user quits
        while True:
            # READ FRAME FROM CAMERA
            # ret = success (True/False)
            # frame = the captured image
            ret, frame = cap.read()
            
            # Check if frame captured successfully
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # FLIP FRAME HORIZONTALLY
            # Makes it like a mirror (easier for user)
            # 1 = flip horizontal, 0 = flip vertical
            frame = cv2.flip(frame, 1)
            
            # ADD INSTRUCTIONS OVERLAY
            # .copy() = create copy so we don't modify original
            display_frame = self.display_instructions(frame.copy())
            
            # SHOW FRAME IN WINDOW
            cv2.imshow('Data Collection', display_frame)
            
            # HANDLE KEY PRESSES
            # waitKey(1) = wait 1 millisecond for key press
            # & 0xFF = get only last 8 bits (handles cross-platform issues)
            key = cv2.waitKey(1) & 0xFF
            
            # QUIT - User pressed 'q'
            if key == ord('q'):
                break
            
            # SAVE IMAGE - User pressed SPACE
            # ord(' ') = ASCII code for space character
            elif key == ord(' ') and self.current_gesture:
                # Save current frame for current gesture
                self.save_image(frame, self.current_gesture)
            
            # SELECT GESTURE - User pressed A-Z
            # Check if key is between 'a' and 'z'
            elif key >= ord('a') and key <= ord('z'):
                # Convert to uppercase letter
                # chr(key) = convert ASCII code to character
                self.current_gesture = chr(key).upper()
                
                # Reset counter for new gesture
                self.image_count = 0
                
                # Print confirmation
                print(f"\nNow collecting: {self.current_gesture}")
                print("Press SPACE to capture images")
        
        # CLEANUP - Release resources
        cap.release()           # Release camera
        cv2.destroyAllWindows() # Close all windows
        
        # Print summary
        print(f"\nData collection complete!")
        print(f"Dataset saved to: {self.dataset_path}")

# RUN IF THIS FILE IS EXECUTED DIRECTLY
if __name__ == "__main__":
    # Create collector instance
    collector = DataCollector()
    
    # Start collection process
    collector.run()