# OpenCV - Computer vision library (read/process images)
import cv2

# MediaPipe - Google's hand detection library
import mediapipe as mp

# NumPy - Arrays and math operations
import numpy as np

class MediaPipeProcessor:
    """
    This class handles hand detection and keypoint extraction.
    
    Think of it as a specialized camera that can "see" hands
    and tell you where each finger joint is located.
    """
    
    def __init__(self):
        """
        Constructor - Runs when we create a new MediaPipeProcessor
        Sets up the hand detector with our preferred settings
        """
        
        # Get MediaPipe's hand detection module
        self.mp_hands = mp.solutions.hands
        
        # Create hand detector with specific settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # False = Video mode (faster)
            max_num_hands=1,          # Only detect 1 hand
            min_detection_confidence=0.5,  # 50% confidence to detect hand
            min_tracking_confidence=0.5    # 50% confidence to track hand
        )
        
        # Drawing utilities (for visualization, if needed)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_keypoints(self, image):
        """
        Find hand in image and extract 21 landmark positions.
        
        Think: Finding dots on each finger joint and palm
        
        Args:
            image: OpenCV image (BGR format, like a photo)
            
        Returns:
            List of 63 numbers: [x1, y1, z1, x2, y2, z2, ... x21, y21, z21]
            - Each hand has 21 landmarks (joints/points)
            - Each landmark has x, y, z coordinates
            - Total: 21 landmarks × 3 coordinates = 63 values
            
            Returns None if no hand detected
        """
        
        # STEP 1: CONVERT COLOR FORMAT
        # OpenCV uses BGR (Blue-Green-Red)
        # MediaPipe uses RGB (Red-Green-Blue)
        # We need to convert!
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # STEP 2: DETECT HAND
        # Process image through MediaPipe
        # This is where the AI magic happens!
        results = self.hands.process(image_rgb)
        
        # STEP 3: CHECK IF HAND FOUND
        # multi_hand_landmarks = list of detected hands
        # If list is empty or None, no hand found
        if not results.multi_hand_landmarks:
            return None  # No hand detected
        
        # STEP 4: GET FIRST HAND'S LANDMARKS
        # [0] = first hand (we only detect 1 hand anyway)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # STEP 5: EXTRACT COORDINATES
        # Create empty list to store all coordinates
        keypoints = []
        
        # Loop through all 21 landmarks
        # landmark.x = horizontal position (0.0 to 1.0)
        # landmark.y = vertical position (0.0 to 1.0)
        # landmark.z = depth position (relative to wrist)
        for landmark in hand_landmarks.landmark:
            # Add all 3 coordinates to list
            # extend = add multiple items at once
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        # STEP 6: RETURN KEYPOINTS
        # keypoints now has 63 values: [x1, y1, z1, x2, y2, z2, ...]
        return keypoints
    
    def draw_landmarks(self, image, keypoints):
        """
        Draw dots and lines on hand for visualization.
        Useful for debugging - see what AI "sees"
        
        Args:
            image: OpenCV image to draw on
            keypoints: List of 63 values from extract_keypoints
            
        Returns:
            Image with drawn landmarks
        """
        
        # If no keypoints, return original image
        if keypoints is None:
            return image
        
        # Make a copy so we don't modify original
        image_copy = image.copy()
        
        # Get image dimensions
        h, w, _ = image.shape  # height, width, channels
        
        # STEP 1: RESHAPE KEYPOINTS
        # Convert from [x1,y1,z1,x2,y2,z2,...] 
        # To: [[x1,y1,z1], [x2,y2,z2], ...]
        # This makes it easier to work with
        landmarks = np.array(keypoints).reshape(21, 3)
        
        # STEP 2: DRAW DOTS (circles) FOR EACH LANDMARK
        for i, (x, y, z) in enumerate(landmarks):
            # Convert normalized coordinates (0-1) to pixel coordinates
            cx = int(x * w)  # x position in pixels
            cy = int(y * h)  # y position in pixels
            
            # Draw green circle at this point
            # (cx, cy) = center position
            # 5 = radius
            # (0, 255, 0) = green color in BGR
            # -1 = fill the circle
            cv2.circle(image_copy, (cx, cy), 5, (0, 255, 0), -1)
            
            # Draw landmark number next to dot
            cv2.putText(
                image_copy,           # Image to draw on
                str(i),               # Text (landmark number)
                (cx + 10, cy),        # Position (slightly right of dot)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font
                0.3,                  # Font scale
                (255, 255, 255),      # White color
                1                     # Thickness
            )
        
        # STEP 3: DRAW LINES CONNECTING LANDMARKS
        # MediaPipe defines which landmarks should be connected
        # (which joints to draw lines between)
        connections = self.mp_hands.HAND_CONNECTIONS
        
        # Loop through each connection (pair of landmarks)
        for connection in connections:
            start_idx, end_idx = connection  # e.g., (0, 1)
            
            # Get pixel positions of both landmarks
            start_point = (
                int(landmarks[start_idx][0] * w),  # x of start
                int(landmarks[start_idx][1] * h)   # y of start
            )
            end_point = (
                int(landmarks[end_idx][0] * w),    # x of end
                int(landmarks[end_idx][1] * h)     # y of end
            )
            
            # Draw line between the two points
            cv2.line(
                image_copy,      # Image to draw on
                start_point,     # Start position
                end_point,       # End position
                (0, 255, 0),     # Green color
                2                # Line thickness
            )
        
        return image_copy
    
    def __del__(self):
        """
        Destructor - Runs when object is deleted
        Clean up MediaPipe resources to avoid memory leaks
        """
        if hasattr(self, 'hands'):
            self.hands.close()