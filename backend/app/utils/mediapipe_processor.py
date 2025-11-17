import cv2
import mediapipe as mp
import numpy as np

class MediaPipeProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_keypoints(self, image):
        """
        Extract hand keypoints from an image using MediaPipe.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of 63 normalized values (21 landmarks * 3 coordinates)
            or None if no hand detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Check if hand landmarks were detected
        if not results.multi_hand_landmarks:
            return None
        
        # Get the first hand's landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract and normalize keypoints
        keypoints = []
        for landmark in hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        return keypoints
    
    def draw_landmarks(self, image, keypoints):
        """
        Draw hand landmarks on the image for visualization.
        
        Args:
            image: OpenCV image
            keypoints: List of 63 values from extract_keypoints
            
        Returns:
            Image with drawn landmarks
        """
        if keypoints is None:
            return image
        
        image_copy = image.copy()
        h, w, _ = image.shape
        
        # Reshape keypoints to (21, 3)
        landmarks = np.array(keypoints).reshape(21, 3)
        
        # Draw landmarks
        for i, (x, y, z) in enumerate(landmarks):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(image_copy, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(image_copy, str(i), (cx + 10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw connections
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_point = (int(landmarks[start_idx][0] * w), 
                          int(landmarks[start_idx][1] * h))
            end_point = (int(landmarks[end_idx][0] * w), 
                        int(landmarks[end_idx][1] * h))
            cv2.line(image_copy, start_point, end_point, (0, 255, 0), 2)
        
        return image_copy
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()