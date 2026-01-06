"""
Face Detection Module
Extracts face regions from webcam or image input
"""

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self):
        """Initialize MediaPipe Face Detection"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
    def detect_face(self, image):
        """
        Detect face in image and return cropped face region
        
        Args:
            image: BGR image from OpenCV
            
        Returns: 
            face_crop: 48x48 grayscale face image, or None if no face detected
            bbox:  Bounding box coordinates (x, y, w, h), or None
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return None, None
            
        # Get first face detection
        detection = results.detections[0]
        
        # Get bounding box
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        
        # Convert to absolute coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Add padding (10%)
        padding = int(0.1 * min(width, height))
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(w - x, width + 2 * padding)
        height = min(h - y, height + 2 * padding)
        
        # Crop face
        face_crop = image[y:y+height, x: x+width]
        
        # Resize to 48x48
        face_crop = cv2.resize(face_crop, (48, 48))
        
        # Convert to grayscale
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        return face_crop, (x, y, width, height)
    
    def draw_face_box(self, image, bbox):
        """
        Draw bounding box on image
        
        Args:
            image: BGR image
            bbox: (x, y, w, h)
            
        Returns:
            image with bounding box drawn
        """
        if bbox is None:
            return image
            
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
    
    def close(self):
        """Release resources"""
        self.face_detection.close()


def test_face_detection():
    """Test face detection with webcam"""
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save face crop")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect face
        face_crop, bbox = detector.detect_face(frame)
        
        # Draw bounding box
        if bbox is not None:
            frame = detector.draw_face_box(frame, bbox)
            
        # Show result
        cv2.imshow('Face Detection', frame)
        
        if face_crop is not None:
            cv2.imshow('Face Crop (48x48)', face_crop)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and face_crop is not None: 
            cv2.imwrite('face_crop.jpg', face_crop)
            print("Saved face_crop.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    test_face_detection()