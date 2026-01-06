"""
Test trained emotion model on images
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from emotion_model import EmotionClassifier, EMOTIONS
from face_detection import FaceDetector


def test_on_image(image_path, model_path='models/emotion_model. h5'):
    """
    Test emotion detection on a single image
    
    Args:
        image_path:  Path to image file
        model_path: Path to trained model
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error:  Could not load image from {image_path}")
        return
    
    # Detect face
    detector = FaceDetector()
    face_crop, bbox = detector.detect_face(image)
    
    if face_crop is None:
        print("No face detected in image")
        detector.close()
        return
    
    # Draw bounding box
    image = detector.draw_face_box(image, bbox)
    
    # Predict emotion
    classifier = EmotionClassifier(model_path)
    emotion_probs = classifier.predict_emotion(face_crop)
    top_emotion, confidence = classifier.get_top_emotion(face_crop)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Detected Emotion: {top_emotion} (confidence: {confidence:.2f})")
    print("=" * 50)
    print("\nAll emotion probabilities:")
    for emotion in EMOTIONS:
        prob = emotion_probs[emotion]
        bar = "█" * int(prob * 50)
        print(f"{emotion: 12s}: {prob:.4f} {bar}")
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show original image with bounding box
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title('Face Detection')
    ax1.axis('off')
    
    # Show emotion probabilities
    emotions_list = list(emotion_probs.keys())
    probs_list = list(emotion_probs.values())
    colors = ['red' if e == top_emotion else 'skyblue' for e in emotions_list]
    
    ax2.barh(emotions_list, probs_list, color=colors)
    ax2.set_xlabel('Probability')
    ax2.set_title(f'Emotion:  {top_emotion} ({confidence:.2%})')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('outputs/test_result.png')
    print("\nResult saved to outputs/test_result. png")
    plt.show()
    
    detector.close()


def test_on_webcam(model_path='models/emotion_model.h5'):
    """
    Test emotion detection on webcam feed
    
    Args:
        model_path: Path to trained model
    """
    detector = FaceDetector()
    classifier = EmotionClassifier(model_path)
    cap = cv2.VideoCapture(0)
    
    print("\n" + "=" * 50)
    print("WEBCAM EMOTION DETECTION")
    print("=" * 50)
    print("Press 'q' to quit")
    print("=" * 50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face
        face_crop, bbox = detector. detect_face(frame)
        
        if face_crop is not None:
            # Predict emotion
            top_emotion, confidence = classifier.get_top_emotion(face_crop)
            
            # Draw bounding box
            frame = detector. draw_face_box(frame, bbox)
            
            # Draw emotion text
            x, y, w, h = bbox
            text = f"{top_emotion}:  {confidence:.2f}"
            cv2.putText(
                frame, text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2
            )
        
        # Display
        cv2.imshow('Emotion Detection (Press Q to quit)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test emotion recognition model')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--webcam', action='store_true', help='Test on webcam')
    parser.add_argument('--model', type=str, default='models/emotion_model.h5',
                        help='Path to model file')
    
    args = parser.parse_args()
    
    if args.webcam:
        test_on_webcam(args.model)
    elif args.image:
        test_on_image(args.image, args.model)
    else:
        print("Please specify --image <path> or --webcam")
        print("\nExamples:")
        print("  python test_emotion_model.py --image test. jpg")
        print("  python test_emotion_model.py --webcam")