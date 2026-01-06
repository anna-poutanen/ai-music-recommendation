"""
Emotion Recognition Model Architecture
7-class emotion classifier with focal loss support
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np


EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for multi-class classification
    Must match the training script version
    """
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.- epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create CNN model for emotion recognition
    
    Args:
        input_shape: (height, width, channels)
        num_classes: Number of emotion classes
        
    Returns:
        Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


class EmotionClassifier:
    """Wrapper class for emotion prediction"""
    
    def __init__(self, model_path='models/emotion_model.h5'):
        """
        Load trained emotion model
        
        Args: 
            model_path: Path to saved model
        """
        try:
            self.model = keras.models.load_model(
                model_path,
                custom_objects={'focal_loss': focal_loss()},
                compile=False
            )
            
            self.model.compile(
                optimizer='adam',
                loss=focal_loss(),
                metrics=['accuracy']
            )
            print(f"✅ Model loaded successfully from {model_path}")
            
        except Exception as e: 
            print(f"❌ Error loading model: {e}")
            print(f"   Trying alternative loading method...")
            
            try:
                self.model = create_emotion_model()
                self.model.load_weights(model_path)
                print(f"✅ Model weights loaded successfully")
                
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                print(f"\n💡 Solution:  Retrain the model using:")
                print(f"   python3 src/train_emotion_model.py")
                raise
        
        self.emotions = EMOTIONS
        
    def predict_emotion(self, face_image):
        """
        Predict emotion from face image
        
        Args: 
            face_image: 48x48 grayscale numpy array
            
        Returns: 
            dict:  {emotion: probability} for all emotions
        """
        face_image = face_image / 255.0
        face_image = face_image.reshape(1, 48, 48, 1)
        predictions = self.model.predict(face_image, verbose=0)[0]
        
        emotion_probs = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotions, predictions)
        }
        
        return emotion_probs
    
    def get_top_emotion(self, face_image, confidence_threshold=0.4):
        """
        Get dominant emotion with confidence threshold
        
        Args:
            face_image:  48x48 grayscale numpy array
            confidence_threshold:  Minimum confidence to return emotion
            
        Returns: 
            tuple: (emotion, confidence) or ('Neutral', conf) if below threshold
        """
        emotion_probs = self.predict_emotion(face_image)
        
        top_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[top_emotion]
        
        if confidence < confidence_threshold:
            return 'Neutral', confidence
            
        return top_emotion, confidence


def model_summary():
    """Print model architecture"""
    model = create_emotion_model()
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    model_summary()

