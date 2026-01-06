"""
Emotion Interpretation Layer
Processes raw emotion predictions into usable insights
"""

import numpy as np
from collections import deque


class EmotionEngine:
    """
    Processes and smooths emotion predictions
    """
    
    def __init__(self, confidence_threshold=0.4, smoothing_window=5):
        """
        Args:
            confidence_threshold:  Minimum confidence for non-neutral prediction
            smoothing_window: Number of frames to smooth over
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.emotion_history = deque(maxlen=smoothing_window)
        
    def process_emotion(self, emotion_probs, use_smoothing=False):
        """
        Process raw emotion probabilities
        
        Args: 
            emotion_probs: Dict of {emotion: probability}
            use_smoothing: Whether to apply temporal smoothing
            
        Returns: 
            dict: Processed emotion data
        """
        # Get top emotion
        top_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[top_emotion]
        
        # Apply confidence threshold
        if confidence < self. confidence_threshold:
            top_emotion = 'Neutral'
            confidence = emotion_probs.get('Neutral', confidence)
        
        # Add to history for smoothing
        if use_smoothing:
            self.emotion_history.append(top_emotion)
            top_emotion = self._get_smoothed_emotion()
        
        # Get emotion intensity (weighted by top 2 emotions)
        sorted_emotions = sorted(
            emotion_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        intensity = sorted_emotions[0][1]
        if len(sorted_emotions) > 1:
            intensity += 0.3 * sorted_emotions[1][1]
        intensity = min(intensity, 1.0)
        
        return {
            'emotion': top_emotion,
            'confidence': confidence,
            'intensity': intensity,
            'all_probabilities': emotion_probs
        }
    
    def _get_smoothed_emotion(self):
        """
        Get most common emotion from history
        
        Returns: 
            Most common emotion in recent history
        """
        if not self.emotion_history:
            return 'Neutral'
        
        # Count occurrences
        counts = {}
        for emotion in self.emotion_history:
            counts[emotion] = counts.get(emotion, 0) + 1
        
        # Return most common
        return max(counts, key=counts.get)
    
    def reset(self):
        """Reset emotion history"""
        self.emotion_history.clear()
    
    def get_emotion_features(self, emotion_data):
        """
        Convert emotion to audio features for music recommendation
        
        Args:
            emotion_data: Dict from process_emotion()
            
        Returns:
            dict: Audio feature targets {feature:  value}
        """
        emotion = emotion_data['emotion']
        intensity = emotion_data['intensity']
        
        # Define emotion-to-audio-feature mapping
        emotion_map = {
            'Happy': {
                'valence': 0.8,
                'energy': 0.7,
                'danceability': 0.7,
                'tempo': 120
            },
            'Sad': {
                'valence':  0.2,
                'energy': 0.3,
                'danceability':  0.3,
                'tempo': 80
            },
            'Angry': {
                'valence':  0.2,
                'energy': 0.9,
                'danceability':  0.5,
                'tempo': 140
            },
            'Fearful': {
                'valence': 0.3,
                'energy': 0.6,
                'danceability': 0.4,
                'tempo': 110
            },
            'Surprised': {
                'valence':  0.6,
                'energy': 0.7,
                'danceability': 0.6,
                'tempo': 125
            },
            'Disgusted': {
                'valence': 0.25,
                'energy': 0.5,
                'danceability':  0.4,
                'tempo': 100
            },
            'Neutral': {
                'valence':  0.5,
                'energy': 0.5,
                'danceability': 0.5,
                'tempo': 110
            }
        }
        
        # Get base features for emotion
        features = emotion_map.get(emotion, emotion_map['Neutral']).copy()
        
        # Adjust by intensity
        for key in ['valence', 'energy', 'danceability']: 
            # Scale feature by intensity
            base_val = features[key]
            if base_val > 0.5:
                features[key] = 0.5 + (base_val - 0.5) * intensity
            else:
                features[key] = 0.5 - (0.5 - base_val) * intensity
        
        return features


# Test the emotion engine
if __name__ == "__main__": 
    engine = EmotionEngine()
    
    # Test emotion probabilities
    test_probs = {
        'Happy':  0.75,
        'Neutral': 0.15,
        'Sad': 0.05,
        'Angry': 0.03,
        'Fearful':  0.01,
        'Surprised': 0.01,
        'Disgusted': 0.0
    }
    
    result = engine.process_emotion(test_probs)
    print("Emotion Processing Result:")
    print(f"  Emotion: {result['emotion']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Intensity: {result['intensity']:.2f}")
    
    features = engine.get_emotion_features(result)
    print("\nAudio Features:")
    for feature, value in features.items():
        print(f"  {feature}:  {value}")