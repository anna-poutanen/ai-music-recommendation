"""
Streamlit Web Application
Facial Expression-Based Music Recommendation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from face_detection import FaceDetector
from emotion_model import EmotionClassifier
from emotion_engine import EmotionEngine
from recommender import MusicRecommender


# Page config
st.set_page_config(
    page_title="🎵 Emotion Music Recommender",
    page_icon="🎵",
    layout="wide"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align:  center;
        margin-bottom:  2rem;
    }
    . emotion-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius:  10px;
        text-align: center;
        font-size: 2rem;
        margin: 20px 0;
    }
    .track-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius:  10px;
        margin: 10px 0;
        border-left: 4px solid #1DB954;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 15px;
        border-radius:  5px;
        border-left: 4px solid #ffc107;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = FaceDetector()
if 'classifier' not in st.session_state:
    if os.path.exists('models/emotion_model.h5'):
        st.session_state.classifier = EmotionClassifier('models/emotion_model.h5')
    else:
        st.session_state.classifier = None
if 'engine' not in st.session_state:
    st.session_state.engine = EmotionEngine()
if 'recommender' not in st. session_state:
    try:
        st.session_state.recommender = MusicRecommender()
    except ValueError as e:
        st.session_state.recommender = None
        st.session_state.spotify_error = str(e)


def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Facial Expression Music Recommender</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
        <div class="disclaimer">
            <strong>Privacy & Ethics Notice</strong><br>
            • No images are stored or transmitted<br>
            • Emotion detection is approximate and for entertainment purposes<br>
            • You can always override the detected emotion<br>
            • This tool does not make claims about your actual emotional state
        </div>
    """, unsafe_allow_html=True)
    
    # Check if model exists
    if st.session_state.classifier is None:
        st.error("Emotion model not found!  Please train the model first.")
        st.code("python src/train_emotion_model.py")
        return
    
    # Check Spotify credentials
    if st.session_state.recommender is None:
        st. error(f"{st.session_state.get('spotify_error', 'Spotify setup error')}")
        st.info("Please set up your Spotify credentials in the . env file")
        return
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Capture Your Expression")
        
        # Camera input
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            # Convert to OpenCV format
            image = Image.open(camera_image)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect face
            face_crop, bbox = st.session_state.detector.detect_face(image_bgr)
            
            if face_crop is None:
                st.warning("⚠️ No face detected.  Please try again with better lighting.")
            else:
                # Draw bounding box
                image_with_box = st.session_state.detector.draw_face_box(
                    image_bgr. copy(), bbox
                )
                image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
                st.image(image_with_box_rgb, caption="Face Detection", use_container_width=True)
                
                # Predict emotion
                emotion_probs = st.session_state.classifier.predict_emotion(face_crop)
                emotion_data = st.session_state.engine.process_emotion(emotion_probs)
                
                # Display emotion
                emotion = emotion_data['emotion']
                confidence = emotion_data['confidence']
                
                st.markdown(f"""
                    <div class="emotion-box">
                        Detected Emotion: <strong>{emotion}</strong><br>
                        <small>Confidence: {confidence:.0%}</small>
                    </div>
                """, unsafe_allow_html=True)
                
                # Emotion probabilities
                with st.expander("View All Emotion Probabilities"):
                    for emo, prob in sorted(emotion_probs.items(), 
                                           key=lambda x: x[1], reverse=True):
                        st.progress(prob, text=f"{emo}: {prob:.1%}")
                
                # Emotion override
                st.subheader("Override Emotion (Optional)")
                override_emotion = st.selectbox(
                    "Select a different emotion:",
                    ['Use Detected'] + list(emotion_probs. keys())
                )
                
                if override_emotion != 'Use Detected':
                    emotion_data['emotion'] = override_emotion
                    st.info(f"Using emotion: {override_emotion}")
                
                # Get music recommendations
                if st.button("Get Music Recommendations", type="primary"):
                    with st.spinner("Finding the perfect songs for you..."):
                        # Get emotion features
                        emotion_features = st.session_state.engine.get_emotion_features(
                            emotion_data
                        )
                        
                        # Get recommendations
                        recommendations = st.session_state.recommender.get_recommendations(
                            emotion_features, num_tracks=5
                        )
                        
                        # Store in session state
                        st.session_state.recommendations = recommendations
                        st.session_state.current_emotion = emotion_data['emotion']
    
    with col2:
        st.subheader("🎵 Your Personalized Playlist")
        
        if 'recommendations' in st.session_state:
            st.success(f"✨ Songs for your **{st.session_state.current_emotion}** mood:")
            
            for i, track in enumerate(st.session_state.recommendations, 1):
                with st.container():
                    st.markdown(f"""
                        <div class="track-card">
                            <h3>{i}. {track['name']}</h3>
                            <p><strong>Artist:</strong> {track['artists']}</p>
                            <p><strong>Album:</strong> {track['album']}</p>
                            <p><strong>Match Score:</strong> {track['score']:.0%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Spotify link
                    st.link_button(
                        "Play on Spotify",
                        track['url'],
                        use_container_width=True
                    )
                    
                    # Album artwork
                    if track['image_url']:
                        st. image(track['image_url'], width=200)
                    
                    st.divider()
        else:
            st.info("👈 Take a photo to get personalized music recommendations!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            Made using Streamlit, TensorFlow, and Spotify API<br>
            <small>Emotion detection powered by CNN trained on FER-2013 dataset</small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__": 
    main()