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
    page_title="Emotion Music",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Apple-inspired Custom CSS
st.markdown("""
    <style>
    /* Import SF Pro Display-like font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #fafafa 0%, #ffffff 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #000000 0%, #434343 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0 0.5rem 0;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.25rem;
        color: #6e6e73;
        font-weight: 400;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    /* Privacy Notice */
    .privacy-notice {
        background: rgba(0, 113, 227, 0.06);
        border: 1px solid rgba(0, 113, 227, 0.15);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem auto;
        max-width: 900px;
        color: #1d1d1f;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .privacy-notice strong {
        font-weight: 600;
        color: #0071e3;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Emotion Display */
    .emotion-box {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(209, 213, 219, 0.3);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
    }
    
    .emotion-text {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .confidence-text {
        font-size: 1rem;
        color: #86868b;
        font-weight: 500;
    }
    
    /* Emotion Probabilities */
    .prob-container {
        background: #ffffff;
        border: 1px solid #e5e5e7;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .prob-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    
    .prob-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f5f5f7;
    }
    
    .prob-item:last-child {
        border-bottom: none;
    }
    
    .prob-label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #1d1d1f;
    }
    
    .prob-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #86868b;
    }
    
    .prob-bar {
        flex-grow: 1;
        height: 6px;
        background: #f5f5f7;
        border-radius: 3px;
        margin: 0 1rem;
        overflow: hidden;
    }
    
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #000000, #434343);
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    /* Track Card */
    .track-card {
        background: #ffffff;
        border: 1px solid #e5e5e7;
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .track-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.08);
        border-color: #d2d2d7;
    }
    
    .track-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .track-info {
        font-size: 0.95rem;
        color: #86868b;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    
    .match-score {
        display: inline-block;
        background: linear-gradient(135deg, #000000 0%, #434343 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: #000000;
        color: white;
        border: none;
        border-radius: 980px;
        padding: 0.85rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        letter-spacing: -0.01em;
    }
    
    .stButton > button:hover {
        background: #1d1d1f;
        transform: scale(1.02);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
    }
    
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Link Button (Spotify) */
    .stLinkButton > a {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%) !important;
        color: white !important;
        border: none;
        border-radius: 980px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: inline-block;
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.25);
    }
    
    .stLinkButton > a:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.35);
    }
    
    /* Camera Input */
    .stCameraInput > div {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid #e5e5e7;
    }
    
    /* Image */
    img {
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 1px solid #e5e5e7;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 16px;
        border: 1px solid #e5e5e7;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #86868b;
    }
    
    .empty-state-text {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        color: #1d1d1f;
    }
    
    .empty-state-hint {
        font-size: 0.95rem;
        color: #86868b;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0 2rem 0;
        color: #86868b;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e5e5e7;
        margin: 2rem 0;
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
if 'recommender' not in st.session_state:
    try:
        st.session_state.recommender = MusicRecommender()
    except ValueError as e:
        st.session_state.recommender = None
        st.session_state.spotify_error = str(e)


def main():
    # Header
    st.markdown('<h1 class="main-header">Emotion Music</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Music that matches your mood, powered by AI</p>', unsafe_allow_html=True)
    
    # Privacy Notice - Always visible
    st.markdown("""
        <div class="privacy-notice">
            <strong>Privacy First:</strong> All processing happens on your device. No images are stored or transmitted. 
            Emotion detection is for entertainment purposes only.
        </div>
    """, unsafe_allow_html=True)
    
    # Check if model exists
    if st.session_state.classifier is None:
        st.error("Emotion model not found. Please train the model first.")
        st.code("python src/train_emotion_model.py", language="bash")
        return
    
    # Check Spotify credentials
    if st.session_state.recommender is None:
        st.error(f"{st.session_state.get('spotify_error', 'Spotify setup error')}")
        st.info("Please set up your Spotify credentials in the .env file")
        return
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main layout - Two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">Capture Your Expression</div>', unsafe_allow_html=True)
        
        # Camera input
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
        
        if camera_image is not None:
            # Convert to OpenCV format
            image = Image.open(camera_image)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect face
            face_crop, bbox = st.session_state.detector.detect_face(image_bgr)
            
            if face_crop is None:
                st.warning("No face detected. Please try again with better lighting.")
            else:
                # Draw bounding box
                image_with_box = st.session_state.detector.draw_face_box(image_bgr.copy(), bbox)
                image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
                st.image(image_with_box_rgb, use_column_width=True)
                
                # Predict emotion
                emotion_probs = st.session_state.classifier.predict_emotion(face_crop)
                emotion_data = st.session_state.engine.process_emotion(emotion_probs)
                
                # Display emotion
                emotion = emotion_data['emotion']
                confidence = emotion_data['confidence']
                
                st.markdown(f"""
                    <div class="emotion-box">
                        <div class="emotion-text">{emotion}</div>
                        <div class="confidence-text">{confidence:.0%} confidence</div>
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if camera_image is not None and face_crop is not None:
            st.markdown('<div class="section-header">Emotion Analysis</div>', unsafe_allow_html=True)
            
            # Emotion probabilities - Always visible, styled nicely
            sorted_probs = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            
            prob_html = '<div class="prob-container"><div class="prob-title">All Detected Emotions</div>'
            for emo, prob in sorted_probs:
                prob_html += f'''
                <div class="prob-item">
                    <span class="prob-label">{emo}</span>
                    <div class="prob-bar">
                        <div class="prob-bar-fill" style="width: {prob*100}%"></div>
                    </div>
                    <span class="prob-value">{prob:.0%}</span>
                </div>
                '''
            prob_html += '</div>'
            st.markdown(prob_html, unsafe_allow_html=True)
            
            # Emotion override
            st.markdown("<br>", unsafe_allow_html=True)
            override_emotion = st.selectbox(
                "Adjust mood (optional)",
                ['Use Detected'] + list(emotion_probs.keys())
            )
            
            if override_emotion != 'Use Detected':
                emotion_data['emotion'] = override_emotion
                st.info(f"Using: **{override_emotion}**")
            
            # Get music recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Get Music Recommendations", use_container_width=True):
                with st.spinner("Finding the perfect songs..."):
                    # Get emotion features
                    emotion_features = st.session_state.engine.get_emotion_features(emotion_data)
                    
                    # Get recommendations
                    recommendations = st.session_state.recommender.get_recommendations(
                        emotion_features, num_tracks=5
                    )
                    
                    # Store in session state
                    st.session_state.recommendations = recommendations
                    st.session_state.current_emotion = emotion_data['emotion']
    
    # Recommendations section - Full width below
    if 'recommendations' in st.session_state and st.session_state.recommendations:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Your Playlist</div>', unsafe_allow_html=True)
        st.success(f"Curated for your **{st.session_state.current_emotion}** mood")
        
        # Display tracks in a grid
        for i, track in enumerate(st.session_state.recommendations, 1):
            col_track, col_img = st.columns([2, 1])
            
            with col_track:
                st.markdown(f"""
                    <div class="track-card">
                        <div class="track-title">{track['name']}</div>
                        <div class="track-info">{track['artists']}</div>
                        <div class="track-info">{track['album']}</div>
                        <span class="match-score">{track['score']:.0%} match</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.link_button(
                    "Play on Spotify",
                    track['url'],
                    use_container_width=True
                )
            
            with col_img:
                if track['image_url']:
                    st.image(track['image_url'], use_column_width=True)
            
            if i < len(st.session_state.recommendations):
                st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            Powered by TensorFlow and Spotify<br>
            <small style="color: #a1a1a6;">Emotion detection trained on FER-2013 dataset</small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__": 
    main()