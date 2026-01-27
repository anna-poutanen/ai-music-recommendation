"""
Streamlit Web Application
Facial Expression-Based Music Recommendation
"""

# Disable Metal GPU - must be before TensorFlow import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Force CPU-only mode for TensorFlow
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from face_detection import FaceDetector
from emotion_model import EmotionClassifier
from emotion_engine import EmotionEngine
from local_recommender import LocalMusicRecommender

# Try to import Spotify recommender
SPOTIFY_AVAILABLE = False
try: 
    from recommender import MusicRecommender
    SPOTIFY_AVAILABLE = True
except ImportError:
    pass


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
        border: 1px solid rgba(0, 113, 227, 0.2);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem auto 2rem auto;
        max-width: 800px;
        font-size: 0.9rem;
        color: #1d1d1f;
        line-height: 1.6;
    }
    
    .privacy-notice strong {
        font-weight: 600;
        color: #0071e3;
    }
    
    /* Local Mode Notice */
    .notice-box {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem auto 2rem auto;
        max-width: 800px;
        border: 1px solid;
    }
    
    .notice-info {
        background: rgba(255, 149, 0, 0.06);
        border-color: rgba(255, 149, 0, 0.2);
    }
    
    .notice-title {
        font-weight: 600;
        color: #ff9500;
        margin-bottom: 0.25rem;
    }
    
    .notice-text {
        font-size: 0.9rem;
        color: #1d1d1f;
        line-height: 1.6;
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
    .emotion-display {
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
    
    .emotion-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #86868b;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .emotion-value {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }
    
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #000000 0%, #434343 100%);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Emotion Probabilities Table */
    .prob-table {
        margin: 1.5rem 0;
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #f5f5f7;
    }
    
    .prob-row {
        display: flex;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f5f5f7;
    }
    
    .prob-row:last-child {
        border-bottom: none;
    }
    
    .prob-label {
        width: 120px;
        font-weight: 500;
        color: #1d1d1f;
        font-size: 0.95rem;
    }
    
    .prob-bar-container {
        flex-grow: 1;
        background: #f5f5f7;
        border-radius: 8px;
        height: 8px;
        margin: 0 1rem;
        overflow: hidden;
    }
    
    .prob-bar {
        background: linear-gradient(90deg, #000000, #434343);
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    
    .prob-value {
        width: 50px;
        text-align: right;
        color: #86868b;
        font-size: 0.9rem;
        font-weight: 500;
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
        display: flex;
        align-items: center;
        gap: 1.25rem;
    }
    
    .track-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.08);
        border-color: #d2d2d7;
    }
    
    .track-number {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #000000, #434343);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
        flex-shrink: 0;
    }
    
    .track-info {
        flex-grow: 1;
        min-width: 0;
    }
    
    .track-name {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }
    
    .track-artist {
        color: #86868b;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .track-meta {
        text-align: right;
        flex-shrink: 0;
    }
    
    .match-score {
        background: linear-gradient(135deg, #000000 0%, #434343 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
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
    
    .footer-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #f5f5f7;
        padding: 0.6rem 1.25rem;
        border-radius: 20px;
        margin-bottom: 0.75rem;
        color: #1d1d1f;
        font-weight: 500;
    }
    
    /* Button Styling */
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
    
    /* Link Button */
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
    
    /* Camera Input Styling */
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
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e5e5e7;
        margin: 2rem 0;
    }
    
    /* Column gap adjustment */
    [data-testid="column"] {
        padding: 0 1rem;
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
    st.session_state.recommender = LocalMusicRecommender()
    st.session_state.using_local = True


def render_track_card(index: int, track: dict):
    """Render a styled track card."""
    score_percent = int(track['score'] * 100)
    
    st.markdown(f"""
        <div class="track-card">
            <div class="track-number">{index}</div>
            <div class="track-info">
                <div class="track-name">{track['name']}</div>
                <div class="track-artist">{track['artists']} • {track['album']}</div>
            </div>
            <div class="track-meta">
                <span class="match-score">{score_percent}% match</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Play button
    if "spotify.com" in track['url']: 
        button_text = "Play on Spotify"
    else:
        button_text = "Find on YouTube"
    
    st.link_button(button_text, track['url'], use_container_width=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">Emotion Music</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Music that matches your mood, powered by AI</p>', unsafe_allow_html=True)
    
    # Mode Banner
    if st.session_state.get('using_local', False):
        st.markdown("""
            <div class="notice-box notice-info">
                <div class="notice-title">Local Mode</div>
                <div class="notice-text">
                    Using local song database. Recommendations will open in YouTube search.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Privacy Notice - Always Visible
    st.markdown("""
        <div class="privacy-notice">
            <strong>Privacy First</strong> — All processing happens on your device. No images are stored or transmitted. 
            Emotion detection is for entertainment purposes only.
        </div>
    """, unsafe_allow_html=True)
    
    # Model Check
    if st.session_state.classifier is None:
        st.error("Emotion model not found. Please train the model first.")
        st.code("python src/train_emotion_model.py", language="bash")
        return
    
    if st.session_state.recommender is None:
        st.error("Recommender setup error")
        return
    
    # Main Layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">Capture Your Expression</div>', unsafe_allow_html=True)
        
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
        
        if camera_image is not None:
            # Process image
            image = Image.open(camera_image)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect face
            face_crop, bbox = st.session_state.detector.detect_face(image_bgr)
            
            if face_crop is None:
                st.warning("No face detected. Try again with better lighting or face the camera directly.")
            else:
                # Predict emotion
                emotion_probs = st.session_state.classifier.predict_emotion(face_crop)
                emotion_data = st.session_state.engine.process_emotion(emotion_probs)
                
                emotion = emotion_data['emotion']
                confidence = emotion_data['confidence']
                
                # Emotion Display
                st.markdown(f"""
                    <div class="emotion-display">
                        <div class="emotion-label">Detected Emotion</div>
                        <div class="emotion-value">{emotion}</div>
                        <span class="confidence-badge">{confidence:.0%} confidence</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Emotion probabilities - Always visible, styled
                st.markdown("#### All Emotion Probabilities")
                
                # Build the table HTML
                sorted_probs = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                
                rows = []
                for emo, prob in sorted_probs:
                    prob_percentage = prob * 100
                    row = f'<div class="prob-row"><div class="prob-label">{emo}</div><div class="prob-bar-container"><div class="prob-bar" style="width: {prob_percentage:.1f}%"></div></div><div class="prob-value">{prob:.1%}</div></div>'
                    rows.append(row)
                
                prob_table_html = '<div class="prob-table">' + ''.join(rows) + '</div>'
                
                st.markdown(prob_table_html, unsafe_allow_html=True)
                
                # Override option
                st.markdown("#### Adjust Mood (Optional)")
                override_emotion = st.selectbox(
                    "Choose a different emotion",
                    ['Use Detected'] + list(emotion_probs.keys()),
                    help="Not feeling the detected emotion? Choose your own.",
                    label_visibility="collapsed"
                )
                
                if override_emotion != 'Use Detected':
                    emotion_data['emotion'] = override_emotion
                    st.info(f"Using: **{override_emotion}**")
                
                # Get recommendations button
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Get Music Recommendations", use_container_width=True):
                    with st.spinner("Finding the perfect songs..."):
                        emotion_features = st.session_state.engine.get_emotion_features(emotion_data)
                        recommendations = st.session_state.recommender.get_recommendations(
                            emotion_features, num_tracks=5
                        )
                        st.session_state.recommendations = recommendations
                        st.session_state.current_emotion = emotion_data['emotion']
    
    with col2:
        st.markdown('<div class="section-header">Your Playlist</div>', unsafe_allow_html=True)
        
        if 'recommendations' in st.session_state and st.session_state.recommendations:
            st.success(f"Curated for your **{st.session_state.current_emotion}** mood")
            
            for i, track in enumerate(st.session_state.recommendations, 1):
                render_track_card(i, track)
                
                # Show album art if available
                if track.get('image_url'):
                    st.image(track['image_url'])
                
                if i < len(st.session_state.recommendations):
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-text">No recommendations yet</div>
                    <div class="empty-state-hint">Capture your expression to discover music</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <div class="footer-badge">
                Powered by TensorFlow & Streamlit
            </div>
            <div style="color: #a1a1a6;">Emotion detection trained on FER-2013 dataset</div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()