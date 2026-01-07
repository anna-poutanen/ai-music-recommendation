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
    page_title="Emotion Music Recommender",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 50%, #169c46 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Cards */
    .card {
        background:  linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        border:  1px solid #e5e7eb;
    }
    
    /* Emotion Display */
    .emotion-display {
        background:  linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin:  1.5rem 0;
        box-shadow: 0 10px 25px -5px rgba(29, 185, 84, 0.4);
    }
    
    .emotion-label {
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .emotion-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.375rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        backdrop-filter: blur(4px);
    }
    
    /* Track Card */
    .track-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin:  0.75rem 0;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .track-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #1DB954;
    }
    
    .track-number {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #1DB954, #1ed760);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
        flex-shrink: 0;
    }
    
    .track-info {
        flex-grow: 1;
        min-width: 0;
    }
    
    .track-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow:  ellipsis;
    }
    
    .track-artist {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    .track-meta {
        text-align: right;
        flex-shrink: 0;
    }
    
    .match-score {
        background: #ecfdf5;
        color: #059669;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Notice Boxes */
    .notice-box {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .notice-content {
        flex-grow: 1;
    }
    
    .notice-title {
        font-weight: 600;
        margin-bottom:  0.25rem;
    }
    
    .notice-text {
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    .notice-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border:  1px solid #93c5fd;
    }
    
    .notice-info .notice-title { color: #1d4ed8; }
    .notice-info .notice-text { color: #1e40af; }
    
    .notice-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
    }
    
    .notice-warning.notice-title { color: #b45309; }
    .notice-warning.notice-text { color: #92400e; }
    
    /* Section Headers */
    .section-header {
        font-size:  1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    /* Empty State */
    .empty-state {
        text-align:  center;
        padding: 3rem 2rem;
        color: #6b7280;
    }
    
    .empty-state-text {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-hint {
        font-size: 0.9rem;
        opacity: 0.7;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #9ca3af;
        font-size: 0.875rem;
    }
    
    .footer-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #f3f4f6;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border:  none;
        border-radius: 9999px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size:  1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px 0 rgba(29, 185, 84, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(29, 185, 84, 0.5);
    }
    
    /* Camera Input Styling */
    .stCameraInput > div {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        border-radius: 9999px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #374151;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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


def render_track_card(index:  int, track: dict):
    """Render a styled track card."""
    score_percent = int(track['score'] * 100)
    
    st.markdown(f"""
        <div class="track-card">
            <div class="track-number">{index}</div>
            <div class="track-info">
                <div class="track-name">{track['name']}</div>
                <div class="track-artist">{track['artists']} | {track['album']}</div>
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
    st.markdown('<h1 class="main-header">Emotion Music Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Let your face pick the playlist</p>', unsafe_allow_html=True)
    
    # Mode Banner
    if st.session_state.get('using_local', False):
        st.markdown("""
            <div class="notice-box notice-info">
                <div class="notice-content">
                    <div class="notice-title">Local Mode Active</div>
                    <div class="notice-text">
                        Using local song database. Songs will open in YouTube search.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Privacy Notice (collapsible)
    with st.expander("Privacy & Ethics Notice"):
        st.markdown("""
        - **No images are stored** — All processing happens locally
        - **Emotion detection is approximate** — For entertainment purposes only
        - **You're in control** — Override detected emotions anytime
        - **No claims made** — This tool doesn't define your actual emotional state
        """)
    
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
                st.markdown("""
                    <div class="notice-box notice-warning">
                        <div class="notice-content">
                            <div class="notice-title">No Face Detected</div>
                            <div class="notice-text">
                                Try again with better lighting or face the camera directly.
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Show detected face
                image_with_box = st.session_state.detector.draw_face_box(image_bgr.copy(), bbox)
                image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
                st.image(image_with_box_rgb, caption="Face detected", use_container_width=None)
                
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
                        <span class="confidence-badge">Confidence: {confidence:.0%}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability breakdown
                with st.expander("View All Emotion Probabilities"):
                    for emo, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.progress(prob)
                        with col_b:
                            st.caption(f"{emo}: {prob:.0%}")
                
                # Override option
                st.markdown("---")
                override_emotion = st.selectbox(
                    "Override emotion (optional)",
                    ['Use Detected'] + list(emotion_probs.keys()),
                    help="Not feeling the detected emotion? Choose your own."
                )
                
                if override_emotion != 'Use Detected':
                    emotion_data['emotion'] = override_emotion
                    st.info(f"Using:  **{override_emotion}**")
                
                # Get recommendations button
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Get Music Recommendations", type="primary", use_container_width=True):
                    with st.spinner("Finding perfect songs for your mood..."):
                        emotion_features = st.session_state.engine.get_emotion_features(emotion_data)
                        recommendations = st.session_state.recommender.get_recommendations(
                            emotion_features, num_tracks=5
                        )
                        st.session_state.recommendations = recommendations
                        st.session_state.current_emotion = emotion_data['emotion']
    
    with col2:
        st.markdown('<div class="section-header">Your Personalized Playlist</div>', unsafe_allow_html=True)
        
        if 'recommendations' in st.session_state and st.session_state.recommendations:
            st.success(f"Songs curated for your **{st.session_state.current_emotion}** mood")
            
            for i, track in enumerate(st.session_state.recommendations, 1):
                render_track_card(i, track)
                
                # Show album art if available
                if track.get('image_url'):
                    st.image(track['image_url'], width=150)
        else:
            st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-text">No recommendations yet</div>
                    <div class="empty-state-hint">Take a photo to discover music that matches your mood</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <div class="footer-badge">
                <span>Powered by TensorFlow & Streamlit</span>
            </div>
            <div>Emotion detection trained on FER-2013 dataset</div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()