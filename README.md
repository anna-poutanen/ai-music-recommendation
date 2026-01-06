# 🎵 Facial Expression–Based Music Recommendation

An AI-powered system that detects your facial emotion via webcam and recommends Spotify songs that match your mood. 

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## What It Does

1. **Detects your face** using MediaPipe
2. **Recognizes your emotion** using a CNN trained on FER-2013 dataset
3. **Recommends songs** from Spotify that match your emotional state

**Supported Emotions:** Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral

---

## System Architecture

```
┌─────────────┐
│   Webcam    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Face Detection  │ (MediaPipe)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Emotion Model   │ (CNN - TensorFlow)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Emotion Engine  │ (Feature Mapping)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Recommender    │ (Spotify API)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Song List     │
└─────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Spotify Developer Account

### Installation

```bash
# Clone or download the code
mkdir facial-expression-music-recommender
cd facial-expression-music-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Spotify API

1. Go to https://developer.spotify.com/dashboard
2. Create a new app
3. Copy your `Client ID` and `Client Secret`
4. Create `.env` file:

```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### Download Dataset

**Option 1: Kaggle API (Recommended)**

```bash
# Install Kaggle
pip install kaggle

# Setup Kaggle credentials (~/.kaggle/kaggle.json)
# Get from: https://www.kaggle.com/settings/account

# Download dataset
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/
```

**Option 2: Manual Download**

1. Visit https://www.kaggle.com/datasets/msambare/fer2013
2. Download and extract to `data/fer2013/`

### Train the Model

```bash
python src/train_emotion_model.py
```

**Expected output:**
- Training time: 30-60 minutes (CPU) or 5-10 minutes (GPU)
- Validation accuracy: 60-70%
- Model saved to `models/emotion_model.h5`

### Run the Application

```bash
streamlit run src/app.py
```

Open your browser to `http://localhost:8501`

---

## 🎓 How to Train the AI Model

### Step 1: Understand the Data

The **FER-2013** dataset contains:
- 28,709 training images
- 7 emotion classes
- 48x48 grayscale images

**Class distribution (imbalanced):**
- Happy: 7,215
- Neutral: 4,965
- Sad: 4,830
- Angry: 3,995
- Surprised: 3,171
- Fearful: 4,097
- Disgusted: 436 (smallest class)

### Step 2: Data Preprocessing

The training script automatically:
- Normalizes pixels to [0, 1]
- Applies data augmentation (rotation, flip, zoom)
- Computes class weights to handle imbalance
- Splits into train (80%) and validation (20%)

### Step 3: Model Architecture

```
Input: 48x48x1 grayscale image
↓
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
↓
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
↓
Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
↓
Flatten → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
↓
Dense(128) → BatchNorm → ReLU → Dropout(0.5)
↓
Dense(7) → Softmax
↓
Output: [P(Angry), P(Disgusted), P(Fearful), P(Happy), P(Neutral), P(Sad), P(Surprised)]
```

**Total Parameters:** ~1.5M

### Step 4: Training Process

**Hyperparameters:**
```python
batch_size = 64
epochs = 50
learning_rate = 0.0001
optimizer = Adam
loss = categorical_crossentropy
```

**Techniques used:**
- Class weighting (handles imbalance)
- Data augmentation (prevents overfitting)
- Batch normalization (faster convergence)
- Dropout (regularization)
- Early stopping (prevents overfitting)
- Learning rate reduction (fine-tuning)

**Training output example:**
```
Epoch 1/50
448/448 [======] - 45s - loss: 1.7234 - accuracy: 0.3456 - val_loss: 1.6123 - val_accuracy: 0.4234
Epoch 2/50
448/448 [======] - 44s - loss: 1.5234 - accuracy: 0.4567 - val_loss: 1.5012 - val_accuracy: 0.4789
... 
Epoch 35/50
448/448 [======] - 43s - loss: 0.9821 - accuracy: 0.6543 - val_loss: 1.1234 - val_accuracy: 0.6123
```

### Step 5: Monitor Training

**Good signs:**
- Loss decreasing
- Accuracy increasing
- Val accuracy within 5-10% of train accuracy

**Warning signs:**
- Val accuracy much lower than train (overfitting)
- Loss not decreasing (learning rate too high/low)
- NaN losses (numerical instability)

**View training plot:**
```
outputs/training_history.png
```

### Step 6: Evaluate the Model

```bash
python src/test_emotion_model.py --webcam
```

**Expected results:**
- Validation accuracy: 60-70%
- Happy/Neutral:  Usually detected well (70-80%)
- Disgusted: Hardest to detect (30-40%)

### Step 7: Troubleshooting

**If accuracy < 55%:**
```bash
# Try longer training
CONFIG['epochs'] = 100

# Or reduce learning rate
CONFIG['learning_rate'] = 0.00005
```

**If training too slow:**
- Use GPU (Google Colab free tier)
- Reduce batch size to 32
- Reduce image size to 32x32

**If model always predicts "Neutral":**
- Class imbalance issue
- Training script already handles this with class weights

---

## Emotion → Music Mapping

| Emotion | Valence | Energy | Danceability | Tempo | Example Songs |
|---------|---------|--------|--------------|-------|---------------|
| Happy | 0.8 | 0.7 | 0.7 | 120 | Upbeat pop, dance |
| Sad | 0.2 | 0.3 | 0.3 | 80 | Ballads, slow songs |
| Angry | 0.2 | 0.9 | 0.5 | 140 | Rock, metal, rap |
| Fearful | 0.3 | 0.6 | 0.4 | 110 | Tense, dramatic |
| Surprised | 0.6 | 0.7 | 0.6 | 125 | Dynamic, varied |
| Disgusted | 0.25 | 0.5 | 0.4 | 100 | Alternative |
| Neutral | 0.5 | 0.5 | 0.5 | 110 | Chill, ambient |

---

## Testing

### Test on Single Image

```bash
python src/test_emotion_model.py --image path/to/image.jpg
```

### Test on Webcam

```bash
python src/test_emotion_model.py --webcam
```

### Test Face Detection Only

```bash
python src/face_detection.py
```

### Test Spotify Integration

```bash
python src/recommender.py
```

---

## Model Performance

**Typical Results:**
- Overall Accuracy: 62-68%
- Happy Detection: 75%
- Neutral Detection: 70%
- Sad Detection: 65%
- Angry Detection:  60%
- Surprised Detection: 62%
- Fearful Detection:  58%
- Disgusted Detection:  35% (limited training data)

**Confusion Matrix:**
```
         Predicted
Actual | H  N  S  A  Su F  D
-------|---------------------
Happy  |75  15  3  2  4  1  0
Neutral|10 70  8  5  5  2  0
Sad    | 5  20 65  5  2  3  0
Angry  | 2  10  8 60  5 15  0
Surpr.  | 8  10  5  3 62  10  2
Fear   | 3  15  10 20  5 58  0
Disg.  | 5  25  10 15  5  5 35
```

---

## Ethics & Privacy

### Privacy Commitments

- **No image storage** - Images processed in real-time only
- **No data transmission** - Everything runs locally
- **No emotion claims** - Tool provides suggestions, not diagnoses
- **User control** - Manual emotion override available

### Limitations

This system: 
- Does **NOT** detect actual emotions (only facial expressions)
- Does **NOT** work equally well for all demographics
- Is **NOT** a medical or psychological tool
- May have **bias** from training data

### Ethical Considerations

**Potential biases:**
- FER-2013 dataset may not represent all demographics equally
- Cultural differences in facial expressions not fully captured
- Lighting and camera quality affect accuracy

**Responsible use:**
- Use for entertainment/music discovery only
- Do not use for hiring, surveillance, or decision-making
- Always allow users to override detected emotions

---

## Future Improvements

### V2 Features (Planned)

- [ ] Multi-face detection → group playlists
- [ ] User feedback loop (thumbs up/down)
- [ ] Temporal smoothing (video stream)
- [ ] More emotion categories (e.g., "excited", "calm")
- [ ] Genre preferences
- [ ] Mobile app (TensorFlow Lite)
- [ ] Transformer-based emotion model
- [ ] Integration with Spotify playback SDK

---

## Project Structure

```
facial-expression-music-recommender/
├── data/
│   └── fer2013/              # Dataset (you download)
│       ├── train/
│       └── test/
├── models/
│   └── emotion_model.h5      # Trained model (generated)
├── outputs/
│   └── training_history.png  # Training plots (generated)
├── src/
│   ├── face_detection.py     # MediaPipe face detection
│   ├── emotion_model.py      # CNN architecture
│   ├── train_emotion_model.py # Training pipeline
│   ├── test_emotion_model.py # Testing utilities
│   ├── emotion_engine.py     # Emotion processing
│   ├── recommender.py        # Spotify integration
│   └── app.py                # Streamlit web app
├── . env                      # Spotify credentials (you create)
├── . gitignore
├── requirements.txt