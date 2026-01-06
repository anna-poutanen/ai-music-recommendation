"""
Fixed Training Pipeline for Emotion Recognition Model
- FER-2013 dataset (7 emotions)
- Proper train/val split BEFORE oversampling
- No data leakage
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from emotion_model import create_emotion_model, EMOTIONS

# ---------------- CONFIG ---------------- #
CONFIG = {
    'data_dir': 'data/fer2013',
    'model_save_path': 'models/emotion_model.h5',
    'output_dir': 'outputs',
    'img_size': 48,
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split':  0.2,
}

# Correct label mapping matching EMOTIONS list
EMOTION_MAP = {
    'angry': 0,      # Angry
    'disgust': 1,    # Disgusted
    'fear':  2,       # Fearful
    'happy': 3,      # Happy
    'neutral': 4,    # Neutral
    'sad': 5,        # Sad
    'surprise': 6    # Surprised
}

# ---------------- DATA LOADING ---------------- #
def load_data(data_dir, subset='train'):
    """Load images from folder structure"""
    subset_dir = os.path.join(data_dir, subset)
    
    X, y = [], []
    print(f"\n📂 Loading {subset} data:")
    
    for emotion_folder, label in EMOTION_MAP.items():
        folder_path = os.path.join(subset_dir, emotion_folder)
        
        if not os.path.exists(folder_path):
            print(f"   ⚠️ {emotion_folder}:  folder not found")
            continue
        
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size']))
                    X.append(img)
                    y.append(label)
                    count += 1
        
        print(f"   {emotion_folder}: {count} images → label {label} ({EMOTIONS[label]})")
    
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    print(f"   Total:  {len(X)} images")
    return X, y


def preprocess(X, y):
    """Normalize and reshape"""
    X = X / 255.0
    X = X.reshape(-1, CONFIG['img_size'], CONFIG['img_size'], 1)
    y = to_categorical(y, num_classes=7)
    return X, y


# ---------------- TRAINING ---------------- #
def train_model():
    """Main training function with proper data handling"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load train and test data
    X_train, y_train = load_data(CONFIG['data_dir'], 'train')
    X_test, y_test = load_data(CONFIG['data_dir'], 'test')
    
    # Split train into train/val BEFORE any processing
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=CONFIG['validation_split'],
        random_state=42,
        stratify=y_train  # Keep class distribution
    )
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train)}")
    print(f"   Val:  {len(X_val)}")
    print(f"   Test: {len(X_test)}")
    
    # Preprocess
    X_train, y_train = preprocess(X_train, y_train)
    X_val, y_val = preprocess(X_val, y_val)
    X_test, y_test = preprocess(X_test, y_test)
    
    # Data augmentation for training only
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    
    # Build model
    print("\n🧠 Building model...")
    model = create_emotion_model(
        input_shape=(CONFIG['img_size'], CONFIG['img_size'], 1),
        num_classes=7
    )
    
    # Use legacy optimizer for M1/M2 Mac
    try:
        optimizer = keras.optimizers.legacy.Adam(learning_rate=CONFIG['learning_rate'])
    except:
        optimizer = keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            CONFIG['model_save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    steps_per_epoch = len(X_train) // CONFIG['batch_size']
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG['epochs'],
        validation_data=(X_val, y_val),  # Use raw validation data, no augmentation
        callbacks=callbacks
    )
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n✅ Test accuracy: {test_acc:.4f}")
    print(f"   Test loss: {test_loss:.4f}")
    
    # Per-class accuracy
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n📈 Per-class accuracy:")
    for i, emotion in enumerate(EMOTIONS):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"   {emotion}: {acc:.2%} ({mask.sum()} samples)")
    
    # Save final model
    model.save('models/emotion_model_final.h5')
    print("\n💾 Model saved!")
    
    return model, history


def plot_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/training_history.png", dpi=150)
    print(f"📊 Saved training plot to {CONFIG['output_dir']}/training_history.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("🎵 EMOTION RECOGNITION MODEL TRAINING")
    print("=" * 60)
    
    model, history = train_model()
    plot_history(history)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)