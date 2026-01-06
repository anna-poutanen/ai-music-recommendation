"""
Improved Training Pipeline for Emotion Recognition Model
- FER-2013 dataset (7 emotions)
- Class balancing + data augmentation
- Focal Loss for rare classes
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from emotion_model import create_emotion_model, EMOTIONS

# ---------------- CONFIG ---------------- #
CONFIG = {
    'data_dir': 'data/fer2013',
    'model_save_path': 'models/emotion_model.h5',
    'output_dir': 'outputs',
    'img_size': 48,
    'batch_size':  64,
    'epochs': 50,
    'learning_rate': 1e-4,
    'validation_split': 0.2,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
    'use_focal_loss': True,
    'use_oversampling': True,
}

# ---------------- FOCAL LOSS ---------------- #
def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for multi-class classification
    Helps model focus on hard-to-classify examples
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

# ---------------- DATA LOADING ---------------- #
def load_fer2013_from_folders(data_dir):
    """Load FER-2013 from folder structure"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }

    def load_images(folder_path):
        X, y = [], []
        print(f"\nLoading from {folder_path}:")
        for emotion, label in emotion_map.items():
            path = os.path.join(folder_path, emotion)
            if not os.path.exists(path):
                print(f"  ⚠️  {emotion}:  folder not found")
                continue
            
            count = 0
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size']))
                    X.append(img)
                    y.append(label)
                    count += 1
            
            print(f"  {emotion}: {count} images (label {label})")
        
        return np.array(X, dtype='float32'), np.array(y, dtype='int32')
    
    X_train, y_train = load_images(train_dir)
    X_test, y_test = load_images(test_dir)
    
    print(f"\n✅ Total - Training: {len(X_train)}, Test: {len(X_test)}")
    return X_train, y_train, X_test, y_test

# ---------------- PREPROCESS ---------------- #
def preprocess_data(X_train, y_train, X_test, y_test):
    """Normalize and reshape data"""
    print("\n🔧 Preprocessing data...")
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    X_train = X_train.reshape(-1, CONFIG['img_size'], CONFIG['img_size'], 1)
    X_test = X_test.reshape(-1, CONFIG['img_size'], CONFIG['img_size'], 1)
    
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    
    return X_train, y_train, X_test, y_test

# ---------------- OVERSAMPLING ---------------- #
def balance_dataset(X_train, y_train):
    """Oversample minority classes to balance dataset"""
    if not CONFIG['use_oversampling']: 
        print("\n⚠️  Oversampling disabled")
        return X_train, y_train
    
    print("\n⚖️  Balancing dataset with oversampling...")
    
    y_labels = np.argmax(y_train, axis=1)
    unique, counts = np.unique(y_labels, return_counts=True)
    
    print("Original class distribution:")
    for label, count in zip(unique, counts):
        print(f"  {EMOTIONS[label]}: {count}")
    
    max_count = counts.max()
    
    X_balanced, y_balanced = [], []
    
    for i in range(7):
        idx = np.where(y_labels == i)[0]
        
        if len(idx) == 0:
            continue
        
        repeat_full = max_count // len(idx)
        remainder = max_count % len(idx)
        
        if repeat_full > 0:
            X_balanced.append(np.tile(X_train[idx], (repeat_full, 1, 1, 1)))
            y_balanced.append(np.tile(y_train[idx], (repeat_full, 1)))
        
        if remainder > 0:
            random_idx = np.random.choice(idx, size=remainder, replace=False)
            X_balanced.append(X_train[random_idx])
            y_balanced.append(y_train[random_idx])
    
    X_balanced = np.concatenate(X_balanced, axis=0)
    y_balanced = np.concatenate(y_balanced, axis=0)
    
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    print(f"\n✅ Balanced dataset size: {len(X_balanced)}")
    
    y_bal_labels = np.argmax(y_balanced, axis=1)
    unique_bal, counts_bal = np.unique(y_bal_labels, return_counts=True)
    print("Balanced class distribution:")
    for label, count in zip(unique_bal, counts_bal):
        print(f"  {EMOTIONS[label]}: {count}")
    
    return X_balanced, y_balanced

# ---------------- DATA AUGMENTATION ---------------- #
def create_generators(X_train, y_train):
    """Create training and validation generators"""
    print("\n🎨 Creating data generators...")
    
    X_train_bal, y_train_bal = balance_dataset(X_train, y_train)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_bal, y_train_bal,
        test_size=CONFIG['validation_split'],
        random_state=42,
        stratify=np.argmax(y_train_bal, axis=1)
    )
    
    print(f"  Training samples: {len(X_tr)}")
    print(f"  Validation samples: {len(X_val)}")
    
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_gen = train_datagen.flow(
        X_tr, y_tr,
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    
    val_gen = val_datagen.flow(
        X_val, y_val,
        batch_size=CONFIG['batch_size'],
        shuffle=False
    )
    
    return train_gen, val_gen

# ---------------- CLASS WEIGHTS ---------------- #
def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    print("\n⚖️  Computing class weights...")
    
    y_int = np.argmax(y_train, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
    class_weight_dict = {i:  w for i, w in enumerate(weights)}
    
    for i, w in class_weight_dict.items():
        print(f"  {EMOTIONS[i]}: {w:.2f}")
    
    return class_weight_dict

# ---------------- TRAIN MODEL ---------------- #
def train_model():
    """Main training function"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_fer2013_from_folders(CONFIG['data_dir'])
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    train_gen, val_gen = create_generators(X_train, y_train)
    
    class_weights = compute_class_weights(y_train)
    
    print("\n🧠 Creating model...")
    model = create_emotion_model(
        input_shape=(CONFIG['img_size'], CONFIG['img_size'], 1),
        num_classes=7
    )
    
    if CONFIG['use_focal_loss']: 
        loss_fn = focal_loss(gamma=CONFIG['focal_gamma'], alpha=CONFIG['focal_alpha'])
        print(f"  Using Focal Loss (gamma={CONFIG['focal_gamma']}, alpha={CONFIG['focal_alpha']})")
    else:
        loss_fn = 'categorical_crossentropy'
        print("  Using Categorical Cross-Entropy")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    model.summary()
    
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
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        class_weight=class_weights if not CONFIG['use_oversampling'] else None
    )
    
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n✅ Test accuracy: {test_acc:.4f}")
    print(f"   Test loss: {test_loss:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n📈 Per-class performance:")
    for i, emotion in enumerate(EMOTIONS):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"  {emotion}: {acc:.2%} ({mask.sum()} samples)")
    
    model.save('models/emotion_model_final.h5')
    print("\n💾 Model saved to models/emotion_model_final.h5")
    
    return model, history

# ---------------- PLOT HISTORY ---------------- #
def plot_training_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/training_history.png", dpi=150)
    print(f"📊 Training plots saved to {CONFIG['output_dir']}/training_history.png")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("=" * 60)
    print("🎵 EMOTION RECOGNITION MODEL TRAINING")
    print("   Strategy:  Focal Loss + Oversampling")
    print("=" * 60)
    
    model, history = train_model()
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Test with:  python3 src/test_emotion_model.py --webcam")
    print("  2. Run app: streamlit run src/app.py")