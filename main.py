import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_PATH = 'data/'
# We make every image 128 pixels tall (pitch) and 128 pixels wide (time)
IMG_SIZE = (128, 128) 

def load_and_preprocess_data():
    X = [] # This will store our "Images"
    y = [] # This will store our "Emotions" (0 to 7)
    
    print("Step 1: Processing audio files into images...")
    
    for actor_folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, actor_folder)
        if not os.path.isdir(folder_path): continue
        
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                # 1. Get the emotion label (3rd number in filename)
                # Filename: 03-01-05-01... -> "05" is Angry. 
                # We subtract 1 because computers count from 0.
                emotion = int(file.split('-')[2]) - 1
                
                # 2. Load audio (only first 3 seconds to keep it fast)
                path = os.path.join(folder_path, file)
                audio, sr = librosa.load(path, duration=3, res_type='soxr_hq')
                
                # 3. Trim silence & Convert to Mel-Spectrogram
                audio, _ = librosa.effects.trim(audio)
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
                log_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 4. Make sure every image is exactly 128x128 (Padding/Clipping)
                if log_spec.shape[1] < 128:
                    log_spec = np.pad(log_spec, ((0,0), (0, 128 - log_spec.shape[1])))
                else:
                    log_spec = log_spec[:, :128]
                
                X.append(log_spec)
                y.append(emotion)
                
    return np.array(X), np.array(y)

# --- EXECUTION ---

# 1. Load the data
X, y = load_and_preprocess_data()
# Reshape for the CNN: (Number of samples, Height, Width, Channels)
X = X.reshape(X.shape[0], 128, 128, 1)

# 2. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the CNN "Brain"
# 
model = models.Sequential([
    # First layer: Looks for simple patterns
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second layer: Looks for more complex emotional patterns
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3), # Prevents the model from memorizing
    
    # Flatten: Turns the 2D image into a long list of numbers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Output: 8 emotions (Neutral, Happy, Sad, etc.)
    layers.Dense(8, activation='softmax') 
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Start Training
print("\nStep 2: Training the AI... (This will take about 15-30 mins on a Mac)")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 5. Save the Results
model.save('emotion_model.h5')
print("\nSUCCESS! The model is saved as 'emotion_model.h5'")

# Optional: Plot the training accuracy to see if it learned
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()