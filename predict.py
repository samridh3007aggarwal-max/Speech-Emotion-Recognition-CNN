import librosa
import numpy as np
import tensorflow as tf

# 1. Load the "Brain" you just built
model = tf.keras.models.load_model('emotion_model.h5')

# 2. Emotion list (Must be in this exact order)
emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

def predict_emotion(file_path):
    # Process the audio exactly like we did during training
    audio, sr = librosa.load(file_path, duration=3)
    audio, _ = librosa.effects.trim(audio)
    
    # Convert to Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Pad or clip to 128x128
    if log_spec.shape[1] < 128:
        log_spec = np.pad(log_spec, ((0,0), (0, 128 - log_spec.shape[1])))
    else:
        log_spec = log_spec[:, :128]
        
    # Shape it for the model
    log_spec = log_spec.reshape(1, 128, 128, 1)
    
    # Make prediction
    prediction = model.predict(log_spec)
    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return emotions[index], confidence

# Test it
file_to_test = 'data/Actor_01/03-01-05-01-01-01-01.wav' # Change to any file
result, conf = predict_emotion(file_to_test)
print(f"Predicted Emotion: {result} ({conf:.2f}% confidence)")