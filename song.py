import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# --- Project Setup ---

# Path to dataset (each subfolder in 'genres/' is a genre name)
DATA_PATH = "genres"

# Get list of genre names from folder names
genres = [g for g in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, g))]

# --- Feature Extraction Function ---

def extract_features(file_path):
    """
    Extract MFCC features from an audio file.

    Args:
        file_path (str): Path to the .wav file

    Returns:
        np.ndarray: 13-dimensional mean MFCC features
    """
    y, sr = librosa.load(file_path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# --- Prepare Data ---

X = []
y = []

for idx, genre in enumerate(genres):
    genre_path = os.path.join(DATA_PATH, genre)
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(idx)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(genres))

# --- Train/Test Split ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Build Model ---

model = Sequential([
    Input(shape=(13,)),              # Input: 13 MFCC features
    Dense(100, activation='relu'),   # Hidden layer 1
    Dense(50, activation='relu'),    # Hidden layer 2
    Dense(len(genres), activation='softmax')  # Output: Genre classification
])

# Compile
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# --- Train the Model ---

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# --- Evaluate ---

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# --- Save Model (Binary + JSON Format) ---

model.save("music_genre_model.keras")
print("Saved trained model to music_genre_model.keras")

model_json = model.to_json()
with open("model_config.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model architecture to model_config.json")
