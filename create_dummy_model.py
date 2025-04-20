"""
This script creates a simple dummy Random Forest model for genre prediction
to use if you don't have the actual trained model file.
This is compatible with older scikit-learn versions that don't have 'feature_names_in_'.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

# Create a dummy dataset to train the model
np.random.seed(42)

# Define genres
genres = [
    'acoustic', 'afrobeat', 'alternative', 'blues', 'children', 
    'chill', 'club', 'country', 'dance', 'disco', 'disney', 
    'edm', 'electro', 'emo', 'funk', 'groove', 'happy', 
    'house', 'jazz', 'pop'
]

# Define features
features = [
    'energy', 'mode', 'key', 'valence', 'tempo',
    'emotion_joy', 'emotion_anger', 'emotion_fear', 'emotion_sadness', 'emotion_love', 'emotion_surprise',
    'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu',
    'textblob_polarity', 'textblob_subjectivity',
    'joy_to_sadness_ratio', 'anger_to_love_ratio', 'energy_to_valence_ratio', 'surprise_to_fear_ratio',
    'energy_x_tempo', 'energy_x_valence', 'joy_x_tempo', 'sadness_x_valence',
    'energy_squared', 'tempo_log', 'valence_squared',
    'total_emotion_intensity', 'positive_emotions', 'negative_emotions', 'emotion_ratio', 'emotional_diversity'
]

# Add dominant emotion features
for emotion in ['joy', 'anger', 'fear', 'sadness', 'love', 'surprise']:
    features.append(f'dominant_{emotion}')

# Create a dummy dataset with 1000 samples
n_samples = 1000
X = np.random.rand(n_samples, len(features))

# Adjust some features to be more realistic
X[:, features.index('mode')] = np.random.choice([0, 1], size=n_samples)  # Mode is binary
X[:, features.index('key')] = np.random.randint(0, 12, size=n_samples)  # Key is 0-11
X[:, features.index('tempo')] = np.random.uniform(60, 200, size=n_samples)  # Tempo is typically 60-200 BPM

# Create somewhat meaningful dominant emotion features
# Ensure only one dominant emotion is 1, rest are 0
for i in range(n_samples):
    dom_idx = np.random.randint(0, 6)
    for j, emotion in enumerate(['joy', 'anger', 'fear', 'sadness', 'love', 'surprise']):
        feat_idx = features.index(f'dominant_{emotion}')
        X[i, feat_idx] = 1 if j == dom_idx else 0

# Create a DataFrame
X_df = pd.DataFrame(X, columns=features)

# Generate random target labels
y = np.random.choice(genres, size=n_samples)

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_df, y)

# Save the model with feature names
# Create a custom dictionary with model and feature names
model_data = {
    'model': model,
    'feature_names': features,
    'classes': genres
}

# Save the model
joblib.dump(model_data, 'random_forest_best_model.joblib')

print("Dummy model created successfully with feature names included!")
print("This model is for testing purposes only and makes random predictions.")