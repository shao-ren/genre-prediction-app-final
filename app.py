import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
import os
from dotenv import load_dotenv
import lyricsgenius
import nltk
from transformers import pipeline

# Set page configuration
st.set_page_config(
    page_title="Music Genre Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Define genre mappings - mapping from numeric labels to genre names
# This needs to match the order of classes in your model
GENRE_MAPPINGS = {
    0: "Acoustic/Chill",
    1: "Blues/Funk", 
    2: "Country",
    3: "Electronic",
    4: "Other",
    5: "Pop",
    6: "Rock",
    7: "World"
}

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('random_forest_best_model.joblib')
        
        # Check if the model is a dictionary with the model inside
        if isinstance(model_data, dict) and 'model' in model_data:
            # Return both the model and any stored feature names
            return {
                'model': model_data['model'],
                'feature_names': model_data.get('feature_names', None)
            }
        else:
            # If it's just the model itself, return it with no feature names
            return {
                'model': model_data,
                'feature_names': None
            }
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'random_forest_best_model.joblib' is in the app directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_emotion_model():
    try:
        return pipeline("text-classification", 
                       model="j-hartmann/emotion-english-distilroberta-base", 
                       top_k=None)
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        return None

# Load models
model_data = load_model()
emotion_classifier = load_emotion_model()

# Initialize Genius API client
def init_genius():
    genius_token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not genius_token:
        st.warning("Genius API token not found. Set it as GENIUS_API_TOKEN in .env file.")
        return None
    return lyricsgenius.Genius(genius_token, timeout=15)

# Function to fetch lyrics
def fetch_lyrics(song_name, artist_name=None):
    genius = init_genius()
    if not genius:
        return None
    
    try:
        if artist_name:
            song = genius.search_song(song_name, artist_name)
        else:
            song = genius.search_song(song_name)
        
        if song:
            return song.lyrics
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching lyrics: {e}")
        return None

# Function to clean lyrics
def clean_lyrics(lyrics):
    if not lyrics:
        return ""
    
    # Remove section headers like [Verse], [Chorus], etc.
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
    # Remove Genius-specific footer
    lyrics = re.sub(r'\d+Embed$', '', lyrics)
    lyrics = re.sub(r'Embed$', '', lyrics)
    
    # Remove extra whitespace
    lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
    lyrics = lyrics.strip()
    
    return lyrics

# Function to chunk text for emotion analysis (to handle long lyrics)
def chunk_text(text, max_length=500):
    """Split text into chunks of max_length characters, trying to break at sentence boundaries."""
    if len(text) <= max_length:
        return [text]
    
    # Try to split at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is longer than max_length, split it
        if len(sentence) > max_length:
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_length:
                    temp_chunk += (" " + word if temp_chunk else word)
                else:
                    chunks.append(temp_chunk)
                    temp_chunk = word
            if temp_chunk:
                chunks.append(temp_chunk)
        # Otherwise, add sentences to chunks while respecting max_length
        elif len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Function to map emotions from classifier to our required format
def map_emotions_from_classifier(emotion_results_list):
    # Initialize emotion scores
    emotions = {
        'joy': 0.0,
        'anger': 0.0,
        'fear': 0.0,
        'sadness': 0.0,
        'love': 0.0,
        'surprise': 0.0
    }
    
    # Count how many chunks we're averaging over
    chunk_count = len(emotion_results_list)
    
    # Process each chunk's results
    for emotion_results in emotion_results_list:
        # Map emotions from the classifier output
        for item in emotion_results[0]:
            label = item['label']
            score = item['score']
            
            if label in emotions:
                emotions[label] += score / chunk_count  # Average the scores
            elif label == 'neutral':
                # Distribute neutral score among positive emotions
                emotions['joy'] += (score * 0.3) / chunk_count
                emotions['love'] += (score * 0.3) / chunk_count
            elif label == 'disgust':
                # Map disgust to anger and fear
                emotions['anger'] += (score * 0.5) / chunk_count
                emotions['fear'] += (score * 0.5) / chunk_count
    
    # If love is missing (some models don't have it), estimate it
    if emotions['love'] == 0:
        emotions['love'] = emotions['joy'] * 0.5 * (1 - (emotions['anger'] + emotions['fear'] + emotions['sadness']) / 3)
    
    # If surprise is missing, estimate it
    if emotions['surprise'] == 0:
        emotions['surprise'] = sum(emotions.values()) / len(emotions)
    
    return emotions

# Function to analyze lyrics and extract emotion features
def analyze_lyrics(lyrics):
    if not lyrics or len(lyrics.strip()) < 10:
        st.warning("Lyrics are too short or not available.")
        return None
    
    # Get emotions from transformer model
    if emotion_classifier:
        try:
            # Split lyrics into manageable chunks to avoid tensor size mismatch
            lyrics_chunks = chunk_text(lyrics)
            
            # Process each chunk separately
            emotion_results_list = []
            for chunk in lyrics_chunks:
                chunk_result = emotion_classifier(chunk)
                emotion_results_list.append(chunk_result)
            
            # Combine emotions from all chunks
            emotions = map_emotions_from_classifier(emotion_results_list)
            
        except Exception as e:
            st.error(f"Error analyzing emotions: {e}")
            # Provide fallback emotion estimation (average values)
            emotions = {
                'joy': 0.5,
                'anger': 0.1,
                'fear': 0.1,
                'sadness': 0.2,
                'love': 0.3,
                'surprise': 0.2
            }
    else:
        # Fallback emotion estimation if model not available
        emotions = {
            'joy': 0.5,
            'anger': 0.1,
            'fear': 0.1,
            'sadness': 0.2,
            'love': 0.3,
            'surprise': 0.2
        }
    
    # Combine all features
    features = {
        # Emotion features only
        'emotion_joy': emotions['joy'],
        'emotion_anger': emotions['anger'],
        'emotion_fear': emotions['fear'],
        'emotion_sadness': emotions['sadness'],
        'emotion_love': emotions['love'],
        'emotion_surprise': emotions['surprise']
    }
    
    return features

# Function to estimate audio features based on emotions and lyrics
def estimate_audio_features(lyrics_features, song_name, artist_name):
    if not lyrics_features:
        return {
            'energy': 0.5,
            'mode': 1,
            'key': 0,
            'valence': 0.5,
            'tempo': 120.0,
            'word_count': 100  # Default word count
        }
    
    # Estimate energy based on emotions
    energy = (lyrics_features['emotion_joy'] * 0.3 + 
              lyrics_features['emotion_anger'] * 0.3 + 
              lyrics_features['emotion_surprise'] * 0.2)
    
    # Estimate valence (positivity) based on emotions
    valence = (lyrics_features['emotion_joy'] * 0.4 + 
               lyrics_features['emotion_love'] * 0.3 - 
               lyrics_features['emotion_sadness'] * 0.2)
    
    # Ensure values are between 0 and 1
    energy = max(0.0, min(1.0, energy))
    valence = max(0.0, min(1.0, valence))
    
    # Estimate mode (major or minor) based on emotions
    # Higher valence usually indicates major mode (1), lower valence indicates minor mode (0)
    mode = 1 if valence > 0.5 else 0
    
    # Estimate tempo using a mapping from emotions
    # Joy and anger often correlate with higher tempos, sadness with lower tempos
    base_tempo = 120.0  # Average tempo
    tempo_modifier = ((lyrics_features['emotion_joy'] * 40) + 
                      (lyrics_features['emotion_anger'] * 30) - 
                      (lyrics_features['emotion_sadness'] * 30))
    tempo = base_tempo + tempo_modifier
    
    # Constrain tempo to realistic BPM range
    tempo = max(60, min(200, tempo))
    
    # Key is hard to estimate from lyrics alone, so we'll use a placeholder
    # In a real application, you would get this from an audio analysis API
    key = 0  # C key as default
    
    return {
        'energy': energy,
        'mode': mode,
        'key': key,
        'valence': valence,
        'tempo': tempo,
        'word_count': 100  # Default word count
    }

# Function to apply feature engineering as was done during model training
def apply_feature_engineering(features):
    """Apply the same feature engineering steps that were used during model training"""
    engineered_features = features.copy()
    
    # 1. Create interaction features
    engineered_features['energy_tempo'] = features['energy'] * features['tempo']
    engineered_features['valence_energy'] = features['valence'] * features['energy']
    
    # 2. Create emotion ratios for contrasting emotions
    engineered_features['joy_sadness_ratio'] = features['emotion_joy'] / (features['emotion_sadness'] + 0.001)
    engineered_features['love_anger_ratio'] = features['emotion_love'] / (features['emotion_anger'] + 0.001)
    
    # 3. Create aggregate emotional features
    engineered_features['positive_emotions'] = features['emotion_joy'] + features['emotion_love'] + features['emotion_surprise']
    engineered_features['negative_emotions'] = features['emotion_anger'] + features['emotion_fear'] + features['emotion_sadness']
    engineered_features['emotion_contrast'] = engineered_features['positive_emotions'] - engineered_features['negative_emotions']
    
    # 4. Add polynomial features for important audio features
    engineered_features['energy_squared'] = features['energy'] ** 2
    engineered_features['valence_squared'] = features['valence'] ** 2
    
    # 5. Add derived text features (using default word count)
    if 'word_count' not in features:
        engineered_features['word_count'] = 100  # Default word count
    
    return engineered_features

# Function to check number of features in the model
def get_model_expected_features():
    if not model_data or not model_data.get('model'):
        return None
    
    model = model_data.get('model')
    
    # Different ways to get number of expected features
    if hasattr(model, 'n_features_in_'):
        return model.n_features_in_
    elif hasattr(model, 'feature_importances_'):
        return len(model.feature_importances_)
    else:
        return None

# Function to map numeric genre label to string genre name
def map_genre_label(label):
    """Convert numeric genre label to human-readable genre name"""
    if isinstance(label, (int, np.integer)):
        return GENRE_MAPPINGS.get(int(label), f"Unknown Genre ({label})")
    
    # If it's already a string, return it as is
    return label

# Define expected features after engineering
ENGINEERED_FEATURES = [
    # Base features
    'energy', 'mode', 'key', 'valence', 'tempo', 'word_count',
    'emotion_joy', 'emotion_anger', 'emotion_fear', 'emotion_sadness', 'emotion_love', 'emotion_surprise',
    # Engineered features
    'energy_tempo', 'valence_energy',
    'joy_sadness_ratio', 'love_anger_ratio',
    'positive_emotions', 'negative_emotions', 'emotion_contrast',
    'energy_squared', 'valence_squared'
]

# Function to predict genre
def predict_genre(features):
    if not model_data:
        st.error("Model not loaded. Cannot make predictions.")
        return None, None
    
    try:
        # Get the actual model from the model_data
        model = model_data.get('model')
        if not model:
            st.error("Invalid model loaded. Model object not found.")
            return None, None
        
        # Apply feature engineering to match the training process
        engineered_features = apply_feature_engineering(features)
        
        # Add word_count if it wasn't present
        if 'word_count' not in engineered_features:
            engineered_features['word_count'] = 100  # Default value
        
        # Convert features to DataFrame
        df = pd.DataFrame([engineered_features])
        
        # Get expected feature count
        expected_features = get_model_expected_features()
        if expected_features and df.shape[1] != expected_features:
            st.warning(f"Feature count mismatch: Model expects {expected_features} features, but we have {df.shape[1]}")
            st.write(f"Current features: {df.columns.tolist()}")
        
        # Make prediction
        label = model.predict(df)[0]
        
        # Map numerical label to genre name
        genre = map_genre_label(label)
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(df)[0]
                predicted_idx = label if isinstance(label, (int, np.integer)) else list(model.classes_).index(label)
                confidence = probabilities[predicted_idx]
            except Exception as e:
                st.warning(f"Could not calculate confidence: {e}")
        
        return genre, confidence
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# UI Components for the Streamlit App
def main():
    st.title("🎵 Music Genre Prediction App")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["Manual Feature Input", "Song Lookup"],
        index=0
    )
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info(
        "This app predicts music genres based on audio features and lyrical "
        "emotions. It uses a machine learning model trained on thousands of songs. "
        "\n\nYou can either manually adjust the features using sliders or "
        "enter a song name to automatically analyze its lyrics."
    )
    
    # Display model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.title("Model Information")
    if model_data and model_data.get('model'):
        model = model_data.get('model')
        st.sidebar.success("✅ Model loaded successfully")
        
        # Display model details
        if hasattr(model, 'n_estimators'):
            st.sidebar.info(f"Model type: Random Forest with {model.n_estimators} trees")
        else:
            st.sidebar.info(f"Model type: {type(model).__name__}")
        
        # Display expected features
        n_features = get_model_expected_features()
        if n_features:
            st.sidebar.info(f"Expected features: {n_features}")
        
        # Display genre mappings
        st.sidebar.subheader("Genre Mappings")
        for label, genre in GENRE_MAPPINGS.items():
            st.sidebar.write(f"{label}: {genre}")
            
    else:
        st.sidebar.error("❌ No model loaded")
    
    # Display different app modes
    if app_mode == "Manual Feature Input":
        st.header("Manual Feature Input Mode")
        st.write("Adjust the sliders to set audio and emotion features, then click 'Predict Genre'.")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Audio features - first column
        with col1:
            st.subheader("Audio Features")
            energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01, help="Energy represents intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.")
            mode = st.radio("Mode", [0, 1], index=1, help="Mode indicates the modality (major or minor) of a track. Major is 1, minor is 0.")
            
            # Updated key selection with proper pitch class notation
            key_options = {
                -1: "No key detected",
                0: "C",
                1: "C♯/D♭",
                2: "D",
                3: "D♯/E♭",
                4: "E",
                5: "F",
                6: "F♯/G♭",
                7: "G",
                8: "G♯/A♭",
                9: "A",
                10: "A♯/B♭",
                11: "B"
            }
            key_selection = st.selectbox(
            "Key", 
            options=list(key_options.keys()), 
            format_func=lambda x: key_options[x],
            index=1,  # Default to C (0)
            help="The key the track is in. Integers map to pitches using standard Pitch Class notation."
            )
            key = key_selection  # Store the numeric value
            # key = st.selectbox("Key", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], index=0, help="The key the track is in. Integers map to pitches using standard Pitch Class notation.")
            valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01, help="Valence describes the musical positiveness conveyed by a track. High valence sounds more positive.")
            tempo = st.slider("Tempo (BPM)", 60.0, 200.0, 120.0, 1.0, help="The overall estimated tempo of a track in beats per minute (BPM).")
            word_count = st.slider("Word Count", 0, 500, 100, 10, help="Approximate number of words in the lyrics.")
        
        # Emotion features - second column
        with col2:
            st.subheader("Emotion Features")
            emotion_joy = st.slider("Joy", 0.0, 1.0, 0.5, 0.01, help="The level of joy expressed in the lyrics.")
            emotion_anger = st.slider("Anger", 0.0, 1.0, 0.1, 0.01, help="The level of anger expressed in the lyrics.")
            emotion_fear = st.slider("Fear", 0.0, 1.0, 0.1, 0.01, help="The level of fear expressed in the lyrics.")
            emotion_sadness = st.slider("Sadness", 0.0, 1.0, 0.2, 0.01, help="The level of sadness expressed in the lyrics.")
            emotion_love = st.slider("Love", 0.0, 1.0, 0.3, 0.01, help="The level of love/affection expressed in the lyrics.")
            emotion_surprise = st.slider("Surprise", 0.0, 1.0, 0.2, 0.01, help="The level of surprise expressed in the lyrics.")
        
        # Combine features
        features = {
            # Audio features
            'energy': energy,
            'mode': mode,
            'key': key,
            'valence': valence,
            'tempo': tempo,
            'word_count': word_count,
            # Emotion features
            'emotion_joy': emotion_joy,
            'emotion_anger': emotion_anger,
            'emotion_fear': emotion_fear,
            'emotion_sadness': emotion_sadness,
            'emotion_love': emotion_love,
            'emotion_surprise': emotion_surprise
        }
        
        # Display debug info 
        if st.checkbox("Show Debug Information"):
            # Apply feature engineering for visualization
            engineered = apply_feature_engineering(features)
            st.write("Engineered Features:")
            for name, value in engineered.items():
                if name not in features:  # Only show the new engineered features
                    st.write(f"{name}: {value:.4f}")
        
        # Predict button
        if st.button("Predict Genre", type="primary", use_container_width=True):
            with st.spinner("Predicting genre..."):
                genre, confidence = predict_genre(features)
            
            # Display prediction
            if genre:
                st.success(f"Predicted Genre: **{genre}**")
                
                if confidence:
                    # Create a confidence meter
                    st.write(f"Confidence: {confidence:.2f}")
                    st.progress(float(confidence))
                
                # Display feature importances or explanations
                st.subheader("Feature Values Used for Prediction")
                
                # Create two columns for the feature visualization
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Audio features visualization
                    audio_chart_data = pd.DataFrame({
                        'Feature': ['Energy', 'Valence', 'Tempo/200'],
                        'Value': [energy, valence, tempo/200]  # Normalize tempo
                    })
                    st.write("Audio Features")
                    st.bar_chart(audio_chart_data.set_index('Feature'))
                
                with viz_col2:
                    # Emotion features visualization
                    emotion_chart_data = pd.DataFrame({
                        'Emotion': ['Joy', 'Love', 'Surprise', 'Anger', 'Fear', 'Sadness'],
                        'Value': [emotion_joy, emotion_love, emotion_surprise, 
                                 emotion_anger, emotion_fear, emotion_sadness]
                    })
                    st.write("Emotion Features")
                    st.bar_chart(emotion_chart_data.set_index('Emotion'))
            else:
                st.error("Failed to predict genre. Please try different feature values.")
                
    else:  # Song Lookup mode
        st.header("Song Lookup Mode")
        st.write("Enter a song name and optionally an artist name to predict its genre.")
        
        # Input fields for song info
        song_name = st.text_input("Song Name", placeholder="Enter song name...")
        artist_name = st.text_input("Artist Name (Optional)", placeholder="Enter artist name...")
        
        # Search button
        if st.button("Search and Predict", type="primary", use_container_width=True):
            if not song_name:
                st.warning("Please enter a song name.")
            else:
                with st.spinner("Fetching lyrics and analyzing..."):
                    # Fetch lyrics
                    lyrics = fetch_lyrics(song_name, artist_name)
                    
                    if lyrics:
                        # Clean and display lyrics
                        clean_lyric_text = clean_lyrics(lyrics)
                        
                        # Calculate word count
                        word_count = len(clean_lyric_text.split())
                        
                        # Create tabs for results
                        tab1, tab2, tab3 = st.tabs(["Prediction", "Features", "Lyrics"])
                        
                        with tab1:
                            # Analyze lyrics
                            with st.spinner("Analyzing lyrics emotions..."):
                                lyrics_features = analyze_lyrics(clean_lyric_text)
                            
                            if lyrics_features:
                                # Estimate audio features
                                audio_features = estimate_audio_features(lyrics_features, song_name, artist_name)
                                
                                # Add word count to audio features
                                audio_features['word_count'] = word_count
                                
                                # Combine features
                                features = {**audio_features, **lyrics_features}
                                
                                # Make prediction
                                genre, confidence = predict_genre(features)
                                
                                if genre:
                                    st.success(f"Predicted Genre: **{genre}**")
                                    
                                    if confidence:
                                        # Create a confidence meter
                                        st.write(f"Confidence: {confidence:.2f}")
                                        st.progress(float(confidence))
                                    
                                    # Display similar genres
                                    st.subheader("Similar Genres")
                                    similar_genres = {
                                        'acoustic': ['folk', 'indie', 'chill'],
                                        'alternative': ['rock', 'indie', 'emo'],
                                        'blues': ['jazz', 'soul', 'funk'],
                                        'children': ['disney', 'happy', 'acoustic'],
                                        'chill': ['acoustic', 'lofi', 'ambient'],
                                        'club': ['dance', 'house', 'edm'],
                                        'country': ['folk', 'acoustic', 'americana'],
                                        'dance': ['edm', 'club', 'house'],
                                        'disco': ['funk', 'dance', 'pop'],
                                        'disney': ['children', 'happy', 'pop'],
                                        'edm': ['dance', 'club', 'electro'],
                                        'electro': ['edm', 'dance', 'house'],
                                        'emo': ['alternative', 'rock', 'punk'],
                                        'funk': ['disco', 'soul', 'groove'],
                                        'groove': ['funk', 'soul', 'disco'],
                                        'happy': ['pop', 'disney', 'children'],
                                        'house': ['edm', 'dance', 'club'],
                                        'jazz': ['blues', 'soul', 'funk'],
                                        'pop': ['dance', 'happy', 'disco'],
                                        'afrobeat': ['funk', 'groove', 'dance'],
                                        # Add mappings for grouped genres
                                        'Acoustic/Chill': ['acoustic', 'chill', 'folk', 'indie'],
                                        'Blues/Funk': ['blues', 'funk', 'groove', 'jazz', 'soul'],
                                        'Country': ['country', 'folk', 'americana'],
                                        'Electronic': ['edm', 'electro', 'house', 'dance', 'club', 'disco'],
                                        'Rock': ['rock', 'alternative', 'emo', 'punk'],
                                        'Pop': ['pop', 'disco', 'dance'],
                                        'World': ['afrobeat', 'latin', 'world'],
                                        'Other': ['children', 'disney', 'happy']
                                    }
                                    
                                    if genre in similar_genres:
                                        st.write(f"If you like {genre}, you might also enjoy: {', '.join(similar_genres[genre])}")
                                else:
                                    st.error("Failed to predict genre. Try a different song.")
                            else:
                                st.error("Failed to analyze lyrics sentiment.")
                        
                        with tab2:
                            # Display extracted features
                            st.subheader("Extracted Features")
                            
                            if lyrics_features and audio_features:
                                # Combine features
                                combined_features = {**audio_features, **lyrics_features}
                                
                                # Apply feature engineering
                                engineered_features = apply_feature_engineering(combined_features)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("Audio Features (Estimated)")
                                    audio_df = pd.DataFrame({
                                        'Feature': list(audio_features.keys()),
                                        'Value': list(audio_features.values())
                                    })
                                    st.dataframe(audio_df, use_container_width=True)
                                
                                with col2:
                                    st.write("Emotion Features (From Lyrics)")
                                    emotion_df = pd.DataFrame({
                                        'Emotion': ['Joy', 'Anger', 'Fear', 'Sadness', 'Love', 'Surprise'],
                                        'Value': [
                                            lyrics_features['emotion_joy'],
                                            lyrics_features['emotion_anger'],
                                            lyrics_features['emotion_fear'],
                                            lyrics_features['emotion_sadness'],
                                            lyrics_features['emotion_love'],
                                            lyrics_features['emotion_surprise']
                                        ]
                                    })
                                    st.dataframe(emotion_df, use_container_width=True)
                                
                                # Display engineered features
                                st.write("Engineered Features")
                                engineered_df = pd.DataFrame({
                                    'Feature': [
                                        'word_count',
                                        'energy_tempo', 'valence_energy', 
                                        'joy_sadness_ratio', 'love_anger_ratio',
                                        'positive_emotions', 'negative_emotions', 'emotion_contrast',
                                        'energy_squared', 'valence_squared'
                                    ],
                                    'Value': [
                                        engineered_features['word_count'],
                                        engineered_features['energy_tempo'],
                                        engineered_features['valence_energy'],
                                        engineered_features['joy_sadness_ratio'],
                                        engineered_features['love_anger_ratio'],
                                        engineered_features['positive_emotions'],
                                        engineered_features['negative_emotions'],
                                        engineered_features['emotion_contrast'],
                                        engineered_features['energy_squared'],
                                        engineered_features['valence_squared']
                                    ]
                                })
                                st.dataframe(engineered_df, use_container_width=True)
                                
                                # Visualization of emotions
                                st.subheader("Emotion Distribution")
                                emotion_chart_data = pd.DataFrame({
                                    'Emotion': ['Joy', 'Love', 'Surprise', 'Anger', 'Fear', 'Sadness'],
                                    'Value': [
                                        lyrics_features['emotion_joy'],
                                        lyrics_features['emotion_love'],
                                        lyrics_features['emotion_surprise'],
                                        lyrics_features['emotion_anger'],
                                        lyrics_features['emotion_fear'],
                                        lyrics_features['emotion_sadness']
                                    ]
                                })
                                st.bar_chart(emotion_chart_data.set_index('Emotion'))
                                
                                # Visualization of engineered emotion features
                                st.subheader("Derived Emotional Features")
                                derived_chart_data = pd.DataFrame({
                                    'Feature': ['Positive Emotions', 'Negative Emotions', 'Emotion Contrast'],
                                    'Value': [
                                        engineered_features['positive_emotions'],
                                        engineered_features['negative_emotions'],
                                        engineered_features['emotion_contrast']
                                    ]
                                })
                                st.bar_chart(derived_chart_data.set_index('Feature'))
                            else:
                                st.write("No features extracted.")
                        
                        with tab3:
                            # Display lyrics
                            st.subheader("Song Lyrics")
                            if clean_lyric_text:
                                st.write(f"Word count: {word_count} words")
                                st.text_area("", clean_lyric_text, height=400)
                            else:
                                st.write("No lyrics found.")
                    else:
                        st.error("Failed to retrieve lyrics. Check the song name and artist, or try a different song.")

# Run the app
if __name__ == "__main__":
    main()