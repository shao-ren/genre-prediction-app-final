# Music Genre Prediction App

This Streamlit application predicts music genres based on audio features and lyrical content. It uses a pre-trained Random Forest model and can work in two modes:

1. **Manual Feature Input Mode**: Allows users to adjust sliders for each audio and emotion feature, then predicts the genre based on these inputs.
2. **Song Lookup Mode**: Users enter a song name (and optionally an artist name), and the app fetches the lyrics, analyzes them for emotional content, estimates audio features, and predicts the genre.

## Features

- Genre prediction using a pre-trained Random Forest model
- Lyrics fetching via the Genius API
- Emotion analysis using a pre-trained transformer model
- Visualization of extracted features and emotions
- Audio feature estimation based on lyrical content
- Interactive user interface with sliders and input fields

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/music-genre-prediction.git
   cd music-genre-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Genius API token:
   - Create a Genius API account at https://genius.com/api-clients
   - Create a `.env` file in the project directory
   - Add your Genius API token to the `.env` file:
     ```
     GENIUS_API_TOKEN=your_token_here
     ```

4. Ensure you have the model file:
   - Place your `random_forest_best_model.joblib` in the project directory
   - If you don't have this file, the app will show an error message

## Running the App

Run the Streamlit app with:

```
streamlit run app.py
```

The app should open in your default web browser. If it doesn't, navigate to the URL displayed in your terminal (usually http://localhost:8501).

## Usage

### Manual Feature Input Mode

1. Adjust the sliders for audio features (energy, mode, key, valence, tempo)
2. Adjust the sliders for emotion features (joy, anger, fear, sadness, love, surprise)
3. Adjust the sliders for sentiment features (VADER and TextBlob scores)
4. Click the "Predict Genre" button to get a prediction

### Song Lookup Mode

1. Enter a song name
2. Optionally enter an artist name for more accurate results
3. Click the "Search and Predict" button
4. View the predicted genre and explore the extracted features and lyrics

## How It Works

1. **Manual Mode**: The app collects all feature values from the sliders and passes them directly to the model for prediction.

2. **Song Lookup Mode**:
   - The app uses the Genius API to fetch lyrics for the specified song
   - It analyzes the lyrics using sentiment analysis and emotion detection
   - It estimates audio features based on the extracted emotions
   - All features are combined and passed to the model for genre prediction

## Model Information

The app uses a pre-trained Random Forest classifier that was trained on a dataset of songs with audio features and emotions extracted from lyrics. The model expects the following features:

- Audio features: energy, mode, key, valence, tempo
- Emotion features: joy, anger, fear, sadness, love, surprise
- Sentiment features: VADER (compound, positive, negative, neutral) and TextBlob (polarity, subjectivity)
- Engineered features: various ratios and combinations of the above features

## Limitations

- The audio feature estimation in Song Lookup Mode is approximate and based on emotional content of lyrics
- Lyrics may not be available for all songs
- The genre predictions are limited to the genres the model was trained on
- The emotion analysis is based on pre-trained models and may not always accurately capture the emotions in the lyrics

## Future Improvements

- Integration with music streaming APIs for more accurate audio feature extraction
- Support for more languages
- Fine-tuning of emotion detection models for music-specific sentiment
- Expanded genre coverage
- User feedback collection for model improvement

## License

This project is licensed under the MIT License - see the LICENSE file for details.