import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import librosa

# Load models
lyrics_model = joblib.load('model/lyrics_model.pkl')  # Placeholder
kmeans_model = joblib.load('model/clustering.pkl')
audio_model = None  # For demo, you can later use a real model

# Load dataset
songs_df = pd.read_csv("data/data.csv")

# Sentiment from free-text input
def analyze_text_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range: -1 to 1

# Audio feature extractor (placeholder)
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    features = {
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
        'rmse': np.mean(librosa.feature.rms(y=y)),
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y))
    }
    return features

# Fusion-based recommendation
def recommend_songs(user_mood, genre, user_text=None):
    filtered = songs_df[songs_df['genre'].str.lower() == genre.lower()]

    if user_text:
        polarity = analyze_text_sentiment(user_text)
        filtered = filtered[filtered['valence'] >= polarity]

    # Add clustering similarity
    if 'cluster' in songs_df.columns:
        target_cluster = filtered['cluster'].mode().iloc[0] if not filtered.empty else None
        if target_cluster is not None:
            filtered = filtered[filtered['cluster'] == target_cluster]

    # Sort by popularity or danceability or energy
    sorted_df = filtered.sort_values(by=['popularity', 'energy'], ascending=False)

    recommendations = []
    for _, row in sorted_df.head(5).iterrows():
        recommendations.append({
            'title': row['track_name'],
            'artist': row['artist_name'],
            'reason': f"Matched mood: {user_mood}, genre: {genre}, and sentiment analysis"
        })
    return recommendations
