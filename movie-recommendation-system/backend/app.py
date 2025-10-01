from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset once
movies = pd.read_csv('netflix_titles.csv')

# Filter to include only Movies (optional; adjust as needed)
movies = movies[movies['type'] == 'Movie'].reset_index()

# Preprocess: Drop unused columns (keep 'description')
movies = movies.drop(
    ['index', 'show_id', 'type', 'date_added', 'release_year', 'duration'], axis=1
)


# Preprocess data to create binary matrices for countries and genres, and TF-IDF for description
def preprocess_features(df):
    # Genres
    genres = df['listed_in'].fillna('').apply(lambda x: [g.strip() for g in x.split(',')])
    mlb_genre = MultiLabelBinarizer()
    genre_matrix = mlb_genre.fit_transform(genres)
    # Countries
    countries = df['country'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])
    mlb_country = MultiLabelBinarizer()
    country_matrix = mlb_country.fit_transform(countries)
    # Description TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    desc_matrix = tfidf.fit_transform(df['description'].fillna(''))
    # Combine all features
    import scipy.sparse
    combined = scipy.sparse.hstack([genre_matrix, country_matrix, desc_matrix]).tocsr()
    return combined, mlb_genre, mlb_country, tfidf

# Preprocess features once at the start
feature_matrix, mlb_genre, mlb_country, tfidf = preprocess_features(movies)

# Build KNN model with cosine similarity
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(feature_matrix)


# Recommendation function using KNN and cosine similarity
def get_recommendations(title, df, knn_model, k=6):
    if title not in df['title'].values:
        return None, None
    idx = df[df['title'] == title].index.item()
    title_vector = feature_matrix[idx]
    distances, indices = knn_model.kneighbors(title_vector, n_neighbors=k)
    recommendations = []
    similarity_scores = []
    for i in range(1, len(indices[0])):  # Skip the first movie (itself)
        neighbor_idx = indices[0][i]
        recommendations.append(df.iloc[neighbor_idx]['title'])
        similarity_scores.append(1 - distances[0][i])  # Cosine similarity: 1 - distance
    return recommendations, similarity_scores


# Flask routes
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Netflix Movie Recommendation API!"})


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data.get('title', '')
    if not title:
        return jsonify({"error": "Please provide a movie title."}), 400
    recommendations, similarity_scores = get_recommendations(title, movies, knn)
    if recommendations is None:
        return jsonify({"error": "Movie not found in the dataset."}), 404
    return jsonify({"title": title, "recommendations": recommendations, "similarity_scores": similarity_scores})

@app.route('/movies', methods=['GET'])
def get_movies():
    # Return all movies and metadata
    return movies.to_json(orient='records')

@app.route('/stats', methods=['GET'])
def get_stats():
    # Genre distribution
    genre_counts = movies['listed_in'].str.split(',').explode().str.strip().value_counts().to_dict()
    # Country distribution
    country_counts = movies['country'].dropna().str.split(',').explode().str.strip().value_counts().to_dict()
    # Correlation (genre-country)
    pairs = []
    for _, row in movies.dropna(subset=['listed_in', 'country']).iterrows():
        genres = [g.strip() for g in row['listed_in'].split(',')]
        countries = [c.strip() for c in row['country'].split(',')]
        for genre in genres:
            for country in countries:
                pairs.append({'genre': genre, 'country': country})
    corr_df = pd.DataFrame(pairs)
    corr_table = pd.crosstab(corr_df['genre'], corr_df['country'])
    correlation = corr_table.to_dict()
    return jsonify({
        "genre_counts": genre_counts,
        "country_counts": country_counts,
        "correlation": correlation
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
