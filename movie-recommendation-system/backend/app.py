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

# Preprocess: Drop unused columns (keep 'description' and 'release_year')
movies = movies.drop(
    ['index', 'show_id', 'type', 'date_added', 'duration'], axis=1
)


# Preprocess data to create binary matrices for countries and genres, and TF-IDF for description
def preprocess_features(df, weights=None):
    # Genres
    genres = df['listed_in'].fillna('').apply(lambda x: [g.strip() for g in x.split(',')])
    mlb_genre = MultiLabelBinarizer()
    genre_matrix = mlb_genre.fit_transform(genres)
    # Countries
    countries = df['country'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])
    mlb_country = MultiLabelBinarizer()
    country_matrix = mlb_country.fit_transform(countries)
    # Director
    directors = df['director'].fillna('').apply(lambda x: [d.strip() for d in x.split(',')])
    mlb_director = MultiLabelBinarizer()
    director_matrix = mlb_director.fit_transform(directors)
    # Cast
    cast = df['cast'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])
    mlb_cast = MultiLabelBinarizer()
    cast_matrix = mlb_cast.fit_transform(cast)
    # Rating
    ratings = pd.get_dummies(df['rating'].fillna(''))
    # Release year (normalized)
    years = df['release_year'].fillna(df['release_year'].mode()[0])
    years_norm = ((years - years.min()) / (years.max() - years.min())).values.reshape(-1, 1)
    # Description TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    desc_matrix = tfidf.fit_transform(df['description'].fillna(''))
    # Feature weights
    if weights is None:
        weights = {
            'genre': 1.0,
            'country': 1.0,
            'director': 0.7,
            'cast': 0.7,
            'rating': 0.5,
            'year': 0.5,
            'desc': 1.0
        }
    import scipy.sparse
    combined = scipy.sparse.hstack([
        genre_matrix * weights['genre'],
        country_matrix * weights['country'],
        director_matrix * weights['director'],
        cast_matrix * weights['cast'],
        ratings.values * weights['rating'],
        years_norm * weights['year'],
        desc_matrix * weights['desc']
    ]).tocsr()
    return combined, mlb_genre, mlb_country, mlb_director, mlb_cast, tfidf


# Preprocess features once at the start
feature_matrix, mlb_genre, mlb_country, mlb_director, mlb_cast, tfidf = preprocess_features(movies)

# Build KNN model with cosine similarity
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(feature_matrix)


# Recommendation function using KNN and cosine similarity
def get_recommendations(title, df, knn_model, k=6, min_similarity=0.0, weights=None):
    # Recompute feature matrix if weights are provided
    if weights is not None:
        feature_matrix_custom, *_ = preprocess_features(df, weights)
    else:
        feature_matrix_custom = feature_matrix
    if title not in df['title'].values:
        return None, None
    idx = df[df['title'] == title].index.item()
    title_vector = feature_matrix_custom[idx]
    distances, indices = knn_model.kneighbors(title_vector, n_neighbors=k+1)  # +1 to skip itself
    recommendations = []
    similarity_scores = []
    metadata = []
    for i in range(1, len(indices[0])):  # Skip the first movie (itself)
        neighbor_idx = indices[0][i]
        score = 1 - distances[0][i]
        if score >= min_similarity:
            recommendations.append(df.iloc[neighbor_idx]['title'])
            similarity_scores.append(score)
            # Add more metadata for each recommendation
            metadata.append({
                'title': df.iloc[neighbor_idx]['title'],
                'genre': df.iloc[neighbor_idx]['listed_in'],
                'country': df.iloc[neighbor_idx]['country'],
                'director': df.iloc[neighbor_idx]['director'],
                'cast': df.iloc[neighbor_idx]['cast'],
                'rating': df.iloc[neighbor_idx]['rating'],
                'release_year': df.iloc[neighbor_idx]['release_year'],
                'description': df.iloc[neighbor_idx]['description'],
                'similarity': score
            })
    return recommendations, similarity_scores, metadata


# Flask routes
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Netflix Movie Recommendation API!"})



@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data.get('title', '')
    k = int(data.get('k', 5))
    min_similarity = float(data.get('min_similarity', 0.0))
    weights = data.get('weights', None)
    if not title:
        return jsonify({"error": "Please provide a movie title."}), 400
    recommendations, similarity_scores, metadata = get_recommendations(title, movies, knn, k=k, min_similarity=min_similarity, weights=weights)
    if recommendations is None:
        return jsonify({"error": "Movie not found in the dataset."}), 404
    # Convert numpy types to native Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return obj
    recommendations = [convert(r) for r in recommendations]
    similarity_scores = [convert(s) for s in similarity_scores]
    metadata_clean = []
    for m in metadata:
        m_clean = {k: convert(v) for k, v in m.items()}
        metadata_clean.append(m_clean)
    return jsonify({
        "title": title,
        "recommendations": recommendations,
        "similarity_scores": similarity_scores,
        "metadata": metadata_clean
    })

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
