from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset once
movies = pd.read_csv('netflix_titles.csv')

# Filter to include only Movies (optional; adjust as needed)
movies = movies[movies['type'] == 'Movie'].reset_index()

# Preprocess: Drop unused columns
movies = movies.drop(
    ['index', 'show_id', 'type', 'date_added', 'release_year', 'duration', 'description'], axis=1
)

# Preprocess data to create binary matrices for actors, directors, countries, genres
def preprocess_data(df):
    def create_binary_matrix(column):
        unique_items = set()
        for entry in df[column].dropna():
            items = re.split(r', \s*', entry)
            unique_items.update(items)

        unique_items = sorted(unique_items)
        matrix = []
        for entry in df[column]:
            row = [
                1.0 if item in (entry if pd.notna(entry) else '') else 0.0 for item in unique_items
            ]
            matrix.append(row)

        return pd.DataFrame(matrix, columns=unique_items)

    binary_actors = create_binary_matrix('cast')
    binary_directors = create_binary_matrix('director')
    binary_countries = create_binary_matrix('country')
    binary_genres = create_binary_matrix('listed_in')

    return pd.concat([binary_actors, binary_directors, binary_countries, binary_genres], axis=1)

# Preprocess data once at the start
binary_movies = preprocess_data(movies)

# Fit the Nearest Neighbors model
# n_neighbors specifies how many recommendations you want to return
neighbors_model = NearestNeighbors(n_neighbors=6, metric='cosine')
neighbors_model.fit(binary_movies)

# Recommendation function using Nearest Neighbors
def get_recommendations(title, binary_data, df, model):
    if title not in df['title'].values:
        return None

    idx = df[df['title'] == title].index.item()
    # Use DataFrame to maintain feature names
    title_vector = binary_data.iloc[idx].to_numpy().reshape(1, -1)

    # Get nearest neighbors (excluding the movie itself)
    distances, indices = model.kneighbors(title_vector)

    recommendations = []
    for i in range(1, len(indices[0])):  # Skip the first movie (itself)
        neighbor_idx = indices[0][i]
        recommendations.append(df.iloc[neighbor_idx]['title'])

    return recommendations

# Flask routes
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Netflix Movie Recommendation API!"})

@app.route('/recommend', methods=['POST'])
def recommend():
    # Extract title from the JSON request body
    data = request.get_json()
    title = data.get('title', '')
    
    if not title:
        return jsonify({"error": "Please provide a movie title."}), 400

    # Get recommendations
    recommendations = get_recommendations(title, binary_movies, movies, neighbors_model)
    if recommendations is None:
        return jsonify({"error": "Movie not found in the dataset."}), 404

    return jsonify({"title": title, "recommendations": recommendations})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
