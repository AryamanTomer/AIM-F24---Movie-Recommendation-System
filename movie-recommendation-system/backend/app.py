from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import faiss

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

# Preprocess data to create binary matrices for countries and genres
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

    binary_countries = create_binary_matrix('country')
    binary_genres = create_binary_matrix('listed_in')

    return pd.concat([binary_countries, binary_genres], axis=1)

# Preprocess data once at the start
binary_movies = preprocess_data(movies)

# Convert the dataframe into a numpy array
movie_vectors = binary_movies.to_numpy().astype('float32')

# Build the FAISS index
index = faiss.IndexFlatL2(movie_vectors.shape[1])  # L2 distance (Euclidean)
index.add(movie_vectors)

# Recommendation function using FAISS
def get_recommendations(title, df, index, k=6):
    if title not in df['title'].values:
        return None

    idx = df[df['title'] == title].index.item()
    title_vector = movie_vectors[idx].reshape(1, -1)

    # Get nearest neighbors (excluding the movie itself)
    distances, indices = index.search(title_vector, k)

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
    recommendations = get_recommendations(title, movies, index)
    if recommendations is None:
        return jsonify({"error": "Movie not found in the dataset."}), 404

    return jsonify({"title": title, "recommendations": recommendations})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
