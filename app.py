from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

try: 
    # Loading the dataset
    df_tv = pd.read_csv('netflix_titles.csv')
    # Dropping the rows with missing values
    df_tv = df_tv[['title', 'release_year', 'duration', 'listed_in', 'rating']].dropna()
    # Finding the duration of the movie in minutes
    df_tv['duration'] = df_tv['duration'].apply(lambda x: int(x.split()[0]) if 'min' in x else 0)
    # Finding the 1st genre listed
    df_tv['listed_in'] = df_tv['listed_in'].apply(lambda x: x.split(',')[0])

except Exception as e:
    print("We cannot load the current dataset")
    exit()

# Preparing the features for the machine learning models
try:
    # Using the release year and duration as numerical features
    features = df_tv[['release_year', 'duration']].copy()

    # Simplifying the rating if the movie is not rated, we assign 0
    features['rating'] = df_tv['rating'].apply(lambda x: 0 if x == 'NR' else 1)

    # Standardize the features values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

except Exception as e:
    print("We cannot prepare the features for the machine learning models")
    exit()

# KNN (K-Nearest Neighbors) Model setup
try:
    # Initialize the KNN model with 5 neighbors
    knn = NearestNeighbors(n_neighbors=5)
    # Fit the KNN model to the scaled features
    knn.fit(scaled_features)
except Exception as e:
    print("Error setting up the KNN model")
    exit()

# K-Means Model setup
try:
    # Initialize the K-Means model with 5 clusters
    kmeans = KMeans(n_clusters=5)
    # Fit the K-Means model to the scaled features
    kmeans.fit(scaled_features)
except Exception as e:
    print("Error in setting up the K-Means model")
    exit()

# Singular Value Decomposition Model Setup (SVD)
try:
    # Initialize SVD with 2 components (dimensionality reduction)
    svd = TruncatedSVD(n_components=2)
    # Fit the SVD model to the scaled features
    svd.fit(scaled_features)  
except Exception as e:
    print("Error setting up SVD model")
    exit()  

# Define home route
@app.route('/')
def home():
    return "Welcome to the Movie Recommendation System API! Use /recommend_knn, /recommend_kmeans, or /recommend_svd to get recommendations."

# Define route for KNN recommendations
@app.route('/recommend_knn', methods=['POST'])
def recommend_knn():
    try:
        # Get the TV Show / Movie title from the POST request
        tv_title = request.json['title']

        # Find the index of the movie or TV show in the dataset (case-insensitive search)
        tv_index = df_tv[df_tv['title'].str.contains(tv_title, case=False)].index[0]

        # Start with 5 neighbors and gradually increase if necessary
        n_neighbors = 6  # Get 6 neighbors initially to account for possible exclusion of the input movie

        while True:
            # Find the nearest neighbors (tv) based on the KNN model
            distances, indices = knn.kneighbors([scaled_features[tv_index]], n_neighbors=n_neighbors)

            # Get the recommended titles from the indices, excluding the input movie itself
            recommended_titles = df_tv.iloc[indices[0]]['title'].tolist()
            recommended_titles = [title for title in recommended_titles if title.lower() != tv_title.lower()]

            # If we have at least 5 recommendations, return them
            if len(recommended_titles) >= 5:
                return jsonify(recommended_titles[:5])
            
            # If not, increase the number of neighbors and try again
            n_neighbors += 1

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define route for KMeans recommedations
@app.route('/recommend_kmeans', methods=['POST'])
def recommend_kmeans():
    try:
        # Get the TV show / movie title from the POST request
        tv_title = request.json['title']

        # Find the index of the TV show / movie in the dataset
        tv_index = df_tv[df_tv['title'].str.contains(tv_title, case=False)].index[0]

        # Predict the cluster for the given movie
        cluster = kmeans.predict([scaled_features[tv_index]])[0]

        # Get all movies in the same cluster
        recommended_tv = df_tv[kmeans.labels_ == cluster]['title'].tolist()

        # Return only the top 5 recommended movies
        recommended_tv_top5 = recommended_tv[:5]

        # Return the top 5 recommendations as a JSON response
        return jsonify(recommended_tv_top5)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Define route for SVD recommendations
@app.route('/recommend_svd', methods=['POST'])
def recommend_svd():
    try:
        # Get the TV title from the POST request
        tv_title = request.json['title']

        # Find the index of the TV show / movie in the dataset
        tv_index = df_tv[df_tv['title'].str.contains(tv_title, case=False)].index[0]

        # Apply SVD transformation on the selected movie
        movie_svd = svd.transform([scaled_features[tv_index]])

        # Transform the entire dataset using SVD (reduced dimensions)
        all_movies_svd = svd.transform(scaled_features)

        # Compute cosine similarity between the selected movie and all other movies
        similarities = cosine_similarity(movie_svd, all_movies_svd)[0]

        # Get the indices of the most similar movies, sorted by similarity score
        similar_indices = similarities.argsort()[::-1]

        # Get the top 5 most similar shows and movies (excluding the input movie)
        top_5_indices = [i for i in similar_indices if i != tv_index][:5]

        # Return the top 5 recommended movies as a JSON response
        recommended_titles = df_tv.iloc[top_5_indices]['title'].tolist()
        return jsonify(recommended_titles)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point for running the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
