from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)

try: 
    # Loading the dataset
    df_tv =  pd.read_csv('C:/Users/aryam/OneDrive/Desktop/movie-recommendation-system/netflix_titles.csv')
    # Dropping the rows with missing values
    df_tv = df_tv[['title', 'release_year', 'duration', 'listed_in', 'rating']].dropna()
    # Finding the duration of the movie in minutes
    df_tv['duration'] = df_tv['duration'].apply(lambda x: int(x.split()[0]) if 'min' in x else 0)
    # Finding the 1st genre listed
    df_tv['listed_in'] = df_tv['listed_in'].apply(lambda x: x.split(',')[0])

except Exception as e:
    print(f"We cannot load the current dataset: {e}")
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
     
#Exit if the feature processing has failed
except Exception as e:
    print(f"We cannot prepare the features for the machine learning models: {e}")
    exit()
    
# KNN (K-Nearest Neighbors) Model setup
try:
    #Initialize the KNN model with 5 neighbors
    knn = NearestNeighbors(n_neighbors=5)
    # Fit the KNN model to the scaled features
    knn.fit(scaled_features)
except Exception as e:
    #Exit the program if the model fails
    print(f"Error setting up the KNN model: {e}")
    exit()
    
# K-Means Model setup
try:
    #Initialize the K-Means model with 5 clusters
    kmeans = KMeans(n_clusters=5)
    #Fit the K-Means model to the scaled features
    kmeans.fit(scaled_features)
except Exception as e:
    print(f"Error in setting up the K-Means model: {e}")
    #Exit the program if the K-Means model setup fails
    exit()
    
    
# Singular Value Decomposition Model Setup (SVD)
try:
    # Initialize SVD with 2 components (dimensionality reduction)
    svd = TruncatedSVD(n_components=2)
    # Fit the SVD model to the scaled features
    svd.fit(scaled_features)  
except Exception as e:
    print(f"Error setting up SVD model: {e}")
    # Exit if SVD model setup fails
    exit()  
    
    
# Define route for KNN recommendations
@app.route('/recommend_knn', methods=['POST'])
def recommend_knn():
    try:
        # Get the TV Show / Movie title from the Post request
        tv_title = request.json['title']
        
        # Find the index of the movie or TV show in the dataset (case-insensitive search)
        tv_index = df_tv[df_tv['title'].str.contains(tv_title, case=False)].index[0]
        
        # Find the nearest neighbors (tv) based on the KNN model
        distances, indices = knn.neighbors([scaled_features[tv_index]])
        
    except Exception as e:
        # Handle errors and return the error message
        return jsonify({'error': str(e)}), 500
    
    

# Define route for K-Means recommendations
@app.route('/recommend_kmeans', methods=['POST'])
def recommend_kmeans():
    try:
        # Get the tv show / movie title from the POST request
        tv_title = request.json['title']
        
         # Find the index of the tv / movie in the dataset
        tv_index = df_tv[df_tv['title'].str.contains(tv_title, case=False)].index[0]
        
         # Predict the cluster for the given movie
        cluster = kmeans.predict([scaled_features[tv_index]])[0]

        # Get all movies in the same cluster
        recommended_tv = df_tv[kmeans.labels_ == cluster]['title'].tolist()

        # Return the recommended movies
        return jsonify(recommended_tv)
    
    except Exception as e:
        # Handle errors and return the error message
        return jsonify({'error': str(e)}), 500
    
    
# Define route for SVD recommendations
@app.route('/recommend_svd', methods=['POST'])
def recommend_svd():
    try:
        # Get the tv title from the POST request
        tv_title = request.json['title']
        
        # Find the index of the tv / movie in the dataset
        tv_index = df_tv[df_tv['title'].str.contains(tv_title, case=False)].index[0]

        # Apply SVD transformation on the selected movie
        movie_svd = svd.transform([scaled_features[tv_index]])

        # Find the similarity between the selected movie or show and all others
        similarities = svd.inverse_transform(movie_svd)

        # Get the top 5 most similar shows and movies
        recommendations = pd.Series(similarities.flatten(), index=df_tv.index)

        # Return the top 5 recommended movies
        return jsonify(df_tv.iloc[recommendations.argsort()[::-1].head(5)]['title'].tolist())
    except Exception as e:
        # Handle errors and return the error message
        return jsonify({'error': str(e)}), 500

# Main entry point for running the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode