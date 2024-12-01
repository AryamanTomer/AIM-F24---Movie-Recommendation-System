from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Loading the dataset
df_tv = pd.read_csv('netflix_titles.csv')

# Dropping the rows with missing values
df_tv = df_tv[['title', 'release_year', 'duration', 'listed_in', 'cast']].dropna()

# Extract the 1st genre listed in 'listed_in'
df_tv['listed_in'] = df_tv['listed_in'].apply(lambda x: x.split(',')[0])

# Preprocessing the cast by splitting on commas (assuming cast is a comma-separated string)
df_tv['cast'] = df_tv['cast'].apply(lambda x: x.split(','))

# Prepare features (listed_in and cast)
df_tv['listed_in'] = df_tv['listed_in'].astype('category')
df_tv['cast'] = df_tv['cast'].apply(lambda x: [actor.strip().lower() for actor in x])  # Clean up actors' names

# Define the route to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get the TV Show / Movie title from the POST request
        tv_title = request.json['title']

        # Find the index of the TV show/movie in the dataset
        tv_index = df_tv[df_tv['title'].str.contains(tv_title, case=False)].index[0]

        # Step 1: Filter by 'listed_in' (primary filter)
        genre = df_tv.iloc[tv_index]['listed_in']
        genre_filtered = df_tv[df_tv['listed_in'] == genre]

        # Step 2: Further filter by 'cast' (secondary filter)
        target_cast = df_tv.iloc[tv_index]['cast']
        recommendations = []

        for idx, row in genre_filtered.iterrows():
            # Calculate the intersection of cast members (shared actors)
            common_cast = set(target_cast) & set(row['cast'])
            if common_cast:
                recommendations.append((row['title'], len(common_cast)))  # Add the number of shared cast members as a tie-breaker

        # Sort recommendations by the number of shared cast members (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Get top 5 recommendations based on shared cast members
        top_recommendations = [item[0] for item in recommendations[:5]]

        return jsonify(top_recommendations)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
