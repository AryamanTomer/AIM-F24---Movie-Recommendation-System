from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process

app = Flask(__name__)

# Load dataset
df_tv = pd.read_csv('netflix_titles.csv')
df_tv = df_tv[['title', 'listed_in', 'cast']].dropna()

# Preprocess the `listed_in` column to handle multiple categories
df_tv['listed_in'] = df_tv['listed_in'].apply(lambda x: x.split(', '))

# Define route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        input_title = request.json['title']

        # Exact title match
        exact_matches = df_tv[df_tv['title'].str.lower() == input_title.lower()]
        if not exact_matches.empty:
            # Recommend titles with the same "Barbie" keyword
            keyword = "Barbie"
            barbie_related = df_tv[df_tv['title'].str.contains(keyword, case=False)]['title'].tolist()
            barbie_related = [title for title in barbie_related if title.lower() != input_title.lower()]
            return jsonify(barbie_related[:5])

        # Fuzzy title matching
        fuzzy_matches = process.extract(input_title, df_tv['title'], scorer=fuzz.partial_ratio)
        similar_titles = [match[0] for match in fuzzy_matches if match[1] > 80]

        if similar_titles:
            # Recommend titles that are closely related to the fuzzy match
            recommendations = df_tv[df_tv['title'].isin(similar_titles)]['title'].tolist()
            recommendations = [title for title in recommendations if title.lower() != input_title.lower()]
            return jsonify(recommendations[:5])

        # Fallback: Use `listed_in` similarity
        input_categories = df_tv.loc[df_tv['title'].str.contains(input_title, case=False), 'listed_in'].values
        if len(input_categories) > 0:
            input_categories = input_categories[0]
            df_tv['category_similarity'] = df_tv['listed_in'].apply(
                lambda x: len(set(input_categories).intersection(set(x)))
            )
            top_matches = df_tv.sort_values('category_similarity', ascending=False)['title'].tolist()
            return jsonify([title for title in top_matches if title.lower() != input_title.lower()][:5])

        # If all else fails
        return jsonify({'error': 'No recommendations found'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
