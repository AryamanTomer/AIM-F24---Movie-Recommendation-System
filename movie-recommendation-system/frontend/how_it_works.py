st.header("Video Script: Netflix Movie Recommendation System")
st.markdown("""
**[Opening]**
Hi, I’m Aryaman, and welcome to my Netflix Movie Recommendation System! In this video, I’ll show you how this full-stack app helps you discover movies and explore the entire Netflix dataset interactively.

---

**[What the App Does]**
This app provides:
- Personalized movie recommendations based on your selected title and preferences
- A data explorer to browse, search, and filter all Netflix movies
- Interactive statistics and visualizations to understand trends and correlations in the dataset

---

**[How It Works]**
The backend is built with Flask, pandas, and scikit-learn. It loads the Netflix dataset, preprocesses features like genre, country, director, cast, rating, release year, and uses advanced NLP (TF-IDF) for movie descriptions. The recommendation engine uses K-Nearest Neighbors and cosine similarity, letting you tune feature weights for personalized results.

The frontend is built with Streamlit. It provides an easy-to-use interface for entering a movie title, adjusting recommendation parameters, exploring the dataset, and viewing interactive charts.

---

**[Demo Walkthrough]**
Let’s walk through the app:
1. On the main page, enter a Netflix movie title.
2. Adjust the number of recommendations, similarity threshold, and feature weights to prioritize aspects like genre or description.
3. Click "Get Recommendations" to see a ranked list of similar movies, each with detailed metadata and similarity scores.
4. Explore all movies in the dataset, filter by genre or country, and view the results in a searchable table.
5. Check out the statistics section for genre and country distributions, and a genre-country correlation heatmap for deeper insights.

---

**[Technical Details]**
The frontend communicates with the backend via REST API endpoints:
- `/recommend` for recommendations
- `/movies` for the full movie list
- `/stats` for analytics

All data is processed and returned in real time, and the app is ready for deployment with Docker and Kubernetes.

---

**[How to Use]**
To run the app:
1. Start the Flask backend: `python app.py`
2. Start the Streamlit frontend: `streamlit run frontendApp.py`
3. Open the app in your browser and start exploring!

---

**[Conclusion]**
This Netflix Movie Recommendation System is flexible, extensible, and user-friendly. You can add new features, connect to other datasets, or deploy it in the cloud. If you’re interested, check out the codebase and feel free to contribute!

Thanks for watching!
""")
import streamlit as st

st.title("Movie Recommendation System: In-Depth Guide")

st.markdown("""
## Overview

This application is a comprehensive, full-stack movie recommendation system built on a Netflix movie dataset. It combines a robust Flask backend with an interactive Streamlit frontend, providing users with personalized recommendations, advanced analytics, and a highly tunable experience.

---

## Data Pipeline

- **Source:** The app uses `netflix_titles.csv`, containing thousands of movies with metadata: title, genre, country, director, cast, rating, release year, and description.
- **Preprocessing:**
	- Unused columns are dropped for efficiency.
	- Categorical features (genre, country, director, cast, rating) are encoded using one-hot or multi-label binarization.
	- Release year is normalized for fair comparison.
	- Descriptions are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency), capturing semantic meaning and keywords.

---

## Backend (Flask)

- **Feature Engineering:**
	- All features are combined into a single matrix representing each movie numerically.
	- Feature weights allow dynamic adjustment of importance (e.g., prioritize genre or description).
- **Recommendation Algorithm:**
	- Uses K-Nearest Neighbors (KNN) with cosine similarity to find movies most similar to the selected title.
	- Returns top-k recommendations, each with a similarity score and full metadata.
- **API Endpoints:**
	- `/recommend`: Accepts a movie title, number of recommendations, similarity threshold, and feature weights. Returns recommendations and metadata.
	- `/movies`: Returns the full movie list and metadata for browsing/searching.
	- `/stats`: Returns dataset statistics (e.g., genre distribution, correlations).
- **Error Handling:**
	- Handles missing/invalid titles gracefully.
	- Ensures all data returned is JSON serializable.

---

## Frontend (Streamlit)

- **User Controls:**
	- Select a movie to get recommendations.
	- Adjust number of recommendations, similarity threshold, and feature weights via sliders.
	- Search, filter, and sort the movie list.
- **Visualization:**
	- Interactive charts show genre distribution, rating breakdown, and correlations.
	- Rich metadata for each recommended movie: title, genre, country, director, cast, rating, release year, description, similarity score.
- **Navigation:**
	- Sidebar navigation between main app and documentation.
	- All features are accessible in a unified interface.

---

## Recommendation Logic (Step-by-Step)

1. **User selects a movie and sets parameters.**
2. **Frontend sends request to backend `/recommend` endpoint.**
3. **Backend locates the movie, extracts its feature vector.**
4. **KNN finds the k most similar movies using cosine similarity.**
5. **Feature weights are applied to emphasize user preferences.**
6. **Backend returns recommendations, similarity scores, and metadata.**
7. **Frontend displays results with interactive controls and charts.**

---

## Advanced Features

- **Feature Weighting:** Users can tune the influence of genre, country, director, cast, rating, year, and description.
- **NLP for Descriptions:** TF-IDF captures important words and context from movie descriptions, improving recommendation quality.
- **Rich Metadata:** Every recommendation includes all available metadata for transparency and exploration.
- **Interactive Analytics:** Users can explore dataset statistics, correlations, and trends visually.

---

## Deployment & Extensibility

- **Docker & Kubernetes:** Ready for scalable deployment in cloud or local environments.
- **Extending the System:**
	- Add new features (e.g., user ratings, reviews, tags).
	- Integrate collaborative filtering for user-based recommendations.
	- Connect to other datasets or streaming platforms.
	- Enhance NLP with embeddings or deep learning for descriptions.

---

## How to Use

1. Enter a movie title in the search box.
2. Adjust recommendation parameters and feature weights as desired.
3. View recommended movies with rich metadata and similarity scores.
4. Explore all movies, filter/search, and view interactive statistics.
5. Read this page for details on how the system works.

---

## Example Use Case

Suppose you want recommendations similar to "Inception" but care most about genre and description. Select "Inception", set genre and description weights higher, and adjust the number of recommendations. The app will return movies most similar in those aspects, with full metadata and similarity scores.

---

## Adapting for Other Projects

- Add more features (e.g., user ratings, reviews) for richer recommendations.
- Use other similarity metrics or algorithms (e.g., collaborative filtering if you have user data).
- Adjust feature weights to prioritize certain aspects.
- Integrate with other data sources or APIs.

---

## Technical Stack

- **Backend:** Python, Flask, scikit-learn, pandas, numpy
- **Frontend:** Python, Streamlit, Altair
- **Deployment:** Docker, Kubernetes

---

## Support & Further Reading

- For technical details, see the codebase and README.
- For questions or contributions, visit the project repository.
""")