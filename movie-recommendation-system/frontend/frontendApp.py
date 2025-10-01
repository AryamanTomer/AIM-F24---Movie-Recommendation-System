
import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt

def show_recommendation_system():
    st.title("Netflix Recommendation System & Data Explorer")
    st.markdown("""
    This app recommends Netflix movies and lets you explore the entire dataset, view statistics, and correlations.
    """)
    # --- Recommendation Section ---
    st.header("Get Recommendations")
    title = st.text_input("Enter a Netflix movie title", "")
    k = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
    min_similarity = st.slider("Minimum similarity threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    st.markdown("**Feature Weights** (adjust to prioritize certain features)")
    genre_weight = st.slider("Genre weight", 0.0, 2.0, 1.0, 0.1)
    country_weight = st.slider("Country weight", 0.0, 2.0, 1.0, 0.1)
    director_weight = st.slider("Director weight", 0.0, 2.0, 0.7, 0.1)
    cast_weight = st.slider("Cast weight", 0.0, 2.0, 0.7, 0.1)
    rating_weight = st.slider("Rating weight", 0.0, 2.0, 0.5, 0.1)
    year_weight = st.slider("Release year weight", 0.0, 2.0, 0.5, 0.1)
    desc_weight = st.slider("Description weight", 0.0, 2.0, 1.0, 0.1)
    weights = {
        "genre": genre_weight,
        "country": country_weight,
        "director": director_weight,
        "cast": cast_weight,
        "rating": rating_weight,
        "year": year_weight,
        "desc": desc_weight
    }
    if st.button("Get Recommendations"):
        if title.strip():
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/recommend",
                    json={
                        "title": title,
                        "k": k,
                        "min_similarity": min_similarity,
                        "weights": weights
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    metadata = data.get("metadata", [])
                    if recommendations:
                        st.success(f"Recommendations for '{data.get('title', '')}':")
                        for idx, rec in enumerate(metadata, 1):
                            st.markdown(f"**{idx}. {rec['title']}** (Similarity: {rec['similarity']:.3f})")
                            st.write(f"Genre: {rec['genre']}")
                            st.write(f"Country: {rec['country']}")
                            st.write(f"Director: {rec['director']}")
                            st.write(f"Cast: {rec['cast']}")
                            st.write(f"Rating: {rec['rating']}")
                            st.write(f"Release Year: {rec['release_year']}")
                            st.write(f"Description: {rec['description']}")
                            st.markdown("---")
                    else:
                        st.warning("No recommendations found. Try another title or adjust the weights/threshold!")
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")
        else:
            st.warning("Please enter a title.")

    # --- Data Explorer Section ---
    st.header("Explore All Movies")
    try:
        from io import StringIO
        movies_response = requests.get("http://127.0.0.1:5000/movies")
        movies_df = pd.read_json(StringIO(movies_response.text))
        # Filtering options
        genre_filter = st.multiselect("Filter by Genre", options=sorted(set(g for gs in movies_df['listed_in'].dropna() for g in gs.split(','))), default=[])
        country_filter = st.multiselect("Filter by Country", options=sorted(set(c for cs in movies_df['country'].dropna() for c in cs.split(','))), default=[])
        filtered_df = movies_df.copy()
        if genre_filter:
            filtered_df = filtered_df[filtered_df['listed_in'].apply(lambda x: any(g.strip() in genre_filter for g in x.split(',')) if pd.notna(x) else False)]
        if country_filter:
            filtered_df = filtered_df[filtered_df['country'].apply(lambda x: any(c.strip() in country_filter for c in x.split(',')) if pd.notna(x) else False)]
        st.dataframe(filtered_df)
    except Exception as e:
        st.error(f"Could not load movies: {e}")

    # --- Statistics & Visualization Section ---
    st.header("Statistics & Correlations")
    try:
        stats_response = requests.get("http://127.0.0.1:5000/stats")
        stats = stats_response.json()
        # Genre distribution
        genre_counts = pd.Series(stats['genre_counts'])
        st.subheader("Genre Distribution")
        st.bar_chart(genre_counts)
        # Country distribution
        country_counts = pd.Series(stats['country_counts'])
        st.subheader("Country Distribution")
        st.bar_chart(country_counts)
        # Correlation heatmap (genre-country)
        st.subheader("Genre-Country Correlation Heatmap")
        corr_df = pd.DataFrame(stats['correlation']).fillna(0)
        if not corr_df.empty:
            st.dataframe(corr_df)
            # Optional: show as heatmap
            heatmap = alt.Chart(corr_df.reset_index().melt('index')).mark_rect().encode(
                x='variable:O',
                y='index:O',
                color='value:Q'
            )
            st.altair_chart(heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load stats: {e}")

    # Footer
    st.markdown("---")
    st.caption("Netflix Recommendation System & Data Explorer | Powered by Flask & Streamlit")

def show_how_it_works():
        st.title("How the Netflix Recommendation System Works: Full App Guide")
        st.markdown("""
## Overview

This app is a full-stack Netflix movie recommendation and data explorer system. It combines a Flask backend (API and recommendation engine) with a Streamlit frontend (interactive UI, analytics, and visualizations).

---

## Backend (Flask)

- **Data Loading & Preprocessing:**
    - Loads the Netflix dataset (`netflix_titles.csv`).
    - Filters for movies and drops unused columns for efficiency.
    - Encodes categorical features (genre, country, director, cast, rating) using one-hot/multi-label binarization.
    - Normalizes release year.
    - Converts movie descriptions to TF-IDF vectors for semantic analysis.

- **Feature Engineering:**
    - All features are combined into a single matrix for each movie.
    - Feature weights allow dynamic adjustment of importance (genre, director, description, etc.).

- **Recommendation Algorithm:**
    - Uses K-Nearest Neighbors (KNN) with cosine similarity to find movies most similar to a selected title.
    - Returns top-k recommendations, each with a similarity score and full metadata.

- **API Endpoints:**
    - `/recommend`: POST endpoint for recommendations. Accepts title, k, min_similarity, and feature weights. Returns recommendations and metadata.
    - `/movies`: GET endpoint for the full movie list and metadata.
    - `/stats`: GET endpoint for dataset statistics (genre/country distribution, genre-country correlation).

- **Error Handling:**
    - Handles missing/invalid titles gracefully.
    - Ensures all returned data is JSON serializable.

---

## Frontend (Streamlit)

- **Recommendation System:**
    - Users enter a movie title and adjust parameters (number of recommendations, similarity threshold, feature weights).
    - Results show recommended movies with full metadata and similarity scores.

- **Data Explorer:**
    - Browse, search, and filter the entire movie dataset by genre and country.
    - View all available metadata for each movie.

- **Statistics & Visualizations:**
    - Interactive charts for genre and country distribution.
    - Genre-country correlation heatmap for deeper insights.

- **Navigation:**
    - Sidebar lets users switch between the recommendation system and the documentation/how-it-works page.

---

## Data Flow & Integration

1. **User interacts with Streamlit frontend.**
2. **Frontend sends requests to Flask backend API endpoints.**
3. **Backend processes requests, runs recommendation logic, and returns results.**
4. **Frontend displays recommendations, analytics, and visualizations.**

---

## Recommendation Logic (Step-by-Step)

1. User selects a movie and sets parameters (k, min_similarity, feature weights).
2. Frontend sends a POST request to `/recommend` with these parameters.
3. Backend locates the movie, extracts its feature vector.
4. KNN finds the k most similar movies using cosine similarity.
5. Feature weights are applied to emphasize user preferences.
6. Backend returns recommendations, similarity scores, and metadata.
7. Frontend displays results with interactive controls and charts.

---

## Features & Customization

- **Feature Weighting:** Tune the influence of genre, country, director, cast, rating, year, and description.
- **NLP for Descriptions:** TF-IDF captures important words and context from movie descriptions.
- **Rich Metadata:** Every recommendation includes all available metadata for transparency and exploration.
- **Interactive Analytics:** Explore dataset statistics, correlations, and trends visually.

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

## Extending the System

- Add new features (e.g., user ratings, reviews, tags).
- Integrate collaborative filtering for user-based recommendations.
- Connect to other datasets or streaming platforms.
- Enhance NLP with embeddings or deep learning for descriptions.

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

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Recommendation System", "How It Works"])

if page == "Recommendation System":
    show_recommendation_system()
else:
    show_how_it_works()
