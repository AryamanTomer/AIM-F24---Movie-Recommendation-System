import streamlit as st
import requests


import pandas as pd
import numpy as np
import altair as alt

st.title("Netflix Recommendation System & Data Explorer")
st.markdown("""
This app recommends Netflix movies and lets you explore the entire dataset, view statistics, and correlations.
""")

# --- Recommendation Section ---
st.header("Get Recommendations")
title = st.text_input("Enter a Netflix movie title", "")
if st.button("Get Recommendations"):
    if title.strip():
        try:
            response = requests.post(
                "http://127.0.0.1:5000/recommend",
                json={"title": title}
            )
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendations", [])
                similarity_scores = data.get("similarity_scores", [])
                if recommendations:
                    st.success(f"Recommendations for '{data.get('title', '')}':")
                    for idx, (rec, score) in enumerate(zip(recommendations, similarity_scores), 1):
                        st.write(f"{idx}. {rec} (Similarity: {score:.3f})")
                else:
                    st.warning("No recommendations found. Try another title!")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
    else:
        st.warning("Please enter a title.")

# --- Data Explorer Section ---
st.header("Explore All Movies")
try:
    movies_response = requests.get("http://127.0.0.1:5000/movies")
    movies_df = pd.read_json(movies_response.text)
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
