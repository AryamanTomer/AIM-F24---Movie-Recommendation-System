import streamlit as st  # Import Streamlit for the frontend interface
import requests  # Import requests to make HTTP requests to the backend

# Setup the title of the streamlit application

st.title("Netflix Movie/Show Recommendation App")

# Input box for the user to enter a movie title
tv_title = st.text_input("Enter a movie or show title:")

# KNN Recommendations Button
if st.button("Get KNN Recommendations"):
    if tv_title:
        try:
            # Send a POST request to the Flask backend for KNN recommendations
            response = requests.post('http://localhost:5000/recommend_knn', json={'title': tv_title})
            # Get the recommendations from the response
            recommendations = response.json()  
            
            if 'error' in recommendations:
                # Display error if there is one
                st.error(recommendations['error'])
            else:
                # Display the top 5 KNN recommendations
                st.write("Top 5 KNN Recommendations:")
                for tv in recommendations:
                    st.write(tv)
        except Exception as e:
            st.error(f"Error: {str(e)}")  # Display any errors
    else:
        st.warning("Please enter a movie or show title.")  # Warn if no title is entered
        
# K-Means Recommendations button
if st.button("Get K-Means Recommendations"):
    if tv_title:
        try:
            # Send a POST request to the Flask backend for K-Means recommendations
            response = requests.post('http://localhost:5000/recommend_kmeans', json={'title': tv_title})
            recommendations = response.json()  # Get the recommendations from the response
            
            if 'error' in recommendations:
                # Display error if there is one
                st.error(recommendations['error'])
            else:
                # Display the top 5 K-Means recommendations
                st.write("Top 5 K-Means Recommendations:")
                for movie in recommendations:
                    st.write(movie)
        except Exception as e:
            st.error(f"Error: {str(e)}")  # Display any errors
    else:
        st.warning("Please enter a movie or show title.")  # Warn if no title is entered

# SVD Recommendations button
if st.button("Get SVD Recommendations"):
    if tv_title:
        try:
            # Send a POST request to the Flask backend for SVD recommendations
            response = requests.post('http://localhost:5000/recommend_svd', json={'title': tv_title})
             # Get the recommendations from the response
            recommendations = response.json() 
            
            if 'error' in recommendations:
                # Display error if there is one
                st.error(recommendations['error'])
            else:
                # Display the top 5 SVD recommendations
                st.write("Top 5 SVD Recommendations:")
                for movie in recommendations:
                    st.write(movie)
        except Exception as e:
            st.error(f"Error: {str(e)}")  # Display any errors
    else:
        st.warning("Please enter a movie or show title.")  # Warn if no title is entered