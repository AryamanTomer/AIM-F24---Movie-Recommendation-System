import streamlit as st
import requests

# Title and description
st.title("Netflix Recommendation System")
st.markdown("""
This app recommends Netflix shows and movies based on the title you input. 
It filters results by genre and shared cast members.
""")

# Input section
title = st.text_input("Enter a Netflix show or movie title", "")

# Button to get recommendations
if st.button("Get Recommendations"):
    if title.strip():
        try:
            # Send request to Flask backend
            response = requests.post(
                "http://127.0.0.1:5000/recommend",  # Adjust this URL if Flask runs on a different host/port
                json={"title": title}
            )
            # Handle response
            if response.status_code == 200:
                recommendations = response.json()
                if recommendations:
                    st.success("Here are the recommendations:")
                    for idx, rec in enumerate(recommendations, 1):
                        st.write(f"{idx}. {rec}")
                else:
                    st.warning("No recommendations found. Try another title!")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
    else:
        st.warning("Please enter a title.")

# Footer
st.markdown("---")
st.caption("Netflix Recommendation System | Powered by Flask & Streamlit")