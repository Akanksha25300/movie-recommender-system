
import pickle
import streamlit as st
import requests
import pandas as pd

def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    try:
        data = requests.get(url)
        data.raise_for_status()  # Raise an exception for bad status codes
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")
    # Return a placeholder if the poster is not found or an error occurs
    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"


def recommend(movie):
    """Recommends 5 similar movies based on the selected movie."""
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in the dataset. Please select another one.")
        return [], []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_data = movies.iloc[i[0]]
        recommended_movie_names.append(movie_data.get('title', 'N/A'))
        recommended_movie_posters.append(fetch_poster(movie_data.get('movie_id')))

    return recommended_movie_names, recommended_movie_posters




st.set_page_config(layout="wide")
st.header('Movie Recommender System Using Machine Learning')

# Load the data files
try:
    movies_dict = pickle.load(open('artificats/movie_list.pkl','rb'))
    movies = pd.DataFrame(movies_dict)
    
    # Print column names to terminal
    print("Movies DataFrame columns:", movies.columns)
    
    similarity = pickle.load(open('artificats/similarity.pkl','rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the data processing notebook first.")
    st.stop()


movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    with st.spinner('Finding recommendations...'):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

        if recommended_movie_names:
           cols = st.columns(5)
           for i, col in enumerate(cols):
              with col:
                 st.text(recommended_movie_names[i])
                 st.image(recommended_movie_posters[i])

