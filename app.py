import streamlit as st
import pickle
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Ensure artificats folder exists
# ----------------------------
os.makedirs("artificats", exist_ok=True)

# ----------------------------
# Google Drive Download Fix
# ----------------------------
def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive handling confirmation tokens."""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ✅ Google Drive file IDs
MOVIE_LIST_ID = "19y2krbrr0FvgXmz7_2LrILgr2AMR5S8E"
SIMILARITY_ID = "1GjoJsDhAwnohT-mV8G7eUamYjcjTIwaR"

# ----------------------------
# Download files if missing
# ----------------------------
if not os.path.exists("artificats/movie_list.pkl"):
    print("Downloading movie_list.pkl...")
    download_file_from_google_drive(MOVIE_LIST_ID, "artificats/movie_list.pkl")

if not os.path.exists("artificats/similarity.pkl"):
    print("Downloading similarity.pkl...")
    download_file_from_google_drive(SIMILARITY_ID, "artificats/similarity.pkl")

# ----------------------------
# Load pickled data
# ----------------------------
with open('artificats/movie_list.pkl', 'rb') as f:
    movies = pickle.load(f)

with open('artificats/similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Pre-calc TF-IDF for description based search
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

# TMDB API Key
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# ----------------------------
# Fetch Poster Utility
# ----------------------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://placehold.co/500x750?text=No+Image"

# ----------------------------
# Movie Based Recommendation
# ----------------------------
def recommend_movie(movie_title):
    movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

option = st.radio("Choose recommendation type:", ('By Movie', 'By Description'))

# ✅ Movie-Based Search
if option == 'By Movie':
    selected_movie = st.selectbox("Select a movie you like:", movies['title'].values)

    if st.button("Recommend"):
        recommended_movies, recommended_posters = recommend_movie(selected_movie)
        num_results = len(recommended_movies)

        cols = st.columns(num_results)
        for i in range(num_results):
            with cols[i]:
                st.text(recommended_movies[i])
                st.image(recommended_posters[i])

# ✅ Description-Based Search
elif option == 'By Description':
    description = st.text_area("Describe the kind of movie you want to watch:")

    if st.button("Recommend"):
        user_tfidf = tfidf.transform([description])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
        similar_indices = cosine_sim[0].argsort()[-10:][::-1]

        recommended_movies = movies.iloc[similar_indices]['title'].values
        recommended_posters = [fetch_poster(movies.iloc[i].movie_id) for i in similar_indices]

        num_results = min(len(recommended_movies), 10)

        for start in range(0, num_results, 5):
            cols = st.columns(5)
            for idx in range(start, min(start + 5, num_results)):
                with cols[idx - start]:
                    st.text(recommended_movies[idx])
                    st.image(recommended_posters[idx])
