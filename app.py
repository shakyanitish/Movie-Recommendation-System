import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import streamlit as st

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500" + data['poster_path'] if data.get('poster_path') else ""
    except:
        return ""

@st.cache_data
def load_movies():
    movies = pd.read_csv(
        'movies_metadata.csv',
        usecols=['id', 'title', 'overview', 'vote_count', 'vote_average'],
        dtype={'id': str, 'title': str, 'overview': str, 'vote_count': float, 'vote_average': float},
        low_memory=True
    )
    movies = movies[movies['vote_count'] >= 50]
    movies['overview'] = movies['overview'].fillna('')
    return movies

movies = load_movies()

@st.cache_data
def vectorize_overview(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform(movies['overview'])
    return vectors

vectors = vectorize_overview(movies)

def recommend(movie_title, top_n=5):
    movie_title_lower = movie_title.lower()
    if movie_title_lower not in movies['title'].str.lower().values:
        return [], []
    idx = movies[movies['title'].str.lower() == movie_title_lower].index[0]
    movie_vector = vectors[idx]
    distances = cosine_similarity(movie_vector, vectors).flatten()
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
    recommended_movies = []
    recommended_posters = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]]['id']
        recommended_movies.append(movies.iloc[i[0]]['title'])
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

st.title("🎬 Movie Recommender System")
selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    if names:
        cols = st.columns(len(names))
        for col, name, poster in zip(cols, names, posters):
            col.subheader(name)
            if poster:
                col.image(poster, use_container_width=True)
    else:
        st.write("No recommendations found. Try another movie.")
