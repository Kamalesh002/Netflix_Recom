import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
from textblob import TextBlob

# Load Lottie Animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Data Loading and Preprocessing
df = pd.read_csv('netflix_titles.csv')
df.rename(columns={'listed_in': 'genres'}, inplace=True)
df['rating'].fillna('NaN', inplace=True)
df.dropna(inplace=True)
df = df.reset_index(drop=True)

# Filter Movies and TV Shows
movies_df = df[df['type'] == 'Movie'].reset_index(drop=True)
tv_show = df[df['type'] == 'TV Show'].reset_index(drop=True)

# Text Preprocessing
movies = movies_df[['title', 'director', 'cast', 'country', 'rating', 'genres', 'description', 'release_year']]
movies['director'] = movies['director'].apply(nfx.remove_stopwords)
movies['cast'] = movies['cast'].apply(nfx.remove_stopwords)
movies['country'] = movies['country'].apply(nfx.remove_stopwords)
movies['genres'] = movies['genres'].apply(nfx.remove_stopwords)
movies['country'] = movies['country'].apply(nfx.remove_special_characters)

# Vectorization
countVector = CountVectorizer(binary=True)
country = countVector.fit_transform(movies['country']).toarray()
director = countVector.fit_transform(movies['director']).toarray()
cast = countVector.fit_transform(movies['cast']).toarray()
genres = countVector.fit_transform(movies['genres']).toarray()

# Combine Vectors
movies_binary = np.concatenate([director, cast, country, genres], axis=1)
movies_sim = cosine_similarity(movies_binary)

# Repeat for TV Shows
tv_df = tv_show[['title', 'director', 'cast', 'country', 'rating', 'genres', 'description', 'release_year']]
tv_df['cast'] = tv_df['cast'].apply(nfx.remove_stopwords)
tv_df['country'] = tv_df['country'].apply(nfx.remove_stopwords)
tv_df['genres'] = tv_df['genres'].apply(nfx.remove_stopwords)
tv_df['country'] = tv_df['country'].apply(nfx.remove_special_characters)

cast = countVector.fit_transform(tv_df['cast']).toarray()
country = countVector.fit_transform(tv_df['country']).toarray()
genres = countVector.fit_transform(tv_df['genres']).toarray()

tv_binary = np.concatenate([cast, country, genres], axis=1)
tv_sim = cosine_similarity(tv_binary)

# Recommendation Function
def recommend(title):
    if title in movies_df['title'].values:
        index = movies_df[movies_df['title'] == title].index.item()
        scores = dict(enumerate(movies_sim[index]))
    elif title in tv_show['title'].values:
        index = tv_show[tv_show['title'] == title].index.item()
        scores = dict(enumerate(tv_sim[index]))
    else:
        return None
    
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    selected_index = list(sorted_scores.keys())
    selected_scores = list(sorted_scores.values())
    
    if title in movies_df['title'].values:
        rec_df = movies_df.iloc[selected_index].copy()
    else:
        rec_df = tv_show.iloc[selected_index].copy()
        
    rec_df['similarity'] = selected_scores
    return rec_df[1:6]  # Skip the first row as it's the selected movie/TV show

# Sentiment Analysis Function
def predict_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Streamlit Interface
st.header('Netflix Movie Recommendation System')
lottie_coding = load_lottiefile("netflix-logo.json")
st_lottie(lottie_coding, speed=1, loop=True, quality="low", height=220)

movie_list = sorted(movies_df['title'].tolist() + tv_df['title'].tolist())
selected_movie = st.selectbox("Type or select a movie/TV show from the dropdown", movie_list)

if st.button('Show Recommendation'):
    # Display summary of the selected movie/TV show
    if selected_movie in movies_df['title'].values:
        selected_summary = movies_df[movies_df['title'] == selected_movie].iloc[0]
    else:
        selected_summary = tv_df[tv_df['title'] == selected_movie].iloc[0]
    
    st.subheader("Selected Movie/TV Show Summary")
    st.write(f"**Title:** {selected_summary['title']}")
    st.write(f"**Director:** {selected_summary['director']}")
    st.write(f"**Cast:** {selected_summary['cast']}")
    st.write(f"**Country:** {selected_summary['country']}")
    st.write(f"**Rating:** {selected_summary['rating']}")
    st.write(f"**Genres:** {selected_summary['genres']}")
    st.write(f"**Release Year:** {selected_summary['release_year']}")
    st.write(f"**Description:** {selected_summary['description']}")
    
    # Show recommendations
    recommended_movie_names = recommend(selected_movie)
    if recommended_movie_names is not None:
        st.subheader("Top 5 Recommended Movies/TV Shows")
        st.dataframe(recommended_movie_names[['title', 'country', 'genres', 'rating', 'similarity']])

        # Sentiment Analysis on movie/TV show descriptions
        sentiments = recommended_movie_names['title'].apply(lambda x: predict_sentiment(x))
        recommended_movie_names['Sentiment'] = sentiments
        st.subheader("Sentiment Analysis")
        st.dataframe(recommended_movie_names[['title', 'Sentiment']])
    else:
        st.error("Title not found in dataset. Please check the spelling.")
