import streamlit as st
import pandas as pd
import numpy as np
import joblib
@st.cache_data
def load_data():
    return pd.read_csv("merged_movies_ratings.csv")
df = load_data()
@st.cache_resource
def load_svd_model():
    return joblib.load("svd_model.pkl")
U, sigma, Vt = load_svd_model()
st.title(" Movie Recommendation System")
min_rating = st.slider("Select Minimum Rating", 0.0, 5.0, 3.5, 0.1)
genre = st.selectbox("Select Genre", df['genres'].str.split('|').explode().unique())
num_recommendations = st.slider("Number of Recommendations", 1, 20, 5)
filtered_df = df[(df['rating'] >= min_rating) & (df['genres'].str.contains(genre, na=False))]
user_movie_ratings = filtered_df.pivot(index='userId', columns='title', values='rating')
user_means = user_movie_ratings.mean(axis=1)
user_movie_ratings = user_movie_ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_movie_ratings.index, columns=user_movie_ratings.columns)
def recommend_movies(num_recommendations=5):
    avg_predicted_ratings = predicted_ratings_df.mean(axis=0).sort_values(ascending=False)
    return avg_predicted_ratings.head(num_recommendations)
if st.button("Get Recommendations"):
    recommendations = recommend_movies(num_recommendations)
    st.write(" Top Movie Recommendations (SVD Predicted):")
    st.write(recommendations)