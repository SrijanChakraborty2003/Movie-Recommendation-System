import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# Load merged dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_movies_ratings.csv")

df = load_data()

st.title("🎬 Movie Recommendation System")

# Sidebar Inputs
min_rating = st.slider("Select Minimum Rating", 0.0, 5.0, 3.5, 0.1)
genre = st.selectbox("Select Genre", df['genres'].str.split('|').explode().unique())
num_recommendations = st.slider("Number of Recommendations", 1, 20, 5)

# Filtering based on minimum rating and genre
filtered_df = df[(df['rating'] >= min_rating) & (df['genres'].str.contains(genre, na=False))]

# Pivot Table
user_movie_ratings = filtered_df.pivot(index='userId', columns='title', values='rating')

# Fill missing values
user_means = user_movie_ratings.mean(axis=1)
global_mean = user_movie_ratings.stack().mean()
user_movie_ratings = user_movie_ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
user_movie_ratings = user_movie_ratings.fillna(global_mean)

# Matrix Factorization
matrix = user_movie_ratings.values
U, sigma, Vt = svds(matrix, k=50)
sigma = np.diag(sigma)

# Recommendation Function
def recommend_movies(num_recommendations=5):
    avg_ratings = user_movie_ratings.mean(axis=0).sort_values(ascending=False)
    return avg_ratings.head(num_recommendations)

# Display Recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies(num_recommendations)
    st.write("🎥 **Top Movie Recommendations**:")
    st.write(recommendations)
