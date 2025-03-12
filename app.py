
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# Load merged dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_movies_ratings.csv")

df = load_data()

st.title("Movie Recommender System")

# Pivot Table
user_movie_ratings = df.pivot(index='userId', columns='title', values='rating')

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
def recommend_movies(user_id, num_recommendations=5):
    user_ratings_pred = np.dot(np.dot(U[user_id-1], sigma), Vt)
    user_ratings_pred_df = pd.DataFrame(user_ratings_pred, index=user_movie_ratings.columns, columns=["Predicted Rating"])
    top_recommendations = user_ratings_pred_df.sort_values("Predicted Rating", ascending=False).head(num_recommendations)
    return top_recommendations

# User Input
user_id = st.number_input("Enter User ID:", min_value=1, max_value=len(user_movie_ratings), step=1)
if st.button("Recommend"):
    st.write(recommend_movies(user_id))
