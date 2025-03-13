import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_movies_ratings.csv")

df = load_data()

# Load SVD model
@st.cache_resource
def load_svd_model():
    U, sigma, Vt, original_user_ids, original_movie_titles = joblib.load("svd_model.pkl")
    sigma = np.diag(sigma)  # Convert sigma back to diagonal matrix
    return U, sigma, Vt, original_user_ids, original_movie_titles

U, sigma, Vt, original_user_ids, original_movie_titles = load_svd_model()

# Streamlit UI
st.title("Movie Recommendation System")
min_rating = st.slider("Select Minimum Rating", 0.0, 5.0, 3.5, 0.1)
genre = st.selectbox("Select Genre", df['genres'].str.split('|').explode().unique())
num_recommendations = st.slider("Number of Recommendations", 1, 20, 5)

# Step 1: Create User-Movie Ratings Matrix (Using Full Data)
user_movie_ratings = df.pivot(index='userId', columns='title', values='rating')

# Step 2: Fill Missing Values
user_movie_ratings = user_movie_ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
user_movie_ratings = user_movie_ratings.fillna(user_movie_ratings.stack().mean())

# Step 3: Ensure Shape Matches SVD Training Data
user_movie_ratings = user_movie_ratings.reindex(index=original_user_ids, columns=original_movie_titles, fill_value=0)

# Drop extra users/movies that weren’t in training
user_movie_ratings = user_movie_ratings.loc[original_user_ids, original_movie_titles]

# Convert to NumPy array
matrix = user_movie_ratings.to_numpy()

# 🔍 Debugging - Check Shape Before SVD
st.write("User-Movie Ratings Shape:", user_movie_ratings.shape)
st.write("Matrix Shape Before SVD:", matrix.shape)
st.write("U Shape:", U.shape)
st.write("Sigma Shape:", sigma.shape)
st.write("Vt Shape:", Vt.shape)

# Ensure correct shape before applying SVD
if matrix.shape != (U.shape[0], Vt.shape[1]):
    raise ValueError(f"Shape mismatch: user_movie_ratings {matrix.shape}, expected {U.shape[0], Vt.shape[1]}")

# Apply SVD
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_movie_ratings.index, columns=user_movie_ratings.columns)

# Step 5: Apply Filtering AFTER Predictions
filtered_movies = df[(df['rating'] >= min_rating) & (df['genres'].str.contains(genre, na=False))]['title']
predicted_ratings_df = predicted_ratings_df[filtered_movies]  # Apply filtering after predictions

# Step 6: Recommendation Function
def recommend_movies(num_recommendations=5):
    avg_predicted_ratings = predicted_ratings_df.mean(axis=0).sort_values(ascending=False)
    return avg_predicted_ratings.head(num_recommendations)

# Step 7: Streamlit Button to Get Recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies(num_recommendations)
    st.write("Top Movie Recommendations (SVD Predicted):")
    st.write(recommendations)
