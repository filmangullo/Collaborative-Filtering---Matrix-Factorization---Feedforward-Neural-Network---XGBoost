import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    # Jika CSV kamu sudah memiliki header, hapus parameter names=[]
    return pd.read_csv("dataset_hotels/ratings.csv", names=["userId", "itemId", "rating", "timestamp"])

# -----------------------------
# 2. Buat Item-User Matrix
# -----------------------------
def create_item_user_matrix(df):
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # pastikan rating adalah float
    return df.pivot_table(index="itemId", columns="userId", values="rating").fillna(0)

# -----------------------------
# 3. Hitung Item Similarity
# -----------------------------
def compute_item_similarity(matrix):
    similarity = cosine_similarity(matrix)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

# -----------------------------
# 4. Prediksi Rating untuk User
# -----------------------------
def predict_ratings_item_based(userId, matrix, sim_df):
    user_ratings = matrix.T.loc[userId]
    weighted_sum = sim_df.dot(user_ratings)
    sim_sum = sim_df.dot((user_ratings > 0).astype(int))
    pred_ratings = weighted_sum / (sim_sum + 1e-9)
    return pred_ratings

# -----------------------------
# 5. Evaluasi Model
# -----------------------------
def evaluate_model(matrix, sim_df):
    actual, predicted = [], []

    for user in matrix.columns:
        user_ratings = matrix.T.loc[user]
        pred_ratings = predict_ratings_item_based(user, matrix, sim_df)
        for item in user_ratings[user_ratings > 0].index:
            actual.append(user_ratings[item])
            predicted.append(pred_ratings[item])

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🍿 Item-Based Collaborative Filtering - MovieLens")

df = load_data()
item_user_matrix = create_item_user_matrix(df)
item_similarity_df = compute_item_similarity(item_user_matrix)

userIds = list(item_user_matrix.columns)
selected_user = st.selectbox("Pilih User ID", userIds)

# Prediksi dan Rekomendasi
predicted_ratings = predict_ratings_item_based(selected_user, item_user_matrix, item_similarity_df)
rated_items = item_user_matrix.T.loc[selected_user]
recommendations = predicted_ratings.drop(rated_items[rated_items > 0].index)
top_n = st.slider("Jumlah Rekomendasi", 1, 10, 5)
top_recommendations = recommendations.sort_values(ascending=False).head(top_n)

st.subheader(f"Rekomendasi untuk User {selected_user}")
st.dataframe(top_recommendations.reset_index().rename(columns={selected_user: "Predicted Rating"}))

# Evaluasi model
if st.checkbox("Tampilkan Evaluasi MAE, MSE, RMSE"):
    with st.spinner("Menghitung evaluasi..."):
        mae, mse, rmse = evaluate_model(item_user_matrix, item_similarity_df)
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
