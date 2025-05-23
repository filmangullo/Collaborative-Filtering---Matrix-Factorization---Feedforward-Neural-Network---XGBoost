import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Fungsi Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_dummy/ratings.csv")
    return df

# -----------------------------
# Fungsi User-Item Matrix
# -----------------------------
def create_user_item_matrix(df):
    return df.pivot_table(index="userId", columns="itemId", values="rating").fillna(0)

# -----------------------------
# Fungsi Kemiripan dan Prediksi
# -----------------------------
def predict_ratings(user_item_matrix, userId):
    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    sim_scores = similarity_df[userId]
    sim_scores[userId] = 0  # exclude diri sendiri

    weighted_sum = user_item_matrix.T.dot(sim_scores)
    sim_sum = np.array([sim_scores.sum()] * user_item_matrix.shape[1])

    pred_ratings = weighted_sum / (sim_sum + 1e-9)
    return pred_ratings, similarity_df

# -----------------------------
# Evaluasi Keseluruhan
# -----------------------------
def evaluate_model(user_item_matrix, similarity_df):
    actual = []
    predicted = []

    for user in user_item_matrix.index:
        sim_scores = similarity_df[user]
        sim_scores[user] = 0

        weighted_sum = user_item_matrix.T.dot(sim_scores)
        sim_sum = np.array([sim_scores.sum()] * user_item_matrix.shape[1])
        pred = weighted_sum / (sim_sum + 1e-9)

        rated_items = user_item_matrix.loc[user]
        for item in rated_items[rated_items > 0].index:
            actual.append(rated_items[item])
            predicted.append(pred[item])

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎥 User-Based Collaborative Filtering - MovieLens")

df = load_data()
user_item_matrix = create_user_item_matrix(df)

userIds = list(user_item_matrix.index)
selected_user = st.selectbox("Pilih User ID", userIds)

# Prediksi & Kemiripan
predicted_ratings, similarity_df = predict_ratings(user_item_matrix, selected_user)

# Tampilkan rekomendasi
rated_items = user_item_matrix.loc[selected_user]
recommendations = predicted_ratings.drop(rated_items[rated_items > 0].index)
top_n = st.slider("Jumlah Rekomendasi", 1, 10, 5)
top_recommendations = recommendations.sort_values(ascending=False).head(top_n)

st.subheader(f"Rekomendasi untuk User {selected_user}")
st.dataframe(top_recommendations.reset_index().rename(columns={selected_user: "Predicted Rating"}))

# Evaluasi model
if st.checkbox("Tampilkan Evaluasi MAE, MSE, RMSE"):
    with st.spinner("Menghitung evaluasi..."):
        mae, mse, rmse = evaluate_model(user_item_matrix, similarity_df)
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
