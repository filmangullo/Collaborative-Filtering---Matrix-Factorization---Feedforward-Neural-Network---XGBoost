import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("dataset_hotels/ratings.csv", names=["userId", "itemId", "rating"])

    # Pastikan rating berupa float
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Ambil hanya rating tertinggi jika user memberi lebih dari satu rating ke item yang sama
    df = df.groupby(['userId', 'itemId'], as_index=False)['rating'].max()

    return df

# -----------------------------
# 2. Buat User-Item Matrix
# -----------------------------
def create_user_item_matrix(df):
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    return df.pivot_table(index="userId", columns="itemId", values="rating").fillna(0)

# -----------------------------
# 3. Hitung User Similarity
# -----------------------------
def compute_user_similarity(matrix):
    similarity = cosine_similarity(matrix)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

# -----------------------------
# 4. Prediksi Rating untuk User
# -----------------------------
def predict_ratings_user_based(userId, matrix, sim_df):
    user_similarities = sim_df.loc[userId]
    weighted_sum = user_similarities.dot(matrix)
    sim_sum = user_similarities.dot((matrix > 0).astype(int))
    pred_ratings = weighted_sum / (sim_sum + 1e-9)
    return pd.Series(pred_ratings, index=matrix.columns)

# -----------------------------
# 5. Evaluasi Model
# -----------------------------
def evaluate_model(matrix, sim_df):
    actual, predicted = [], []

    for user in matrix.index:
        user_ratings = matrix.loc[user]
        pred_ratings = predict_ratings_user_based(user, matrix, sim_df)
        for item in user_ratings[user_ratings > 0].index:
            actual.append(user_ratings[item])
            predicted.append(pred_ratings[item])

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return mae, mse, rmse, r2

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("👥 User-Based Collaborative Filtering")

df = load_data()
user_item_matrix = create_user_item_matrix(df)
user_similarity_df = compute_user_similarity(user_item_matrix)

userIds = list(user_item_matrix.index)
selected_user = st.selectbox("Pilih User ID", userIds)

# Prediksi dan Rekomendasi
predicted_ratings = predict_ratings_user_based(selected_user, user_item_matrix, user_similarity_df)
rated_items = user_item_matrix.loc[selected_user]
recommendations = predicted_ratings.drop(rated_items[rated_items > 0].index)
top_n = st.slider("Jumlah Rekomendasi", 1, 10, 5)
top_recommendations = recommendations.sort_values(ascending=False).head(top_n)

st.subheader(f"Rekomendasi untuk User {selected_user}")
top_recommendations_df = top_recommendations.reset_index().rename(columns={selected_user: "Predicted Rating"})
top_recommendations_df.columns = top_recommendations_df.columns.astype(str)
st.dataframe(top_recommendations_df)

# Evaluasi model
st.subheader("📊 Evaluasi Model (Seluruh Data)")
with st.spinner("Menghitung evaluasi..."):
    mae, mse, rmse, r2 = evaluate_model(user_item_matrix, user_similarity_df)
    st.metric("MAE", f"{mae:.4f}")
    st.metric("MSE", f"{mse:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("R2", f"{r2:.4f}")
