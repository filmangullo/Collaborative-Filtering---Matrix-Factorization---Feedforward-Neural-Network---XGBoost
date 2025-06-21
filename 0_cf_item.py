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
    df = pd.read_csv("dataset_hotels/ratings.csv", names=["userId", "itemId", "rating"])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.groupby(['userId', 'itemId'], as_index=False)['rating'].max()
    return df

# -----------------------------
# 2. Buat Item-User Matrix
# -----------------------------
def create_item_user_matrix(df):
    matrix = df.pivot_table(index="itemId", columns="userId", values="rating")
    return matrix.fillna(0)

# -----------------------------
# 3. Hitung Kemiripan Antar Item
# -----------------------------
def compute_item_similarity(matrix):
    similarity = cosine_similarity(matrix)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

# -----------------------------
# 4. Prediksi Rating untuk User (Item-Based)
# -----------------------------
def predict_ratings_item_based(user_id, matrix_T, similarity_df):
    if user_id not in matrix_T.columns:
        return pd.Series([], dtype=np.float64)

    user_ratings = matrix_T[user_id]
    rated_items = user_ratings[user_ratings > 0]

    weighted_scores = similarity_df[rated_items.index] @ rated_items
    similarity_sums = similarity_df[rated_items.index].abs().sum(axis=1)

    predicted_ratings = weighted_scores / (similarity_sums + 1e-9)
    return predicted_ratings

# -----------------------------
# 5. Evaluasi Model
# -----------------------------
def evaluate_model(matrix_T, similarity_df):
    actual, predicted = [], []
    for user in matrix_T.columns:
        user_ratings = matrix_T[user]
        pred_ratings = predict_ratings_item_based(user, matrix_T, similarity_df)
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
st.title("📦 Item-Based Collaborative Filtering")

# Load dan proses data
df = load_data()
item_user_matrix = create_item_user_matrix(df)
item_similarity_df = compute_item_similarity(item_user_matrix)

# Transpose untuk akses per user (userId jadi kolom)
matrix_T = item_user_matrix

# Pilih user
user_ids = matrix_T.columns.tolist()
selected_user = st.selectbox("Pilih User ID", user_ids)

# Prediksi rating untuk user
predicted_ratings = predict_ratings_item_based(selected_user, matrix_T, item_similarity_df)
rated_items = matrix_T[selected_user]
unrated_predictions = predicted_ratings.drop(rated_items[rated_items > 0].index)

# Debug info
nonzero = np.count_nonzero(matrix_T)
total = matrix_T.size
sparsity = 1.0 - (nonzero / total)

st.markdown("### ℹ️ Debugging Info")
st.write("Rating oleh User", selected_user, ":", rated_items[rated_items > 0])
st.write("Jumlah item yang tidak dirating:", len(unrated_predictions))
st.write(f"Sparsity Matrix: {sparsity:.2%}")

# Tampilkan rekomendasi
top_n = st.slider("Jumlah Rekomendasi", 1, 10, 5)
top_recommendations = unrated_predictions.sort_values(ascending=False).head(top_n)

st.subheader(f"🎯 Rekomendasi untuk User {selected_user}")
st.dataframe(top_recommendations.reset_index().rename(columns={selected_user: "Final Score Prediction"}))

# Evaluasi model
st.subheader("📊 Evaluasi Model")
with st.spinner("Menghitung metrik evaluasi..."):
    mae, mse, rmse, r2 = evaluate_model(matrix_T, item_similarity_df)

st.metric("MAE", f"{mae:.4f}")
st.metric("MSE", f"{mse:.4f}")
st.metric("RMSE", f"{rmse:.4f}")
st.metric("R²", f"{r2:.4f}")
