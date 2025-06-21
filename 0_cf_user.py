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
# 2. Buat User-Item Matrix
# -----------------------------
def create_user_item_matrix(df):
    matrix = df.pivot_table(index="userId", columns="itemId", values="rating")
    return matrix.fillna(0)

# -----------------------------
# 3. Hitung Kemiripan Antar User
# -----------------------------
def compute_user_similarity(matrix):
    similarity = cosine_similarity(matrix)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

# -----------------------------
# 4. Prediksi Rating untuk User
# -----------------------------
def predict_ratings_user_based(user_id, matrix, similarity_df):
    if user_id not in similarity_df.index:
        return pd.Series([], dtype=np.float64)

    user_sim = similarity_df.loc[user_id]

    weighted_sum = user_sim @ matrix
    sim_sum = user_sim @ (matrix > 0).astype(int)
    pred_ratings = weighted_sum / (sim_sum + 1e-9)

    return pd.Series(pred_ratings, index=matrix.columns)

# -----------------------------
# 5. Evaluasi Model
# -----------------------------
def evaluate_model(matrix, similarity_df):
    actual, predicted = [], []
    for user in matrix.index:
        true_ratings = matrix.loc[user]
        pred_ratings = predict_ratings_user_based(user, matrix, similarity_df)
        for item in true_ratings[true_ratings > 0].index:
            actual.append(true_ratings[item])
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

# Load dan proses data
df = load_data()
user_item_matrix = create_user_item_matrix(df)
user_similarity_df = compute_user_similarity(user_item_matrix)

# Pilih user
user_ids = user_item_matrix.index.tolist()
selected_user = st.selectbox("Pilih User ID", user_ids)

# Prediksi untuk user terpilih
predicted_ratings = predict_ratings_user_based(selected_user, user_item_matrix, user_similarity_df)
rated_items = user_item_matrix.loc[selected_user]
unrated_predictions = predicted_ratings.drop(rated_items[rated_items > 0].index)

# Debug info: Sparsity dan interaksi user
nonzero = np.count_nonzero(user_item_matrix)
total = user_item_matrix.size
sparsity = 1.0 - (nonzero / total)

st.markdown("### ℹ️ Debugging Info")
st.write("Rating oleh User", selected_user, ":", rated_items[rated_items > 0])
st.write("Similarity terhadap user lain:", user_similarity_df.loc[selected_user])
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
    mae, mse, rmse, r2 = evaluate_model(user_item_matrix, user_similarity_df)

st.metric("MAE", f"{mae:.4f}")
st.metric("MSE", f"{mse:.4f}")
st.metric("RMSE", f"{rmse:.4f}")
st.metric("R²", f"{r2:.4f}")
