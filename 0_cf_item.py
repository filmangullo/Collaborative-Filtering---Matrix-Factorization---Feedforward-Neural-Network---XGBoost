import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("dataset_dummy/ratings.csv", names=["userId", "itemId", "rating"])

    # Pastikan rating berupa float
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Ambil hanya rating tertinggi jika user memberi lebih dari satu rating ke item yang sama
    df = df.groupby(['userId', 'itemId'], as_index=False)['rating'].max()

    return df

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
    
    if user_ratings.sum() == 0:
        # User belum punya rating sama sekali
        return pd.Series(0, index=matrix.index)

    # Hitung prediksi
    weighted_sum = sim_df.dot(user_ratings)
    sim_sum = sim_df.dot((user_ratings > 0).astype(int))
    
    # Hindari divide-by-zero dan hasil mendekati 0
    pred_ratings = weighted_sum / (sim_sum + 1e-9)
    pred_ratings = pred_ratings.replace([np.inf, -np.inf], 0).fillna(0)
    
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
    r2 = r2_score(actual, predicted)  # tambahan ini
    return mae, mse, rmse, r2

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🍿 Item-Based Collaborative Filtering")

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
top_recommendations_df = top_recommendations.reset_index().rename(columns={selected_user: "Predicted Rating"})
top_recommendations_df.columns = top_recommendations_df.columns.astype(str)
st.dataframe(top_recommendations_df)




# Evaluasi model
st.subheader("📊 Evaluasi Model (Seluruh Data)")
with st.spinner("Menghitung evaluasi..."):
    mae, mse, rmse, r2 = evaluate_model(item_user_matrix, item_similarity_df)
    st.metric("MAE", f"{mae:.4f}")
    st.metric("MSE", f"{mse:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("R2", f"{r2:.4f}")
