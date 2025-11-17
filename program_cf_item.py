import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# ======================================
# 1. Load Dataset
# ======================================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_movielens/ratings.csv", names=["userId", "itemId", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    # Jika ada duplikat user-item, ambil rating maksimum
    df = df.groupby(["userId", "itemId"], as_index=False)["rating"].max()
    return df


# ======================================
# 2. Train-Test Split per User
#    (untuk hindari data leakage)
# ======================================
def train_test_split_per_user(
    df, test_size=0.2, min_ratings=2, random_state=42
):
    """
    Bagi data per user:
    - kalau rating user < min_ratings -> semua ke train
    - kalau cukup, ambil sebagian (test_size) ke test
    """
    np.random.seed(random_state)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  # shuffle

    train_rows = []
    test_rows = []

    for user, group in df.groupby("userId"):
        if len(group) < min_ratings:
            train_rows.append(group)
            continue

        n_test = max(1, int(len(group) * test_size))
        test_group = group.sample(n=n_test, random_state=random_state)
        train_group = group.drop(test_group.index)

        train_rows.append(train_group)
        test_rows.append(test_group)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    return train_df, test_df


# ======================================
# 3. Buat Item-User Matrix
# ======================================
def create_item_user_matrix(df):
    """
    Menghasilkan matriks itemId x userId
    """
    matrix = df.pivot_table(
        index="itemId", columns="userId", values="rating", aggfunc="mean"
    )
    return matrix.fillna(0.0)


# ======================================
# 4. Hitung Kemiripan Antar Item
# ======================================
def compute_item_similarity(matrix):
    similarity = cosine_similarity(matrix)
    # Hilangkan self-similarity agar tidak bias (diag = 0)
    np.fill_diagonal(similarity, 0.0)
    similarity_df = pd.DataFrame(
        similarity, index=matrix.index, columns=matrix.index
    )
    return similarity_df


# ======================================
# 5. Prediksi Rating untuk User (Item-Based)
# ======================================
def predict_ratings_item_based(user_id, rating_matrix, similarity_df):
    """
    rating_matrix: itemId x userId
    similarity_df: itemId x itemId
    """
    if user_id not in rating_matrix.columns:
        return pd.Series(dtype=np.float64)

    user_ratings = rating_matrix[user_id]
    rated_items = user_ratings[user_ratings > 0]

    if rated_items.empty:
        return pd.Series(dtype=np.float64)

    # Ambil similarity hanya untuk item yang sudah dirating user
    sim_subset = similarity_df[rated_items.index]

    # Weighted sum of ratings
    weighted_scores = sim_subset @ rated_items

    # Jumlah absolut similarity
    similarity_sums = sim_subset.abs().sum(axis=1)

    # Hindari pembagian dengan 0
    similarity_sums = similarity_sums.replace(0, np.nan)

    predicted_ratings = weighted_scores / similarity_sums
    return predicted_ratings


# ======================================
# 6. Evaluasi Model (Train-Test)
# ======================================
def evaluate_model(train_matrix, similarity_df, test_df, rating_threshold=4.0):
    actual = []
    predicted = []

    # Loop setiap baris di data uji (user, item, rating)
    for _, row in test_df.iterrows():
        user = row["userId"]
        item = row["itemId"]
        true_rating = row["rating"]

        # Pastikan user & item ada di data train
        if user not in train_matrix.columns:
            continue
        if item not in train_matrix.index:
            continue

        user_predictions = predict_ratings_item_based(
            user, train_matrix, similarity_df
        )

        if item not in user_predictions.index:
            continue

        pred_rating = user_predictions[item]
        if pd.isna(pred_rating):
            continue

        actual.append(true_rating)
        predicted.append(pred_rating)

    if len(actual) == 0:
        return None  # tidak ada sampel yang bisa dievaluasi

    # --- Metrik rating prediction ---
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    # --- Konversi ke kelas 0/1 untuk Precision, Recall, F1, Accuracy ---
    actual_binary = [1 if r >= rating_threshold else 0 for r in actual]
    predicted_binary = [1 if r >= rating_threshold else 0 for r in predicted]

    precision = precision_score(actual_binary, predicted_binary, zero_division=0)
    recall = recall_score(actual_binary, predicted_binary, zero_division=0)
    f1 = f1_score(actual_binary, predicted_binary, zero_division=0)
    accuracy = accuracy_score(actual_binary, predicted_binary)

    return mae, mse, rmse, r2, precision, recall, f1, accuracy, len(actual)


# ======================================
# STREAMLIT UI
# ======================================
st.title("📦 Item-Based Collaborative Filtering (Dengan Evaluasi Train-Test)")

# Load data
df = load_data()
st.write("Jumlah baris data rating:", len(df))

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Pengaturan Evaluasi")
test_size = st.sidebar.slider(
    "Proporsi data test per user",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
)
min_ratings = st.sidebar.slider(
    "Minimal jumlah rating per user untuk di-split",
    min_value=2,
    max_value=10,
    value=2,
    step=1,
)

# Split train-test
train_df, test_df = train_test_split_per_user(
    df, test_size=test_size, min_ratings=min_ratings
)

st.markdown("### 📂 Ringkasan Data")
col1, col2 = st.columns(2)
with col1:
    st.write("• Jumlah rating (train):", len(train_df))
    st.write("• Jumlah user (train):", train_df["userId"].nunique())
    st.write("• Jumlah item (train):", train_df["itemId"].nunique())
with col2:
    st.write("• Jumlah rating (test):", len(test_df))
    st.write("• Jumlah user (test):", test_df["userId"].nunique())
    st.write("• Jumlah item (test):", test_df["itemId"].nunique())

# Buat matriks & similarity
item_user_matrix_train = create_item_user_matrix(train_df)
item_similarity_df = compute_item_similarity(item_user_matrix_train)

# Sparsity info
nonzero = np.count_nonzero(item_user_matrix_train.values)
total = item_user_matrix_train.size
sparsity = 1.0 - (nonzero / total)

st.markdown("### ℹ️ Info Matrix (Train)")
st.write(f"Sparsity Matrix: **{sparsity:.2%}**")

# ======================================
# Rekomendasi untuk User
# ======================================
st.markdown("---")
st.subheader("🎯 Rekomendasi untuk User Tertentu (berdasarkan data train)")

user_ids = item_user_matrix_train.columns.tolist()
selected_user = st.selectbox("Pilih User ID", user_ids)

predicted_ratings = predict_ratings_item_based(
    selected_user, item_user_matrix_train, item_similarity_df
)

user_ratings_train = item_user_matrix_train[selected_user]
rated_items = user_ratings_train[user_ratings_train > 0]

# Hilangkan item yang sudah dirating user (hanya rekomendasi item baru)
unrated_predictions = predicted_ratings.drop(rated_items.index, errors="ignore")

st.markdown("#### Rating yang sudah pernah diberikan (di data train)")
st.write(rated_items)

top_n = st.slider("Jumlah Rekomendasi yang Ditampilkan", 1, 10, 5)

top_recommendations = (
    unrated_predictions.rename("Final Score Prediction")
    .sort_values(ascending=False)
    .head(top_n)
)

st.markdown(f"#### 🔝 Top-{top_n} Rekomendasi untuk User {selected_user}")
st.dataframe(
    top_recommendations.reset_index().rename(
        columns={"index": "itemId"}
    )
)

# ======================================
# Evaluasi Model
# ======================================
st.markdown("---")
st.subheader("📊 Evaluasi Model (berdasarkan data test)")

rating_threshold = st.slider(
    "Threshold rating untuk dianggap 'positif' (relevan)",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.5,
)

with st.spinner("Menghitung metrik evaluasi..."):
    eval_result = evaluate_model(
        item_user_matrix_train,
        item_similarity_df,
        test_df,
        rating_threshold=rating_threshold,
    )

if eval_result is None:
    st.warning(
        "Tidak ada sampel test yang bisa dievaluasi. "
        "Coba turunkan 'Minimal jumlah rating per user' atau ubah 'test_size'."
    )
else:
    mae, mse, rmse, r2, precision, recall, f1, accuracy, n_samples = eval_result

    st.write(f"Jumlah pasangan user–item yang dipakai evaluasi: **{n_samples}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
    with col2:
        st.metric("MSE", f"{mse:.4f}")
        st.metric("R²", f"{r2:.4f}")
    with col3:
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
        st.metric("Accuracy", f"{accuracy:.4f}")
