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
    df = pd.read_csv(
        "dataset_hotels/ratings.csv",
        names=["userId", "itemId", "rating"]
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    # Kalau ada duplikat user-item, ambil rating maksimum
    df = df.groupby(["userId", "itemId"], as_index=False)["rating"].max()
    return df


# ======================================
# 2. Train-Test Split per User
# ======================================
def train_test_split_per_user(
    df, test_size=0.2, min_ratings=2, random_state=42
):
    """
    Bagi data per user:
    - Jika jumlah rating user < min_ratings -> semua ke train
    - Kalau cukup, ambil sebagian (test_size) ke test
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
# 3. Buat User-Item Matrix
# ======================================
def create_user_item_matrix(df):
    """
    Matriks userId x itemId
    """
    matrix = df.pivot_table(
        index="userId",
        columns="itemId",
        values="rating",
        aggfunc="mean",
    )
    return matrix.fillna(0.0)


# ======================================
# 4. Hitung Kemiripan Antar User
# ======================================
def compute_user_similarity(matrix):
    similarity = cosine_similarity(matrix)
    # Diagonal diset 0 supaya self-similarity tidak mendominasi
    np.fill_diagonal(similarity, 0.0)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)


# ======================================
# 5. Prediksi Rating untuk User (User-Based CF)
# ======================================
def predict_ratings_user_based(user_id, matrix, similarity_df):
    """
    matrix       : userId x itemId
    similarity_df: userId x userId
    """
    if user_id not in similarity_df.index:
        return pd.Series(dtype=np.float64)

    user_sim = similarity_df.loc[user_id]  # vektor kemiripan user ini ke user lain

    # Weighted sum (similarity * rating)
    weighted_sum = user_sim @ matrix

    # Jumlah bobot similarity untuk setiap item (hanya user yang punya rating di item tsb)
    sim_sum = user_sim @ (matrix > 0).astype(int)

    # Hindari pembagian 0
    sim_sum = sim_sum.replace(0, np.nan)

    pred_ratings = weighted_sum / sim_sum
    return pd.Series(pred_ratings, index=matrix.columns)


# ======================================
# 6. Evaluasi Model (Train-Test)
# ======================================
def evaluate_model(train_matrix, similarity_df, test_df, rating_threshold=4.0):
    actual = []
    predicted = []

    # Evaluasi berdasarkan baris di data test (user, item, rating)
    for _, row in test_df.iterrows():
        user = row["userId"]
        item = row["itemId"]
        true_rating = row["rating"]

        # Pastikan user & item ada di data train
        if user not in train_matrix.index:
            continue
        if item not in train_matrix.columns:
            continue

        user_predictions = predict_ratings_user_based(
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

    # --- Metrik prediksi rating ---
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    # --- Konversi ke kelas 0/1 untuk metrik klasifikasi ---
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
st.title("👥 User-Based Collaborative Filtering (Dengan Evaluasi Train-Test)")

# Load data
df = load_data()
st.write("Jumlah baris data rating:", len(df))

# Sidebar pengaturan evaluasi
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

# Matriks & similarity dari data train
user_item_matrix_train = create_user_item_matrix(train_df)
user_similarity_df = compute_user_similarity(user_item_matrix_train)

# Sparsity info
nonzero = np.count_nonzero(user_item_matrix_train.values)
total = user_item_matrix_train.size
sparsity = 1.0 - (nonzero / total)

st.markdown("### ℹ️ Info Matrix (Train)")
st.write(f"Sparsity Matrix: **{sparsity:.2%}**")

# ======================================
# Rekomendasi untuk User Tertentu
# ======================================
st.markdown("---")
st.subheader("🎯 Rekomendasi untuk User Tertentu (berdasarkan data train)")

user_ids = user_item_matrix_train.index.tolist()
selected_user = st.selectbox("Pilih User ID", user_ids)

predicted_ratings = predict_ratings_user_based(
    selected_user, user_item_matrix_train, user_similarity_df
)

rated_items = user_item_matrix_train.loc[selected_user]
rated_items_nonzero = rated_items[rated_items > 0]

# Hanya rekomendasikan item yang belum pernah dirating user
unrated_predictions = predicted_ratings.drop(
    rated_items_nonzero.index,
    errors="ignore"
)

st.markdown("#### Rating yang sudah pernah diberikan (di data train)")
st.write(rated_items_nonzero)

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

st.markdown("#### Similarity User Terhadap User Lain")
st.write(user_similarity_df.loc[selected_user])

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
        user_item_matrix_train,
        user_similarity_df,
        test_df,
        rating_threshold=rating_threshold,
    )

if eval_result is None:
    st.warning(
        "Tidak ada sampel test yang bisa dievaluasi. "
        "Coba turunkan 'Minimal jumlah rating per user' atau ubah 'test_size'."
    )
else:
    (
        mae,
        mse,
        rmse,
        r2,
        precision,
        recall,
        f1,
        accuracy,
        n_samples,
    ) = eval_result

    st.write(
        f"Jumlah pasangan user–item yang digunakan untuk evaluasi: **{n_samples}**"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
    with c2:
        st.metric("MSE", f"{mse:.4f}")
        st.metric("R²", f"{r2:.4f}")
    with c3:
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
    with c4:
        st.metric("F1-Score", f"{f1:.4f}")
        st.metric("Accuracy", f"{accuracy:.4f}")
