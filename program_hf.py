import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import time
import sys
import os
tqdm.pandas()

width_title = 80
start_time = time.time()
print("|" * width_title)
print("".center(width_title))
print("HANDCRAFTED FEATURES".center(width_title))
print("".center(width_title))
print("|" * width_title)

if len(sys.argv) < 2:
    print("Dataset argument missing. Please run from main.py")
    sys.exit(1)

dataset_choice = sys.argv[1]

# Konfigurasi direktori dan nama file
dataset_config = {
    "dummy": {
        "dir": "dataset_dummy/",
        "file_special": "b_ffnn_ratings_x.csv",
        "file_default": "b_ffnn_ratings.csv"
    },
    "movie": {
        "dir": "dataset_movielens/",
        "file_special": "b_ffnn_ratings_x6.csv",
        "file_default": "b_ffnn_ratings.csv"
    },
    "hotel": {
        "dir": "dataset_hotels/",
        "file_special": "b_ffnn_ratings_x6.csv",
        "file_default": "b_ffnn_ratings.csv"
    }
}

# Validasi dataset
if dataset_choice not in dataset_config:
    print("Unknown dataset.")
    sys.exit(1)

# Ambil konfigurasi
config = dataset_config[dataset_choice]
file_dir = config["dir"]
file_special_path = os.path.join(file_dir, config["file_special"])
file_default_path = os.path.join(file_dir, config["file_default"])

# Pilih nama file yang akan digunakan (bukan path lengkap)
if os.path.exists(file_special_path):
    input_path = config["file_special"]
else:
    input_path = config["file_default"]


def compute_features(file_dir, input_path, output_path, rating_column="actual_rating"):
    start_time = time.time()

    input_file = file_dir + input_path
    output_file = file_dir + output_path

    print("🔃 Membaca dataset...")
    df = pd.read_csv(input_file)
    filtered_df = df[df[rating_column] > 0]

    print("📊 Statistik dasar...")
    total_ratings = len(filtered_df)
    total_users = df['userId'].nunique()
    total_items = df['itemId'].nunique()
    print(f"Total rating : {total_ratings}")
    print(f"Total user   : {total_users}")
    print(f"Total item   : {total_items}")

    print("🧠 Menghitung fitur rata-rata...")

    # Global Average Rating
    df['global_average_rating'] = round(filtered_df[rating_column].mean(), 1)

    user_avg = filtered_df.groupby('userId')[rating_column].mean().round(1)
    item_avg = filtered_df.groupby('itemId')[rating_column].mean().round(1)

    # User Average Rating
    df['user_average_rating'] = df['userId'].map(user_avg).fillna(0.0)

    # Item Average Rating
    df['item_average_rating'] = df['itemId'].map(item_avg).fillna(0.0)

    print("🤖 Membangun user-item sparse matrix...")
    # Matrix user-item dan cosine similarity antar user
    user_item_pivot = filtered_df.pivot(index='userId', columns='itemId', values=rating_column).fillna(0)
    user_item_matrix = csr_matrix(user_item_pivot.values)
    user_sim = cosine_similarity(user_item_matrix)

    print("🤖 Membangun item-user sparse matrix...")
    # Matrix item-user dan cosine similarity antar item
    item_user_pivot = filtered_df.pivot(index='itemId', columns='userId', values=rating_column).fillna(0)
    item_user_matrix = csr_matrix(item_user_pivot.values)
    item_sim = cosine_similarity(item_user_matrix)

    # Map userId dan itemId ke index
    user_index_map = {uid: idx for idx, uid in enumerate(user_item_pivot.index)}
    item_index_map = {iid: idx for idx, iid in enumerate(item_user_pivot.index)}

    # Similar Users Rating
    def predict_sim_user(row):
        user_id, item_id = row['userId'], row['itemId']
        if item_id not in user_item_pivot.columns or user_id not in user_index_map:
            return 0.0
        ratings = user_item_pivot[item_id]
        similarities = user_sim[user_index_map[user_id]]
        mask = ratings > 0
        sim_scores = similarities[mask.values]
        if sim_scores.sum() == 0:
            return 0.0
        return round(np.dot(sim_scores, ratings[mask]) / sim_scores.sum(), 1)

    # Similar Items Ratings
    def predict_sim_item(row):
        user_id, item_id = row['userId'], row['itemId']
        if user_id not in item_user_pivot.columns or item_id not in item_index_map:
            return 0.0
        ratings = item_user_pivot[user_id]
        similarities = item_sim[item_index_map[item_id]]
        mask = ratings > 0
        sim_scores = similarities[mask.values]
        if sim_scores.sum() == 0:
            return 0.0
        return round(np.dot(sim_scores, ratings[mask]) / sim_scores.sum(), 1)

    print("📐 Menghitung prediksi berbasis user dan item serupa...")
    df['similar_users_rating'] = df[['userId', 'itemId']].progress_apply(predict_sim_user, axis=1)
    df['similar_items_rating'] = df[['userId', 'itemId']].progress_apply(predict_sim_item, axis=1)

    print("💾 Menyimpan hasil akhir...")
    df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time
    print(f"\n✅ Fitur handcrafted disimpan ke: {output_path}")
    print(f"Total waktu eksekusi : {elapsed:.2f} detik.")

compute_features(
    file_dir=file_dir,
    input_path=input_path,
    output_path="c_hf_ratings.csv",
    rating_column='ffnn_predicted_rating'
)
