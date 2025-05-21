import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

def compute_features(input_path, output_path, rating_column="actual_rating"):
    start_time = time.time()

    # 1. Membaca file
    df = pd.read_csv(input_path)

    # 2. Filter rating valid (>0)
    filtered_df = df[df[rating_column] > 0]

    # 3. Statistik dasar
    total_ratings = len(filtered_df)
    total_users = df['userId'].nunique()
    total_items = df['itemId'].nunique()

    print(f"Total rating : {total_ratings}")
    print(f"Total user   : {total_users}")
    print(f"Total item   : {total_items}")

    # 4. Global average rating
    global_avg = filtered_df[rating_column].mean()
    df['global_average_rating'] = round(global_avg, 1)

    # 5. User average rating
    user_avg = filtered_df.groupby('userId')[rating_column].mean().reset_index()
    user_avg.columns = ['userId', 'user_average_rating']
    df = df.merge(user_avg, on='userId', how='left')
    df['user_average_rating'] = df['user_average_rating'].round(1).fillna(0.0)

    # 6. Item average rating
    item_avg = filtered_df.groupby('itemId')[rating_column].mean().reset_index()
    item_avg.columns = ['itemId', 'item_average_rating']
    df = df.merge(item_avg, on='itemId', how='left')
    df['item_average_rating'] = df['item_average_rating'].round(1).fillna(0.0)

    # 7. Matrix user-item dan cosine similarity antar user
    user_item_matrix = filtered_df.pivot_table(index='userId', columns='itemId', values=rating_column, fill_value=0)
    user_sim_df = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)

    # 8. Matrix item-user dan cosine similarity antar item
    item_user_matrix = filtered_df.pivot_table(index='itemId', columns='userId', values=rating_column, fill_value=0)
    item_sim_df = pd.DataFrame(cosine_similarity(item_user_matrix), index=item_user_matrix.index, columns=item_user_matrix.index)

    # 9. Vektorisasi prediksi rating user serupa dan item serupa
    def predict_similar_user(user_id, item_id):
        if item_id not in user_item_matrix.columns or user_id not in user_sim_df.index:
            return 0.0
        ratings = user_item_matrix[item_id]
        similarities = user_sim_df[user_id]
        mask = ratings > 0
        sim_scores = similarities[mask]
        if sim_scores.sum() == 0:
            return 0.0
        return round(np.dot(sim_scores, ratings[mask]) / sim_scores.sum(), 1)

    def predict_similar_item(user_id, item_id):
        if user_id not in item_user_matrix.columns or item_id not in item_sim_df.index:
            return 0.0
        ratings = item_user_matrix[user_id]
        similarities = item_sim_df[item_id]
        mask = ratings > 0
        sim_scores = similarities[mask]
        if sim_scores.sum() == 0:
            return 0.0
        return round(np.dot(sim_scores, ratings[mask]) / sim_scores.sum(), 1)

    # Gunakan vektorisasi dengan apply
    df['similar_users_rating'] = df.apply(lambda row: predict_similar_user(row['userId'], row['itemId']), axis=1)
    df['similar_items_rating'] = df.apply(lambda row: predict_similar_item(row['userId'], row['itemId']), axis=1)

    # 10. Simpan hasil
    df.to_csv(output_path, index=False)

    # 11. Waktu eksekusi
    elapsed = time.time() - start_time
    print(f"\n📁 Hasil handcrafted features disimpan ke: {output_path}")
    print(f"Total waktu eksekusi : {elapsed:.2f} detik.")

# Pemanggilan fungsi dinamis
compute_features(
    input_path='dataset_hotels/with_1_percent_data/b_ffnn_ratings.csv',
    output_path='c_hf_ratings.csv',
    rating_column='ffnn_predicted_rating'  # Bisa juga 'ffnn_predicted_rating'
)