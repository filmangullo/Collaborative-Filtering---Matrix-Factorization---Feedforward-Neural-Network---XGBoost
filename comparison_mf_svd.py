import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

# ----------------------------
# 0. Program Title
# ----------------------------
print(f"----------------------------------------------------------------")
print(f"   Matrix Factorization via Singular Value Decomposition (SVD)   ")
print(f"----------------------------------------------------------------")

# ----------------------------
# 1. Load dan split dataset
# ----------------------------
file_dir = "dataset_dummy/"  # sesuaikan folder

# Load dataset ratings (kolom: userId, itemId, rating)
ratings = pd.read_csv(file_dir + "ratings.csv")

# Membagi data menjadi data latih dan data uji
train_data, test_data = train_test_split(ratings, test_size=0.1, random_state=42)

# Menghitung total jumlah data
total_data = len(ratings)

# Menghitung persentase data latih dan data uji
persentase_train = (len(train_data) / total_data) * 100
persentase_test = (len(test_data) / total_data) * 100

print(f"Train Data Presentation: {persentase_train:.2f}%")
print(f"Test Data Presentation : {persentase_test:.2f}%")
print()

# ----------------------------
# 2. Buat user–item matrix dari training data
# ----------------------------
R_df = train_data.pivot_table(
    index="userId",
    columns="itemId",
    values="rating",
    aggfunc="mean",
)

# Simpan mapping user dan item
user_ids = R_df.index.tolist()
item_ids = R_df.columns.tolist()

# Isi missing value (NaN) dengan 0
R = R_df.fillna(0).values

num_users, num_items = R.shape

# ----------------------------
# 3. Hyperparameter SVD (jumlah komponen laten)
# ----------------------------
k = 64  # jumlah komponen utama / latent factors

print("Hyperparameter Matrix Factorization (SVD):")
print(f"Latent factors / Dimensi laten: {k}")
print(f"Jumlah user                    : {num_users}")
print(f"Jumlah item                    : {num_items}")
print()

# ----------------------------
# 4. Dekomposisi SVD
# ----------------------------
# SVD penuh: R = U_full * S_full * Vt_full
# U_full: (num_users, num_users)
# S_full: (min(num_users, num_items),)
# Vt_full: (num_items, num_items)
print("Melakukan dekomposisi SVD...")

start_svd = time.time()
U_full, S_full, Vt_full = np.linalg.svd(R, full_matrices=False)
end_svd = time.time()

print(f"Selesai SVD dalam {end_svd - start_svd:.2f} detik.")
print()

# Ambil hanya k komponen teratas
U_k = U_full[:, :k]                    # (num_users, k)
S_k = np.diag(S_full[:k])              # (k, k)
Vt_k = Vt_full[:k, :]                  # (k, num_items)

# ----------------------------
# 5. Rekonstruksi matriks prediksi
# ----------------------------
# R_pred ≈ U_k * S_k * Vt_k
R_pred = np.dot(np.dot(U_k, S_k), Vt_k)

# Batasin ke range rating (misal 1–5)
R_pred = np.clip(R_pred, 1.0, 5.0)

# ----------------------------
# 6. Fungsi evaluasi metrik di TRAIN (opsional)
# ----------------------------
def get_metrics(R_true, R_hat):
    xs, ys = np.where(R_true != 0)  # hanya yang ada rating aslinya
    y_true = R_true[xs, ys]
    y_pred = R_hat[xs, ys]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

train_mae, train_mse, train_rmse = get_metrics(R, R_pred)
print("Evaluasi di data TRAIN (hanya entry yang ada rating-nya):")
print(f"MAE  (train): {train_mae:.4f}")
print(f"MSE  (train): {train_mse:.4f}")
print(f"RMSE (train): {train_rmse:.4f}")
print()

# ----------------------------
# 7. Evaluasi pada data TEST
# ----------------------------
def evaluate_on_test(test_df, user_ids, item_ids, R_pred, threshold=4.0):
    """
    Evaluasi pada data test:
    - MAE, MSE, RMSE (continuous)
    - Precision & Recall (biner; rating >= threshold dianggap relevan)
    """
    y_true, y_pred = [], []

    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    for row in test_df.itertuples():
        u_id, i_id, r = row.userId, row.itemId, row.rating

        if u_id in user_map and i_id in item_map:
            u_idx = user_map[u_id]
            i_idx = item_map[i_id]

            pred = R_pred[u_idx, i_idx]
            pred = np.clip(pred, 1.0, 5.0)

            y_true.append(r)
            y_pred.append(pred)

    if len(y_true) == 0:
        print("⚠️ Tidak ada data test yang cocok dengan user/item di matriks training.")
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    # Regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Binarisasi relevan vs tidak relevan
    actual_bin = [1 if r >= threshold else 0 for r in y_true]
    pred_bin   = [1 if p >= threshold else 0 for p in y_pred]

    precision = precision_score(actual_bin, pred_bin, zero_division=0)
    recall    = recall_score(actual_bin, pred_bin, zero_division=0)

    return mae, mse, rmse, precision, recall

test_mae, test_mse, test_rmse, test_precision, test_recall = evaluate_on_test(
    test_data,
    user_ids,
    item_ids,
    R_pred,
    threshold=3.0,  # rating >= 4 dianggap "suka"
)

print("Evaluasi pada Data Test:")
print(f"MAE       : {test_mae:.4f}")
print(f"MSE       : {test_mse:.4f}")
print(f"RMSE      : {test_rmse:.4f}")
print(f"Precision : {test_precision:.4f} (rating >= 3)")
print(f"Recall    : {test_recall:.4f} (rating >= 3)")
print()

# ----------------------------
# 8. Generate semua prediksi user–item
# ----------------------------
predictions = []

for user_index in range(num_users):
    for item_index in range(num_items):
        predictions.append({
            "userId": user_ids[user_index],
            "itemId": item_ids[item_index],
            "mf_svd_predicted_rating": round(R_pred[user_index][item_index], 2),
        })

predictions_df = pd.DataFrame(predictions)

print("Total baris prediksi:", len(predictions_df))
print(f"Total user : {num_users}")
print(f"Total item : {num_items}")

# ----------------------------
# 9. Simpan ke CSV
# ----------------------------
# start_time_save = time.time()

# output_path = file_dir + "a_mf_svd_ratings.csv"
# predictions_df.to_csv(output_path, index=False)

# end_time_save = time.time()
# elapsed_time_save = end_time_save - start_time_save

# print(f"\n📁 Hasil prediksi (SVD) disimpan ke: {output_path}")
# print(f"Total waktu simpan: {elapsed_time_save:.2f} detik.")
