import pandas as pd
import numpy as np
from tqdm import tqdm
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
print(f"   Matrix Factorization via Stochastic Gradient Descent (SGD)   ")
print(f"----------------------------------------------------------------")

# ----------------------------
# 1. Load dan split dataset
# ----------------------------
file_dir = "dataset_dummy/"

# Load dataset ratings
ratings = pd.read_csv(file_dir + "ratings.csv")  # pastikan file ratings.csv ada di direktori yang sama

# Membagi data menjadi data latih dan data uji
train_data, test_data = train_test_split(ratings, test_size=0.1, random_state=42)

# Menghitung total jumlah data
total_data = len(ratings)

# Menghitung persentase data latih dan data uji
persentase_train = (len(train_data) / total_data) * 100
persentase_test = (len(test_data) / total_data) * 100

# Mencetak hasil
print(f"Train Data Presentation: {persentase_train:.2f}%")
print(f"Test Data Presentation : {persentase_test:.2f}%")
print(f"\n")

# ----------------------------
# 2. Buat user-item matrix dari training data
# ----------------------------
R_df = train_data.pivot_table(
    index="userId",
    columns="itemId",
    values="rating",
    aggfunc="mean",
)

R = R_df.fillna(0).values  # Mengisi NaN dengan 0
user_ids = R_df.index.tolist()
item_ids = R_df.columns.tolist()

# Hyperparameter Matrix Factorization
num_users, num_items = R.shape
k = 64       # latent factors
alpha = 0.04 # learning rate
beta = 0.04  # regularization parameter
epochs = 45  # jumlah epoch

print("Hyperparameter Matrix Factorization:")
print(f"Latent factors / Dimensi laten: {k}")
print(f"Learning rate                 : {alpha}")
print(f"Regularization parameter      : {beta}")
print(f"Jumlah epoch / training       : {epochs}")
print(f"\n")

# ----------------------------
# 3. Inisialisasi latent factor U dan V
# ----------------------------
np.random.seed(42)
U = np.random.normal(scale=1.0 / k, size=(num_users, k))
V = np.random.normal(scale=1.0 / k, size=(num_items, k))

# ----------------------------
# 4. Fungsi evaluasi metrik (di TRAIN matrix)
# ----------------------------
def get_metrics(R, U, V):
    xs, ys = np.where(R != 0)  # Hanya hitung yang bukan 0
    y_true = []
    y_pred = []

    for x, y in zip(xs, ys):
        y_true.append(R[x, y])
        y_pred.append(np.dot(U[x], V[y]))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# ----------------------------
# 5. Fungsi training MF (SGD)
# ----------------------------
def train_mf(R, U, V, alpha, beta, epochs):
    num_users, num_items = R.shape

    for epoch in range(epochs):
        # Bisa diacak jika mau: indices = np.random.permutation(num_users * num_items)
        for i in range(num_users):
            for j in range(num_items):
                if R[i][j] > 0:  # Hanya update jika ada rating
                    prediction = np.dot(U[i, :], V[j, :].T)
                    eij = R[i][j] - prediction

                    # Update rule SGD
                    U[i, :] += alpha * (eij * V[j, :] - beta * U[i, :])
                    V[j, :] += alpha * (eij * U[i, :] - beta * V[j, :])

                    # Optional: batasi nilai untuk stabilitas
                    U[i, :] = np.clip(U[i, :], -5, 5)
                    V[j, :] = np.clip(V[j, :], -5, 5)

        # Cek NaN
        if np.isnan(U).any() or np.isnan(V).any():
            print(f"❌ NaN detected at epoch {epoch + 1} — menghentikan training.")
            break

        mae, mse, rmse = get_metrics(R, U, V)
        print(f"Epoch {epoch + 1}/{epochs}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    return U, V

# ----------------------------
# 6. Training
# ----------------------------
U, V = train_mf(R, U, V, alpha, beta, epochs)

# Prediksi penuh pada matrix train
R_pred = U.dot(V.T)
# Pastikan prediksi di rentang 1–5
R_pred = np.clip(R_pred, 1.0, 5.0)

# ----------------------------
# 7. Evaluasi pada data test (MAE, MSE, RMSE, Precision, Recall)
# ----------------------------
print("\n")

def evaluate_on_test(test_df, user_ids, item_ids, U, V, threshold=4.0):
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

        # Hanya evaluasi jika user & item muncul di matriks training
        if u_id in user_map and i_id in item_map:
            u_idx = user_map[u_id]
            i_idx = item_map[i_id]

            pred = np.dot(U[u_idx], V[i_idx])
            # Jaga-jaga: clip ke [1, 5]
            pred = np.clip(pred, 1.0, 5.0)

            y_true.append(r)
            y_pred.append(pred)

    if len(y_true) == 0 or len(y_pred) == 0:
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

# Panggil evaluasi di test set
test_mae, test_mse, test_rmse, test_precision, test_recall = evaluate_on_test(
    test_data,
    user_ids,
    item_ids,
    U,
    V,
    threshold=3.0,  # rating >= 4 dianggap "suka"
)

print("Evaluasi pada Data Test:")
print(f"MAE       : {test_mae:.4f}")
print(f"MSE       : {test_mse:.4f}")
print(f"RMSE      : {test_rmse:.4f}")
print(f"Precision : {test_precision:.4f} (rating >= 4)")
print(f"Recall    : {test_recall:.4f} (rating >= 4)")

# ----------------------------
# 8. Contoh Otomatis prediksi (semua kombinasi user-item)
# ----------------------------
# R_pred adalah prediksi rating dari user ke-i terhadap item ke-j
predictions = []

for user_index in range(num_users):
    for item_index in range(num_items):
        predictions.append({
            "userId": user_ids[user_index],
            "itemId": item_ids[item_index],
            "actual_rating": R[user_index][item_index],
            "mf_predicted_rating": round(R_pred[user_index][item_index], 1),
        })

predictions_df = pd.DataFrame(predictions)

# ----------------------------
# 9. (Opsional) Contoh prediksi manual (dibiarkan sebagai komentar)
# ----------------------------
# print("\nContoh prediksi rating user 0 terhadap item 1:")
# print(f"Rating aktual  : {R[0][1]}")
# print(f"Rating prediksi: {R_pred[0][1]:.2f}")

# ----------------------------
# 10. Simpan prediksi ke CSV
# ----------------------------
print("\nTotal prediksi:", len(predictions_df))
print(f"Total user: {num_users}")
print(f"Total item: {num_items}")

# =========================================
# Pendekatan 1: Filtering Vektorisasi
# =========================================
start_time_vect = time.time()

# Menggunakan filtering vektorisasi untuk memilih baris yang sesuai
result_df_vect = predictions_df[
    predictions_df["userId"].isin(user_ids) &
    predictions_df["itemId"].isin(item_ids)
].copy()

end_time_vect = time.time()
elapsed_time_vect = end_time_vect - start_time_vect

print("\nHasil dengan filtering vektorisasi:")
print(result_df_vect.head())  # tampilkan sebagian saja biar tidak kebanjiran output

output_path_vect = file_dir + "a_mf_ratings.csv"
result_df_vect.to_csv(output_path_vect, index=False)
print(f"\n📁 Hasil prediksi disimpan ke: {output_path_vect}")
print(f"Total waktu eksekusi (vectorized): {elapsed_time_vect:.2f} detik.")
