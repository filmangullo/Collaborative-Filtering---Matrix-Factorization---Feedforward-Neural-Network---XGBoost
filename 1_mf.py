import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

# ----------------------------
# 0. Program Title
# ----------------------------
print(f"----------------------------------------------------------------")
print(f"   Matrix Factorization via Stochastic Gradient Descent (SGD)   ")
print(f"----------------------------------------------------------------")

# ----------------------------
# 1. Load dan split dataset
# ----------------------------
file_dir = "dataset_hotels/"
# Load dataset Dummy (gunakan file ratings.csv dari Dummny manual)
# ratings = pd.read_csv('dataset_dummy/ratings.csv')  # pastikan file ratings.csv ada di direktori yang sama
# Load dataset MovieLens (gunakan file ratings.csv dari MovieLens)
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
print(f"Test Data Presentation: {persentase_test:.2f}%")
print(f"\n")

# ----------------------------
# 2. Buat user-item matrix dari training data
# ----------------------------
R_df = train_data.pivot_table(index='userId', columns='itemId', values='rating', aggfunc='mean')
R = R_df.fillna(0).values  # Mengisi NaN dengan 0
user_ids = R_df.index.tolist()
item_ids = R_df.columns.tolist()

# Hyperparameter Matrix Factorization
num_users, num_items = R.shape
k = 42     # latent factors
alpha = 0.005     # learning rate
beta = 0.05     # regularization parameter
epochs = 50     #early stopping

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
U = np.random.normal(scale=1./k, size=(num_users, k))
V = np.random.normal(scale=1./k, size=(num_items, k))

# ----------------------------
# 4. Fungsi evaluasi metrik
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
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, mse, rmse

# ----------------------------
# 5. Fungsi training MF (SGD)
# ----------------------------
def train_mf(R, U, V, alpha, beta, epochs):
    for epoch in range(epochs):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:  # Hanya update jika ada rating
                    prediction = np.dot(U[i, :], V[j, :].T)
                    eij = R[i][j] - prediction

                    U[i, :] += alpha * (eij * V[j, :] - beta * U[i, :])
                    V[j, :] += alpha * (eij * U[i, :] - beta * V[j, :])

                    U[i, :] = np.clip(U[i, :], -5, 5)
                    V[j, :] = np.clip(V[j, :], -5, 5)

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
R_pred = U.dot(V.T)
R_pred = np.clip(R_pred, 1.0, 5.0)  # pastikan prediksi di rentang 1-5

# ----------------------------
# 7. Evaluasi pada data test
# ----------------------------
print("\n")
def evaluate_on_test(test_df, user_ids, item_ids, U, V):
    y_true, y_pred = [], []
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    for row in test_df.itertuples():
        u_id, i_id, r = row.userId, row.itemId, row.rating
        if u_id in user_map and i_id in item_map:
            u_idx = user_map[u_id]
            i_idx = item_map[i_id]
            pred = np.dot(U[u_idx], V[i_idx])
            y_true.append(r)
            y_pred.append(pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        print("⚠️ Tidak ada data test yang cocok dengan user/item di matriks training.")
        return float("nan"), float("nan"), float("nan")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, mse, rmse

test_mae, test_mse, test_rmse = evaluate_on_test(test_data, user_ids, item_ids, U, V) #menggunakan test_data
print("Evaluasi pada Data Test:")
print(f"MAE : {test_mae:.4f}")
print(f"MSE : {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")

# ----------------------------
# 8. Contoh Otomatis prediksi
# ----------------------------
# Gunakan R_pred sebagai fitur tambahan dalam model machine learning
# Hasil akhir dari proses Matrix Factorization (faktor U × Vᵗ), 
# yaitu prediksi sistem terhadap semua kombinasi user dan item, termasuk yang belum pernah diberi rating.
# yang berarti R_pred adalah Prediksi rating dari user ke-i terhadap item ke-j, 
# berdasarkan pola laten yang dipelajari dari data historis.
predictions = []

for user_index in range(num_users):
    for item_index in range(num_items):
        predictions.append({
            'userId': user_ids[user_index],
            'itemId': item_ids[item_index],
            'actual_rating': R[user_index][item_index],
            'mf_predicted_rating': round(R_pred[user_index][item_index], 1)
        })

predictions_df = pd.DataFrame(predictions)

# ----------------------------
# 9. Contoh prediksi manual
# ----------------------------
# Contoh prediksi rating user ke-0 terhadap item ke-10
# Fungsi ini untuk mengecek saecara maunual terhadap user dan item.
#    Apa maksud R[0][1]?
#    Ini adalah 2D numpy array dari pivot userId vs itemId.
#      R[0] artinya baris pertama dari matrix R → ini adalah rating dari user pertama (index ke-0) terhadap semua item.
#    R[0][1] artinya:
#      Rating aktual dari user ke-0 terhadap item ke-1 (berdasarkan posisi, bukan ID asli)
# print("\nContoh prediksi rating user 0 terhadap item 1:")
# print(f"Rating aktual: {R[0][1]}")
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
    predictions_df['userId'].isin(user_ids) & predictions_df['itemId'].isin(item_ids)
].copy()

end_time_vect = time.time()
elapsed_time_vect = end_time_vect - start_time_vect

print("\nHasil dengan filtering vektorisasi:")
print(result_df_vect)

output_path_vect = file_dir + "a_mf_ratings.csv"
result_df_vect.to_csv(output_path_vect, index=False)
print(f"\n📁 Hasil prediksi disimpan ke: {output_path_vect}")
print(f"Total waktu eksekusi (vectorized): {elapsed_time_vect:.2f} detik.")


# # =========================================
# # Pendekatan 2: Nested Loop dengan Progress Bar
# # =========================================
# rows = []
# total_iterations = num_users * num_items
# update_interval = 8  # update progress bar setiap 8 iterasi
# counter = 0

# start_time_loop = time.time()

# with tqdm(total=total_iterations, desc="🔄 Menggabungkan hasil prediksi", unit="pair") as pbar:
#     for user_index in range(num_users):
#         for item_index in range(num_items):
#             # Melakukan seleksi baris berdasarkan kombinasi userId dan itemId
#             row = predictions_df[
#                 (predictions_df['userId'] == user_ids[user_index]) &
#                 (predictions_df['itemId'] == item_ids[item_index])
#             ]
#             if not row.empty:
#                 rows.append(row)
#             counter += 1
#             if counter % update_interval == 0:
#                 pbar.update(update_interval)
#     # Update sisa iterasi apabila ada
#     remainder = counter % update_interval
#     if remainder:
#         pbar.update(remainder)

# end_time_loop = time.time()
# elapsed_time_loop = end_time_loop - start_time_loop

# result_df_loop = pd.concat(rows, ignore_index=False)
# print("\nHasil dengan nested loop:")
# print(result_df_loop)

# output_path_loop = "a_mf_ratings_loop.csv"
# result_df_loop.to_csv(output_path_loop, index=False)
# print(f"\n📁 Hasil prediksi disimpan ke: {output_path_loop}")
# print(f"Total waktu eksekusi (nested loop): {elapsed_time_loop:.2f} detik.")