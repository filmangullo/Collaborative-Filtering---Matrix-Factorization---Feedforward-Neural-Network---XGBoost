import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

# ----------------------------
# 1. Load dan split dataset
# ----------------------------
ratings = pd.read_csv("ratings.csv")
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# ----------------------------
# 2. Buat user-item matrix dari training data
# ----------------------------
R_df = train_data.pivot(index='userId', columns='itemId', values='rating')
R = R_df.values  # biarkan NaN untuk masking
user_ids = R_df.index.tolist()
item_ids = R_df.columns.tolist()

num_users, num_items = R.shape
k = 50
alpha = 0.01
beta = 0.005
epochs = 100

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
    xs, ys = np.where(~np.isnan(R))
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
                if not np.isnan(R[i][j]):
                    prediction = np.dot(U[i, :], V[j, :].T)
                    eij = R[i][j] - prediction

                    # Update U dan V
                    U[i, :] += alpha * (eij * V[j, :] - beta * U[i, :])
                    V[j, :] += alpha * (eij * U[i, :] - beta * V[j, :])
        mae, mse, rmse = get_metrics(R, U, V)
        print(f"Epoch {epoch + 1}/{epochs}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    return U, V

# ----------------------------
# 6. Training
# ----------------------------
U, V = train_mf(R, U, V, alpha, beta, epochs)
R_pred = U.dot(V.T)

# ----------------------------
# 7. Evaluasi pada data test
# ----------------------------
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

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, mse, rmse

test_mae, test_mse, test_rmse = evaluate_on_test(test_data, user_ids, item_ids, U, V)
print("\nEvaluasi pada Data Test:")
print(f"MAE:  {test_mae:.4f}")
print(f"MSE:  {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")

# ----------------------------
# 8. Simpan prediksi ke CSV
# ----------------------------
# Buat lookup rating dari dataset asli
# Update loop prediksi
rating_lookup = ratings.set_index(['userId', 'itemId'])['rating'].to_dict()

predictions = []
for user_index in range(num_users):
    for item_index in range(num_items):
        u_id = user_ids[user_index]
        i_id = item_ids[item_index]
        pred_rating = R_pred[user_index][item_index]

        # Cari rating aktual dari dataset asli
        actual = rating_lookup.get((u_id, i_id), None)

        predictions.append({
            'userId': u_id,
            'itemId': i_id,
            'actual_rating': actual,
            'predicted_rating': pred_rating
        })

# Buat DataFrame dan simpan
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("a_mf_rating.csv", index=False)
print("\n✅ Hasil prediksi disimpan di a_mf_rating.csv")

# ----------------------------
# 9. Contoh Otomatis prediksi
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
            'predicted_rating': R_pred[user_index][item_index]
        })

predictions_df = pd.DataFrame(predictions)
# print(predictions_df.head())

print("Total prediksi:", len(predictions_df))
print(f"Total user: {num_users}")
print(f"Total item: {num_items}")
rows = []

for user_index in range(num_users):
    for item_index in range(num_items):
        row = predictions_df[
            (predictions_df['userId'] == user_ids[user_index]) &
            (predictions_df['itemId'] == item_ids[item_index])
        ]
        if not row.empty:
            rows.append(row)

# Gabung semua hasil ke satu DataFrame
result_df = pd.concat(rows, ignore_index=False)
print(result_df)

result_df.to_csv("a_mf_rating.csv", index=False)

# ----------------------------
# 10. Contoh prediksi manual
# ----------------------------
# Contoh prediksi rating user ke-0 terhadap movie ke-10
# Fungsi ini untuk mengecek saecara maunual terhadap user dan item.
#    Apa maksud R[0][1]?
#    Ini adalah 2D numpy array dari pivot userId vs itemId.
#      R[0] artinya baris pertama dari matrix R → ini adalah rating dari user pertama (index ke-0) terhadap semua item.
#    R[0][1] artinya:
#      Rating aktual dari user ke-0 terhadap item ke-1 (berdasarkan posisi, bukan ID asli)
print("\nContoh prediksi rating user 0 terhadap movie 1:")
print(f"Rating aktual: {R[0][1]}")
print(f"Rating prediksi: {R_pred[0][1]:.2f}")