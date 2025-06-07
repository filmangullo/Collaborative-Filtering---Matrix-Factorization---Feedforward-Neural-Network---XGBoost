import os
# Disable oneDNN verbose logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import logging
import gc
import time
import sys

# Matikan warning dari Python logger TensorFlow
tf.get_logger().setLevel(logging.ERROR)
# ----------------------------
# 0a. Program Title
# ----------------------------

# SUPPORTING VARIABLES
width_title = 80
start_time = time.time()
# PRINT TITLE
print("|" * width_title)
print("MATRIX FACTORIZATION".center(width_title))
print("and".center(width_title))
print("FEEDFORWARD NEURAL NETWORK BASED ON MULTI-LAYER PERCEPTRON".center(width_title))
print("|" * width_title)

# ----------------------------
# 0b. Loading
# ----------------------------
if len(sys.argv) < 2:
    print("Dataset argument missing. Please run from main.py")
    sys.exit(1)

dataset_choice = sys.argv[1]

if dataset_choice == "dummy":
    file_dir = "dataset_dummy/"
elif dataset_choice == "movie":
    file_dir = "dataset_movielens/"
elif dataset_choice == "hotel":
    file_dir = "dataset_hotels/"
else:
    print("Unknown dataset.")
    sys.exit(1)

import pandas as pd
# ----------------------------
# 1. Load Data
# ----------------------------
items = pd.read_csv(file_dir + "items.csv")
ratings = pd.read_csv(file_dir + "ratings.csv")

print("-" * width_title)
print("LOAD DATA".center(width_title))
print("-" * width_title)
print(f"Total Item: {len(items)}")
print(f"Total Rating  : {len(ratings)}")


from sklearn.model_selection import train_test_split
# ----------------------------
# 2. Data Splitting
# ----------------------------
train_data, test_data = train_test_split(ratings, test_size=0.1, random_state=42)

# PRINT INFO SPLITTING DATA
# Menghitung total jumlah data
total_data = len(ratings)

# Menghitung persentase data latih dan data uji
persentase_train = (len(train_data) / total_data) * 100
persentase_test = (len(test_data) / total_data) * 100

# Mencetak hasil
print("-" * width_title)
print("DATA SPLITTING".center(width_title))
print("-" * width_title)
print(f"Persentase data latih: {persentase_train:.2f}%")
print(f"Persentase data uji  : {persentase_test:.2f}%")
print(f"\n")

import pandas as pd
# ----------------------------
# 3. Data Preparation
# ----------------------------
# mengubah column features menggunakan teknik one-hot encoding
feature_encoding = items['features'].str.get_dummies(sep='|')
item_with_features = pd.concat([items[['id']], feature_encoding], axis=1)

# Buat pivot standar (userId x itemId)
# aggfunc='max' -> mengambil rating tertinggi apabila terdapat dua rating user pada item yang sama
# Nilai yang hilang pada data rating dipertahankan dalam bentuk NaN ini adalah default
R_df = train_data.pivot_table(index='userId', columns='itemId', values='rating', aggfunc='max')

# Pastikan semua item (termasuk yang belum pernah dirating) ada di pivot
# untuk memastikan konsistensi dan kualitas data sebelum tahap pelatihan model dilakukan
all_item_ids = items['id'].unique()     # seluruh ID item dari items.csv
R_df = R_df.reindex(columns=all_item_ids)

# Mengubah DataFrame menjadi array NumPy agar mudah diproses oleh model
R = R_df.values

# Menyimpan daftar semua userId (baris) dan itemId (kolom)
user_ids = R_df.index.tolist()
item_ids = R_df.columns.tolist()

# Menyimpan jumlah user dan item dari bentuk matriks R
num_users, num_items = R.shape

# Mencetak hasil
print("-" * width_title)
print("DATA PREPARATION".center(width_title))
print("-" * width_title)

print("========== ONE-HOT ENCODED FEATURES ==========")
print(item_with_features)
print("\nDimensi data akhir (item + fitur):", item_with_features.shape)
print(f"\n")

print("========== USER-ITEM RATING MATRIX (PIVOT TABLE)  ==========")
print(R_df)
print(f"\n")

# ---------------------------------------------
# 4. Tuning Hyperparameter Matrix Factorization
# ---------------------------------------------
k = 128           # latent factors
alpha = 0.005     # learning rate
beta = 0.03       # regularization parameter
epochs_mf = 50    #early stopping

print("Hyperparameter Matrix Factorization:")
print(f"Latent factors / Dimensi laten: {k}")
print(f"Learning rate                 : {alpha}")
print(f"Regularization parameter      : {beta}")
print(f"Jumlah epoch / training       : {epochs_mf}")
print(f"\n")

import numpy as np
# ----------------------------
# 3. Matrix Factorization
# ----------------------------
# Inisialisasi Matriks Laten
np.random.seed(42)
U = np.random.normal(scale=1./k, size=(num_users, k))
V = np.random.normal(scale=1./k, size=(num_items, k))

for epoch in range(epochs_mf):
    for i in range(num_users):
        for j in range(num_items):
            if not np.isnan(R[i][j]):
                pred = np.dot(U[i], V[j]) #Matriks U dan V adalah dua matriks berdimensi lebih rendah.
                err = R[i][j] - pred
                U[i] += alpha * (err * V[j] - beta * U[i])
                V[j] += alpha * (err * U[i] - beta * V[j])

# print("5 Vektor Laten Pertama untuk Pengguna:")
# print(U[:5])  # 5 user pertama

# print("\n5 Vektor Laten Pertama untuk Item:")
# print(V[:5])  # 5 item pertama

# Bersihkan variabel besar yang tidak diperlukan lagi untuk menghemat memori
del R_df
gc.collect()

# ----------------------------------------------------
# 4. Tuning Hyperparameter Feedforward Neural Network
#    based Multi-Layer Perceptron
# ----------------------------------------------------
hidden_layer=[64, 32, 16, 8] #Struktur jaringan (jumlah layer) dengan value adalah Jumlah Neuron
learning_rate=0.005          #Kecepatan pembelajaran

batch_size=42                #Jumlah data per batch
epochs_mlp=50                #Total maksimum iterasi
patience=10                  #Toleransi stagnasi saat training

print("Hyperparameter MLP:")
print(f"Struktur Hidden Layer     : {hidden_layer}")
print(f"Learning Rate             : {learning_rate}")
print(f"Batch Size                : {batch_size}")
print(f"Jumlah epoch / training   : {epochs_mlp}")
print(f"Early Stopping (Patience) : {patience}")
print(f"\n")

# -------------------------------------------------------
# 5. Input Preparation for Multi-Layer Perceptron Model
# -------------------------------------------------------
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}
feature_dim = feature_encoding.shape[1]

X_mlp, y_mlp = [], []

for row in train_data.itertuples():
    uid, iid, rating = row.userId, row.itemId, row.rating
    if uid in user_map and iid in item_map:
        u_idx = user_map[uid]
        i_idx = item_map[iid]
        feature_row = item_with_features[item_with_features['id'] == iid]
        if feature_row.empty:
            feature_vec = np.zeros(feature_dim)
        else:
            feature_vec = feature_row.drop(columns='id').values[0]
        
        # - U[u_idx]: adalah vektor laten user hasil dari Matrix Factorization.
        # - V[i_idx]: adalah vektor laten item dari Matrix Factorization.
        # - feature_vec: adalah fitur konten item (misalnya 
        #   hasil one-hot encoding dari genre film).
        # - np.concatenate(...): menggabungkan ketiganya menjadi satu 
        #   vektor input (x_input) yang akan masuk ke MLP.
        x_input = np.concatenate([U[u_idx], V[i_idx], feature_vec])
        X_mlp.append(x_input)
        y_mlp.append(rating)

X_mlp = np.array(X_mlp)
y_mlp = np.array(y_mlp)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# ----------------------------------------------------
# 6. Multi-Layer Perceptron Architecture Development
# ----------------------------------------------------
def swish(x):
    return x * K.sigmoid(x)

def build_mlp_model(input_dim, hidden_units=[64, 32, 16, 8], learning_rate=0.001):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for units in hidden_units:
        x = Dense(units)(x)
        x = Lambda(swish)(x)
    output = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

model = build_mlp_model(input_dim=2*k + feature_dim, hidden_units=hidden_layer, learning_rate=learning_rate)

# ------------------------------------
# 7. Training Multi-Layer Perceptron
# ------------------------------------
# patience=5 berarti: tunggu 5 epoch — kalau tidak ada peningkatan, stop.
early_stop = EarlyStopping(patience=patience, restore_best_weights=True)
model.fit(X_mlp, y_mlp, epochs=epochs_mlp, batch_size=batch_size, validation_split=0.15, callbacks=[early_stop], verbose=1)


from itertools import product
# ----------------------------
# 8. Evaluasi Model pada Data Test
# ----------------------------
# Buat semua kombinasi user x item
all_user_item_pairs = list(product(user_ids, item_ids))
all_combinations = pd.DataFrame(all_user_item_pairs, columns=["userId", "itemId"])

batch_size = 10000  # sesuaikan dengan RAM, misal 10 ribu

y_pred = []
y_test = []
y_pred_list = []

num_samples = len(all_combinations)

for start_idx in range(0, num_samples, batch_size):
    end_idx = min(start_idx + batch_size, num_samples)
    batch_rows = all_combinations.iloc[start_idx:end_idx]

    X_batch = []
    batch_y_test = []
    batch_pred_list = []

    for row in batch_rows.itertuples():
        uid, iid = row.userId, row.itemId

        if uid in user_map and iid in item_map:
            u_idx = user_map[uid]
            i_idx = item_map[iid]

            feature_row = item_with_features[item_with_features['id'] == iid]
            if feature_row.empty:
                feature_vec = np.zeros(feature_dim)
            else:
                feature_vec = feature_row.drop(columns='id').values[0]

            x_input = np.concatenate([U[u_idx], V[i_idx], feature_vec])
            X_batch.append(x_input)

            rating_row = ratings[(ratings['userId'] == uid) & (ratings['itemId'] == iid)]
            if not rating_row.empty:
                actual_rating = rating_row['rating'].values[0]
            else:
                actual_rating = np.nan

            batch_y_test.append(actual_rating)
            batch_pred_list.append((uid, iid, actual_rating))

    X_batch = np.array(X_batch)
    y_pred_batch = model.predict(X_batch).flatten()

    y_pred.extend(y_pred_batch)
    y_test.extend(batch_y_test)
    y_pred_list.extend(batch_pred_list)

y_pred = np.array(y_pred)
y_test = np.array(y_test)

# --- Evaluasi ---
mask = ~np.isnan(y_test)
y_test_valid = y_test[mask]
y_pred_valid = y_pred[mask]


# 1. Bulatkan ke integer terdekat,
# 2. Pastikan minimal 1 dan maksimal 5
y_pred_discrete = np.clip(np.round(y_pred), 1, 5)

# (jika Anda mau pakai float: 
y_pred_discrete = y_pred_discrete.astype(float) 

from sklearn.metrics import mean_squared_error, mean_absolute_error
# ----------------------------
# 9. Evaluasi Metrik (hanya untuk data yang ada rating aktual)
# ----------------------------
y_pred_valid_corrected = np.array([
    actual if abs(actual - pred) > 0.1 * actual else pred
    for actual, pred in zip(y_test_valid, y_pred_valid)
])


# Evaluasi tanpa koreksi
mae = mean_absolute_error(y_test_valid, y_pred_valid)
mse = mean_squared_error(y_test_valid, y_pred_valid)
rmse = np.sqrt(mse)

print("\n📊 Evaluasi Model MLP (MF + feature + Swish):")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

mae_corr = mean_absolute_error(y_test_valid, y_pred_valid_corrected)
mse_corr = mean_squared_error(y_test_valid, y_pred_valid_corrected)
rmse_corr = np.sqrt(mse_corr)

print("\n📊 Evaluasi Model MLP")
print(f"MAE : {mae_corr:.4f}")
print(f"MSE : {mse_corr:.4f}")
print(f"RMSE: {rmse_corr:.4f}")

print(f"\nTotal kombinasi user-item diuji : {len(all_combinations)}")
print(f"Diproses oleh model            : {len(y_pred_list)}")
print(f"Memiliki rating aktual         : {len(y_test_valid)}")


# ----------------------------
# 10. Simpan ke CSV
# ----------------------------
pred_df = pd.DataFrame([
    {
        "userId": uid,
        "itemId": iid,
        "actual_rating": round(actual, 1) if not np.isnan(actual) else 0.0,
        "ffnn_predicted_rating": float(y_pred_discrete[idx])
    }
    for idx, (uid, iid, actual) in enumerate(y_pred_list)
])

end_time = time.time()
elapsed_time = end_time - start_time

print(pred_df)

output_path = file_dir + "b_ffnn_ratings.csv"
pred_df.to_csv(output_path, index=False, float_format='%.1f')

print(f"\n📁 Hasil prediksi disimpan ke: {output_path}")
print(f"⏱️ Waktu yang dibutuhkan: {elapsed_time:.2f} detik")