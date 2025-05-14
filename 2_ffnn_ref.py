import os
# Disable oneDNN verbose logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import logging
import time

# Matikan warning dari Python logger TensorFlow
tf.get_logger().setLevel(logging.ERROR)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from itertools import product


# ----------------------------
# 0. Program Title
# ----------------------------
print(f"----------------------------------------------------------------")
print(f"   Matrix Factorization Feed-forward Neural Network -> MLP   ")
print(f"----------------------------------------------------------------")

# ----------------------------
# 1. Load Data
# ----------------------------
start_time = time.time()
items = pd.read_csv("dataset_dummy/items.csv")
feature_dummies = items['features'].str.get_dummies(sep='|')
item_with_features = pd.concat([items[['id']], feature_dummies], axis=1)

ratings = pd.read_csv('dataset_dummy/ratings.csv')
train_data, test_data = train_test_split(ratings, test_size=0.1, random_state=42)
# train_data = ratings
# test_data = ratings

# Menghitung total jumlah data
total_data = len(ratings)

# Menghitung persentase data latih dan data uji
persentase_train = (len(train_data) / total_data) * 100
persentase_test = (len(test_data) / total_data) * 100

# Mencetak hasil
print(f"Persentase data latih: {persentase_train:.2f}%")
print(f"Persentase data uji  : {persentase_test:.2f}%")
print(f"\n")

# ----------------------------
# 2. Create User-Item Matrix
# ----------------------------
# Buat pivot standar (userId x itemId)
R_df = train_data.pivot(index='userId', columns='itemId', values='rating')

# Pastikan semua item (termasuk yang belum pernah dirating) ada di pivot
all_item_ids = items['id'].unique()     # seluruh ID item dari items.csv
R_df = R_df.reindex(columns=all_item_ids)

# Jika mau memaksa item tak ber-rating dianggap rating=0, isi NaN dengan 0
# (Jika ingin menandai "tidak ada rating", lebih baik biarkan NaN saja.)
R_df = R_df.fillna(0)

# Lanjutkan seperti biasa
R = R_df.values
user_ids = R_df.index.tolist()
item_ids = R_df.columns.tolist()
num_users, num_items = R.shape

# ----------------------------
# 3. Matrix Factorization
# ----------------------------
k = 128     # latent factors
alpha = 0.005     # learning rate
beta = 0.03      # regularization parameter
epochs = 88     #early stopping

print("Hyperparameter Matrix Factorization:")
print(f"Latent factors / Dimensi laten: {k}")
print(f"Learning rate                 : {alpha}")
print(f"Regularization parameter      : {beta}")
print(f"Jumlah epoch / training       : {epochs}")
print(f"\n")

np.random.seed(42)
U = np.random.normal(scale=1./k, size=(num_users, k))
V = np.random.normal(scale=1./k, size=(num_items, k))

for epoch in range(epochs):
    for i in range(num_users):
        for j in range(num_items):
            if not np.isnan(R[i][j]):
                pred = np.dot(U[i], V[j])
                err = R[i][j] - pred
                U[i] += alpha * (err * V[j] - beta * U[i])
                V[j] += alpha * (err * U[i] - beta * V[j])

# ----------------------------
# 4. Siapkan data untuk MLP
# ----------------------------
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}
feature_dim = feature_dummies.shape[1]

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
        x_input = np.concatenate([U[u_idx], V[i_idx], feature_vec])
        X_mlp.append(x_input)
        y_mlp.append(rating)

X_mlp = np.array(X_mlp)
y_mlp = np.array(y_mlp)

# ----------------------------
# 5. Bangun MLP Model
# ----------------------------
def swish(x):
    return x * K.sigmoid(x)

input_layer = Input(shape=(2*k + feature_dim,))
hidden1 = Dense(128)(input_layer)
act1 = Lambda(swish)(hidden1)
hidden2 = Dense(64)(act1)
act2 = Lambda(swish)(hidden2)
output = Dense(1)(act2)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(0.001), loss='mse')

# ----------------------------
# 6. Training MLP
# ----------------------------
# patience=5 berarti: tunggu 5 epoch — kalau tidak ada peningkatan, stop.
early_stop = EarlyStopping(patience=20, restore_best_weights=True)
model.fit(X_mlp, y_mlp, epochs=epochs, batch_size=256, validation_split=0.2, callbacks=[early_stop], verbose=1)

# ----------------------------
# 7. Persiapkan Data Test (hanya user-item dengan rating aktual)
# ----------------------------
test_with_features = test_data.merge(item_with_features, left_on='itemId', right_on='id', how='left').fillna(0)
X_test = []
for row in test_with_features.itertuples():
    uid, iid = row.userId, row.itemId
    if uid in user_map and iid in item_map:
        u_idx = user_map[uid]
        i_idx = item_map[iid]
        feature_vec = np.array(row[5:])  # asumsi kolom 5 dst adalah fitur
        x_input = np.concatenate([U[u_idx], V[i_idx], feature_vec])
        X_test.append(x_input)

X_test = np.array(X_test)
y_test = test_with_features['rating'].values
user_ids_test = test_with_features['userId'].values
item_ids_test = test_with_features['itemId'].values

# ----------------------------
# 8. Prediksi & Evaluasi
# ----------------------------
y_pred = model.predict(X_test, batch_size=2048).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📊 Evaluasi Model MLP (Optimized):")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Total data uji: {len(y_test)}")

# ----------------------------
# 9. Simpan hasil prediksi
# ----------------------------
pred_df = pd.DataFrame({
    "userId": user_ids_test,
    "itemId": item_ids_test,
    "actual_rating": y_test,
    "ffnn_predicted_rating": np.round(y_pred, 1)
})
pred_df.to_csv("b_ffnn_ratings.csv", index=False)
print("📁 Hasil prediksi disimpan.")

