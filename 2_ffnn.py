import os
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

# Disable oneDNN verbose logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ----------------------------
# 1. Load Data
# ----------------------------
items = pd.read_csv("dataset_dummy/items.csv")
feature_dummies = items['features'].str.get_dummies(sep='|')
item_with_features = pd.concat([items[['id']], feature_dummies], axis=1)

ratings = pd.read_csv('dataset_dummy/ratings.csv')
# train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_data = ratings
test_data = ratings

# ----------------------------
# 2. Create User-Item Matrix
# ----------------------------
R_df = train_data.pivot(index='userId', columns='itemId', values='rating')
R = R_df.fillna(np.nan).values
user_ids = R_df.index.tolist()
item_ids = R_df.columns.tolist()
num_users, num_items = R.shape

# ----------------------------
# 3. Matrix Factorization
# ----------------------------
k = 50  # latent factors
alpha = 0.01  # learning rate
beta = 0.005    # regularization parameter
epochs = 100

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
genre_dim = feature_dummies.shape[1]

X_mlp, y_mlp = [], []

for row in train_data.itertuples():
    uid, iid, rating = row.userId, row.itemId, row.rating
    if uid in user_map and iid in item_map:
        u_idx = user_map[uid]
        i_idx = item_map[iid]
        genre_row = item_with_features[item_with_features['id'] == iid]
        if genre_row.empty:
            genre_vec = np.zeros(genre_dim)
        else:
            genre_vec = genre_row.drop(columns='id').values[0]
        x_input = np.concatenate([U[u_idx], V[i_idx], genre_vec])
        X_mlp.append(x_input)
        y_mlp.append(rating)

X_mlp = np.array(X_mlp)
y_mlp = np.array(y_mlp)

# ----------------------------
# 5. Bangun MLP Model
# ----------------------------
def swish(x):
    return x * K.sigmoid(x)

input_layer = Input(shape=(2*k + genre_dim,))
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
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(X_mlp, y_mlp, epochs=100, batch_size=256, validation_split=0.1, callbacks=[early_stop], verbose=1)

# ----------------------------
# 7. Evaluasi Model pada Data Test (versi diperbaiki)
# ----------------------------
from itertools import product

# Buat semua kombinasi user x item
all_user_item_pairs = list(product(user_ids, item_ids))
all_combinations = pd.DataFrame(all_user_item_pairs, columns=["userId", "itemId"])

X_test, y_test, y_pred_list = [], [], []

for row in all_combinations.itertuples():
    uid, iid = row.userId, row.itemId

    if uid in user_map and iid in item_map:
        u_idx = user_map[uid]
        i_idx = item_map[iid]
        
        # Ambil fitur genre
        genre_row = item_with_features[item_with_features['id'] == iid]
        if genre_row.empty:
            genre_vec = np.zeros(genre_dim)
        else:
            genre_vec = genre_row.drop(columns='id').values[0]

        x_input = np.concatenate([U[u_idx], V[i_idx], genre_vec])
        X_test.append(x_input)

        # Cek apakah ada rating aktual
        rating_row = ratings[(ratings['userId'] == uid) & (ratings['itemId'] == iid)]
        if not rating_row.empty:
            actual_rating = rating_row['rating'].values[0]
        else:
            actual_rating = np.nan  # bisa digunakan nanti untuk filtering evaluasi

        y_test.append(actual_rating)
        y_pred_list.append((uid, iid, actual_rating))

X_test = np.array(X_test)
y_test = np.array(y_test)
y_pred = model.predict(X_test).flatten()

# ----------------------------
# 8. Evaluasi Metrik (hanya untuk data yang ada rating aktual)
# ----------------------------
mask = ~np.isnan(y_test)
y_test_valid = y_test[mask]
y_pred_valid = y_pred[mask]

mae = mean_absolute_error(y_test_valid, y_pred_valid)
mse = mean_squared_error(y_test_valid, y_pred_valid)
rmse = np.sqrt(mse)

print("\n📊 Evaluasi Model MLP (MF + Genre + Swish):")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"\nTotal kombinasi user-item diuji : {len(all_combinations)}")
print(f"Diproses oleh model            : {len(y_pred_list)}")
print(f"Memiliki rating aktual         : {len(y_test_valid)}")

# ----------------------------
# 9. Print Prediksi dalam Format Lengkap
# ----------------------------
print("userId,itemId,actual_rating,ffnn_predicted_rating")
for idx, (uid, iid, actual) in enumerate(y_pred_list):
    pred_score = y_pred[idx]
    actual_str = f"{actual:.1f}" if not np.isnan(actual) else "0.0"
    print(f"{uid},{iid},{actual_str},{pred_score:.4f}")


# ----------------------------
# 10. Simpan ke CSV
# ----------------------------
pred_df = pd.DataFrame([
    {"userId": uid, "itemId": iid, "actual_rating": actual if not np.isnan(actual) else 0.0, "ffnn_predicted_rating": y_pred[idx]}
    for idx, (uid, iid, actual) in enumerate(y_pred_list)
])

# Simpan ke file
output_path = "b_ffnn_ratings.csv"
pred_df.to_csv(output_path, index=False)
print(f"\n📁 Hasil prediksi disimpan ke: {output_path}")
