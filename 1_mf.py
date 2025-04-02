import numpy as np
import pandas as pd

# Load dataset MovieLens 100K
# Format: user_id, item_id, rating, timestamp
column_names = ['userId', 'itemId', 'rating', 'timestamp']
ratings = pd.read_csv('ratings.csv', names=column_names, header=0)

# Map user_id dan item_id ke index numerik
user_ids = ratings['userId'].unique()
item_ids = ratings['itemId'].unique()
user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_to_index = {mid: idx for idx, mid in enumerate(item_ids)}

ratings['user_index'] = ratings['userId'].map(user_id_to_index)
ratings['item_index'] = ratings['itemId'].map(item_id_to_index)

# Matrix factorization parameters
num_users = len(user_ids)
num_items = len(item_ids)
K = 20  # Jumlah latent factors
alpha = 0.01  # Learning rate
lambda_reg = 0.02  # Regularisasi
epochs = 20

# Inisialisasi matriks U dan V
U = np.random.normal(scale=1./K, size=(num_users, K))
V = np.random.normal(scale=1./K, size=(num_items, K))

# Fungsi training menggunakan SGD
for epoch in range(epochs):
    total_loss = 0
    for row in ratings.itertuples():
        i = row.user_index
        j = row.item_index
        r_ij = row.rating

        pred = np.dot(U[i], V[j])
        error = r_ij - pred

        total_loss += error ** 2

        # Update rule
        U[i] += alpha * (error * V[j] - lambda_reg * U[i])
        V[j] += alpha * (error * U[i] - lambda_reg * V[j])

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Fungsi untuk prediksi rating
def predict_rating(user_id, item_id):
    if user_id in user_id_to_index and item_id in item_id_to_index:
        i = user_id_to_index[user_id]
        j = item_id_to_index[item_id]
        return np.dot(U[i], V[j])
    else:
        return None  # atau nilai default

# Contoh prediksi
user_id = 1
item_id = 1
pred = predict_rating(user_id, item_id)
print(f"Prediksi rating user {user_id} untuk item {item_id} adalah: {pred:.2f}")
