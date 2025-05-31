import time
# ----------------------------
# 0. Program Title
# ----------------------------

# SUPPORTING VARIABLES
width_title = 80
start_time = time.time()
# PRINT TITLE
print("-" * width_title)
print("OPTIMIZATION OF HYBRID-BASED COLLABORATIVE FILTERING".center(width_title))
print("USING".center(width_title))
print("MATRIX FACTORIZATION, FEEDFORWARD NEURAL NETWORK, AND XGBOOST".center(width_title))
print("TO IMPROVE RECOMMENDATIONS".center(width_title))
print("-" * width_title)

import pandas as pd
# ----------------------------
# 1. Load Data
# ----------------------------
file_dir = "dataset_dump/"
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
print(item_with_features.head())
print("\nDimensi data akhir (item + fitur):", item_with_features.shape)
print(f"\n")