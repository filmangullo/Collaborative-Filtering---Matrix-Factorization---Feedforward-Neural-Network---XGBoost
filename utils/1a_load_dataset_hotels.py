import pandas as pd

# ----------------------------
# Load Data Hotels
# ----------------------------
file_dir = "../dataset_hotels/"  # Ganti jika direktori Anda berbeda
items = pd.read_csv(file_dir + "items.csv")
ratings = pd.read_csv(file_dir + "ratings.csv")

# ----------------------------
# One-hot Encoding pada Kolom 'features'
# ----------------------------
feature_encoding = items['features'].str.get_dummies(sep='|')

# ----------------------------
# Gabungkan ID Item dengan Fitur yang Telah Di-encode
# ----------------------------
item_with_features = pd.concat([items[['id']], feature_encoding], axis=1)

# ----------------------------
# Print Data
# ----------------------------

print("========== DATASET: ITEMS ==========")
print(items.head())
print("\nJumlah data:", items.shape)

print("\n========== DATASET: RATINGS ==========")
print(ratings.head())
print("\nJumlah data:", ratings.shape)

print("\n========== ONE-HOT ENCODED FEATURES ==========")
print(feature_encoding.head())
print("\nJumlah fitur hasil encoding:", feature_encoding.shape[1])

print("\n========== ITEM + FEATURES ==========")
print(item_with_features.head())
print("\nDimensi data akhir (item + fitur):", item_with_features.shape)
