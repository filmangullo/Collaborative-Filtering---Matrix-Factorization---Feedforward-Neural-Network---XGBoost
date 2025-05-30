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
# Hitung jumlah masing-masing genre
# ----------------------------
# Pecah kolom 'features' berdasarkan separator '|'
genre_series = items['features'].str.split('|').explode()

# Hitung jumlah masing-masing genre
genre_count = genre_series.value_counts()

# ----------------------------
# Print Data
# ----------------------------

print("========== DATASET: ITEMS ==========")
preview_items = pd.concat([items.head(8), items.tail(8)])
print(preview_items.to_string(index=False))  # Menampilkan tanpa indeks
print("\nJumlah data:", items.shape)

print("\n========== DATASET: RATINGS ==========")
preview_ratings = pd.concat([ratings.head(8), ratings.tail(8)])
print(preview_ratings.to_string(index=False))  # Menampilkan tanpa indeks
print("\nJumlah data:", ratings.shape)

print("========== JUMLAH SETIAP GENRE ==========")
genre_df = genre_count.reset_index()
genre_df.columns = ['Genre', 'Jumlah']
print(genre_df)

print("\n========== ONE-HOT ENCODED FEATURES ==========")
print(feature_encoding.head())
print("\nJumlah fitur hasil encoding:", feature_encoding.shape[1])

print("\n========== ITEM + FEATURES ==========")
print(item_with_features.head())
print("\nDimensi data akhir (item + fitur):", item_with_features.shape)
