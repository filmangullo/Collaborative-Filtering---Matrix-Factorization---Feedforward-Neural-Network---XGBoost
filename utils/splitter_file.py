import pandas as pd

# Baca file CSV
df = pd.read_csv("../dataset_hotels/b_ffnn_ratings_x6.csv")

# Ambil itemId unik dan urutkan
unique_items = sorted(df['itemId'].unique())

# Bagi dua list itemId
mid_index = len(unique_items) // 2
items_a = unique_items[:mid_index]
items_b = unique_items[mid_index:]

# Filter baris berdasarkan itemId yang sudah dibagi
df_a = df[df['itemId'].isin(items_a)]
df_b = df[df['itemId'].isin(items_b)]

# Simpan ke file terpisah
df_a.to_csv("file_A.csv", index=False)
df_b.to_csv("file_B.csv", index=False)

print("✅ File A dan File B berhasil dibuat berdasarkan itemId.")
print(f"ItemId untuk File A: {items_a}")
print(f"ItemId untuk File B: {items_b}")
