import pandas as pd

# Baca file CSV
df_i = pd.read_csv("../dataset_hotels/items.csv")
df_r = pd.read_csv("../dataset_hotels/ratings.csv")
df_u = df_r['userId'].nunique()

# Semua item yang tersedia
all_item_ids = set(df_i['id'])

# Item yang sudah pernah dirating
rated_item_ids = set(df_r['itemId'])

# Hitung
total_items = len(all_item_ids)
rated_items = len(rated_item_ids)
unrated_items = total_items - rated_items


# Hitung jumlah rating yang diberikan tiap user
df_rc = df_r['userId'].value_counts()
# Cari paling banyak & paling sedikit
max_ratings = df_rc.max()
min_ratings = df_rc.min()

# === Sparsity (kelangkaan rating pada matriks user-item) ===
R = len(df_r)                         # total rating
M = df_r['userId'].nunique()          # jumlah user
N = df_i['id'].nunique()              # jumlah movie (katalog)

sparsity = 1 - (R / (M * N)) if (M * N) else float('nan')
density  = 1 - sparsity               # alternatif: tingkat kepadatan rating

print(f"\n🏨 Hotel Dataset")
print(df_i)
print(f"\n⭐ Hotel Ratings Dataset")
print(df_r)
print(f"\n")

print(f"🏨 Jumlah total item              : {total_items}")
print(f"🏨✔️ Jumlah item yang sudah dirating : {rated_items}")
print(f"🏨❌ Jumlah item yang belum dirating : {unrated_items}")


# Hitung total user unik
print(f"\n👤 Total user: {df_u}\n")

print(f"👤⭐ Jumlah Paling Banyak user memberikan rating : {max_ratings}")
print(f"👤⭐ Jumlah Paling Sedikit user memberikan rating : {min_ratings}")

print(f"\n🧮 Sparsity (1 - R/(M×N)) : {sparsity:.6f}  ({sparsity*100:.2f}%)")
print(f"🧮 Density  (R/(M×N))     : {density:.6f}  ({density*100:.2f}%)")