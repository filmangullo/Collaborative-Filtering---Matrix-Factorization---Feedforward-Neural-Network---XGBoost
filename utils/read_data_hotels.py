import pandas as pd

# Baca file CSV
df_i = pd.read_csv("../dataset_hotels/items.csv")
df_r = pd.read_csv("../dataset_hotels/ratings.csv")
# Hitung jumlah user unik
jumlah_user = df_r['userId'].nunique()

print(f"\n🏨 Hotel Dataset")
print(df_i)
print(f"\n⭐ Hotel Ratings Dataset")
print(df_r)
print(f"\n👤 Jumlah user unik dalam data: {jumlah_user}")
