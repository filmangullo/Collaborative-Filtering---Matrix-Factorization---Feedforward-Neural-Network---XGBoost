import pandas as pd

# Membaca file CSV
df = pd.read_csv("dataset_hotels/processed-hotels.csv")  # Ganti dengan nama file kamu

# Gabungkan rating, city_name, type, dan area_name menjadi satu kolom 'features'
df['features'] = df['rating'].astype(str) + '|' + df['city_name'] + '|' + df['type'] + '|' + df['area_name']

# Pilih hanya kolom yang dibutuhkan: id, hotel_name (diganti ke 'rating' sesuai format contoh), features
df_result = df[['id', 'hotel_name', 'features']]

# Ubah nama kolom hotel_name ke rating agar sesuai contoh kamu
df_result = df_result.rename(columns={'hotel_name': 'rating'})

# Tampilkan hasil
print(df_result)

# Simpan ke file baru jika perlu
df_result.to_csv("output.csv", index=False)