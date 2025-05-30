import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
ratings = pd.read_csv("ratings.csv")

# List untuk hasil akhir
train_list = []
test_list = []

# Bagi per user
for user_id, group in ratings.groupby('userId'):
    if len(group) == 1:
        # Jika hanya 1 data, masukkan ke train (bisa ke test juga sesuai preferensi)
        train_list.append(group)
    else:
        train, test = train_test_split(group, test_size=0.5, random_state=42, shuffle=True)
        train_list.append(train)
        test_list.append(test)

# Gabungkan semua hasil
train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# Cek hasil
print(f"Total rating         : {len(ratings)}")
print(f"Jumlah train ratings : {len(train_df)}")
print(f"Jumlah test ratings  : {len(test_df)}")
print(f"Duplikasi? (should be 0): {pd.merge(train_df, test_df, how='inner', on=['userId', 'itemId', 'rating']).shape[0]}")

# Simpan hasil jika perlu
train_df.to_csv("ratings_train_stratified.csv", index=False)
test_df.to_csv("ratings_test_stratified.csv", index=False)
