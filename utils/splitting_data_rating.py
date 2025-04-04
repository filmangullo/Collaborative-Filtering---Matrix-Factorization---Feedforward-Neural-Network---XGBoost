import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Baca data
items = pd.read_csv("../dataset_movielens/items.csv")
ratings = pd.read_csv("../dataset_movielens/ratings.csv")

# 2. Split ratings.csv → 90% train, 10% test
ratings_train, ratings_test = train_test_split(ratings, test_size=0.1, random_state=42)

# 3. Ambil itemId unik dari masing-masing split
train_item_ids = ratings_train['itemId'].unique()
test_item_ids = ratings_test['itemId'].unique()

# 4. Filter items.csv berdasarkan itemId
items_train = items[items['id'].isin(train_item_ids)]
items_test = items[items['id'].isin(test_item_ids)]

# 5. Simpan ke file CSV
ratings_train.to_csv("train_ratings.csv", index=False)
ratings_test.to_csv("test_ratings.csv", index=False)
items_train.to_csv("train_items.csv", index=False)
items_test.to_csv("test_items.csv", index=False)

print("✅ Data berhasil dibagi dan disimpan ke file:")
print("- train_ratings.csv")
print("- test_ratings.csv")
print("- train_items.csv")
print("- test_items.csv")
