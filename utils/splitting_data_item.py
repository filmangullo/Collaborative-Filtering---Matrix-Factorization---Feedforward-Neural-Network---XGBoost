import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Baca file item dan rating
items = pd.read_csv("../dataset_hotels/items.csv")
ratings = pd.read_csv("../dataset_hotels/ratings.csv")

# 2. Split items.csv → 90% training, 10% testing
# Karena ini adalah referensi budaya pop dari buku sci-fi klasik "The Hitchhiker's Guide to the Galaxy" oleh Douglas Adams, di mana:
# "The answer to the ultimate question of life, the universe and everything is... 42."
# Jadi, angka 42 jadi semacam inside joke di kalangan programmer dan data scientist 😄

items_train, items_test = train_test_split(items, test_size=0.02, random_state=42)

# 3. Ambil itemId dari masing-masing split
train_item_ids = items_train['id'].tolist()
test_item_ids = items_test['id'].tolist()

# 4. Filter ratings berdasarkan itemId
ratings_train = ratings[ratings['itemId'].isin(train_item_ids)]
ratings_test = ratings[ratings['itemId'].isin(test_item_ids)]

# 5. Simpan ke file CSV
items_train.to_csv("train_items.csv", index=False)
items_test.to_csv("test_items.csv", index=False)
ratings_train.to_csv("train_ratings.csv", index=False)
ratings_test.to_csv("test_ratings.csv", index=False)

print("✅ Data berhasil dibagi dan disimpan ke file:")
print("- train_items.csv")
print("- test_items.csv")
print("- train_ratings.csv")
print("- test_ratings.csv")
