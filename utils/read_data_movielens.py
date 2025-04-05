import pandas as pd

# Baca file CSV
df_i = pd.read_csv("../dataset_movielens/items.csv")
df_r = pd.read_csv("../dataset_movielens/ratings.csv")

print(f"\n🎞️ MovieLens Dataset")
print(df_i)
print(f"\n⭐ MovieLens Ratings Dataset")
print(df_r)