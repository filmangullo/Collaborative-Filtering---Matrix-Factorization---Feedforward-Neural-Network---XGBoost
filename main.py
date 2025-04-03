import pandas as pd

# Baca dan timpa file asli
df = pd.read_csv('dataset_movielens/ratings.csv')
df.drop(columns=['timestamp'], inplace=True)
df.to_csv('dataset_movielens/ratings.csv', index=False)  # Timpa file original