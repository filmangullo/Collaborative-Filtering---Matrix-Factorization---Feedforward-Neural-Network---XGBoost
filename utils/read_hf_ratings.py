import pandas as pd

# ----------------------------
# Load Data Movielens
# ----------------------------
file_dir = "../dataset_movielens/"  # Ganti jika direktori Anda berbeda
hf_movielens = pd.read_csv(file_dir + "c_hf_ratings.csv")
print("==========  HASIL HANDCRAFTED FEATURES FILM  ==========")
print(hf_movielens)

# ----------------------------
# Load Data Hotels
# ----------------------------
file_dir = "../dataset_hotels/"  # Ganti jika direktori Anda berbeda
hf_hotels = pd.read_csv(file_dir + "c_hf_ratings.csv")
print("==========  HASIL HANDCRAFTED FEATURES HOTEL  ==========")
print(hf_hotels)