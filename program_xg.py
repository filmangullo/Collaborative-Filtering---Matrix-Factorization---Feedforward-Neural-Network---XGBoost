import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
st.title("🎯 XGBoost Rating Recommender System")

# Langsung baca file dari directory (ganti path sesuai kebutuhan)
if len(sys.argv) < 2:
    print("Dataset argument missing. Please run from main.py")
    sys.exit(1)

dataset_choice = sys.argv[1]

if dataset_choice == "dummy":
    file_path = "dataset_dummy/c_hf_ratings.csv"
elif dataset_choice == "movie":
    file_path = "dataset_movielens/c_hf_ratings.csv"
elif dataset_choice == "hotel":
    file_path = "dataset_hotels/c_hf_ratings.csv"
else:
    print("Unknown dataset.")
    sys.exit(1)
df = pd.read_csv(file_path)

st.subheader("📋 Preview Data")
st.dataframe(df.head())

# Filter data untuk evaluasi
df_filtered = df[df['actual_rating'] != 0.0]

# Fitur dan target
feature_cols = [
    'ffnn_predicted_rating',
    'global_average_rating',
    'user_average_rating',
    'item_average_rating',
    'similar_users_rating',
    'similar_items_rating'
]
X = df_filtered[feature_cols]
y = df_filtered['actual_rating']

# -------------------------------------
# 2. Tuning Hyperparameter Pada XGBoost 
# -------------------------------------
n_estimators = 900
learning_rate = 0.5
max_depth = 9
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
gamma = 0.1
reg_alpha = 0.01
reg_lambda = 1.0
random_state = 42

# Nilai Default
# model = XGBRegressor(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=5,
#     min_child_weight=3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     gamma=0.1,
#     reg_alpha=0.01,
#     reg_lambda=1.0,
#     random_state=42
# )

# Nilai Terbaik
# n_estimators = 588
# learning_rate = 0.9
# max_depth = 9
# min_child_weight = 5
# subsample = 0.8
# colsample_bytree = 0.9
# gamma = 0.1
# reg_alpha = 0.01
# reg_lambda = 1.0
# random_state = 21

# n_estimators       => Jumlah pohon yang dibuat (boosting rounds)
# learning_rate      => Seberapa besar pengaruh tiap pohon baru terhadap model akhir
# max_depth          => Maksimal kedalaman setiap pohon
# min_child_weight   => Jumlah total bobot minimum (jumlah data) pada satu node agar dapat di-split / Minimum bobot yang dibutuhkan untuk membuat daun baru
# subsample          => Persentase data yang digunakan per pohon (prevent overfitting)
# colsample_bytree   => Persentase fitur yang digunakan per pohon
# gamma              => Minimum loss reduction untuk split
# reg_alpha          => L1 regularization (mengontrol kompleksitas model)
# reg_lambda         => L2 regularization
# random_state       => Reproducibility (hasil tetap setiap kali jalan)


# ----------------------------
# 3. Train XGBoost Regressor
# ----------------------------
model = XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    gamma=gamma,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda,
    random_state=random_state
)

model.fit(X, y)

y_pred_raw  = np.round(model.predict(X), 1)
y_pred = np.round(np.where(y_pred_raw < 1.0, 1.0, y_pred_raw), 1)
# ----------------------------
# 4. Evaluation
# ----------------------------
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.markdown("### 📊 Evaluation Metrics")
st.write(f"**MAE**: {mae:.4f}")
st.write(f"**RMSE**: {rmse:.4f}")
st.write(f"**R² Score**: {r2:.4f}")

st.markdown("### 🧾 Data yang Digunakan untuk Evaluasi")
st.dataframe(df_filtered)

# Tambahkan prediksi ke dataframe
df["xgb_predicted_rating"] = model.predict(df[feature_cols])

# ----------------------------
# 5. Plot Actual vs Predicted
# ----------------------------
st.markdown("### 📈 Actual vs Predicted Rating")
fig, ax = plt.subplots()
ax.scatter(y, y_pred, alpha=0.01)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Actual Rating")
ax.set_ylabel("Predicted Rating")
ax.set_title("Perbandingan Actual vs Predicted")

# Tambahkan ticks lebih rapat
ax.set_xticks(np.arange(1, 6.0, 0.5))
ax.set_yticks(np.arange(1, 6.0, 0.5))

st.pyplot(fig)


# ----------------------------
# 6. Rekomendasi Top-N
# ----------------------------
st.markdown("### 🔍 Rekomendasi Top-N Item per User")
selected_user = st.selectbox("Pilih User ID", sorted(df['userId'].unique()))
top_n = st.slider("Top-N Recommendation", 1, 100, 10)

def get_top_n(df, user_id, n=5):
    user_df = df[df['userId'] == user_id]
    return user_df.sort_values(by='xgb_predicted_rating', ascending=False).head(n)[['itemId', 'xgb_predicted_rating']]

recommended_items = get_top_n(df, selected_user, top_n)
st.dataframe(recommended_items)

