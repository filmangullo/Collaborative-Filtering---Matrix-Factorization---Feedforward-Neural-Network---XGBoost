import time
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------
# 0. Program Title
# ----------------------------
width_title = 80
start_time = time.time()
print("|" * width_title)
print("".center(width_title))
print("XGBOOST".center(width_title))
print("".center(width_title))
print("|" * width_title)

# ----------------------------
# 1. Ambil Dataset Path dari Argumen
# ----------------------------
if len(sys.argv) < 2:
    print("Dataset argument missing. Please run from main.py")
    sys.exit(1)

dataset_choice = sys.argv[1]

if dataset_choice == "dummy":
    file_path = "dataset_dummy/c_hf_ratings.csv"
elif dataset_choice == "movie":
    file_path = "dataset_movielens/c_hf_ratings.csv"
elif dataset_choice == "hotel":
    file_path = "dataset_hotels/final_feature_dataset.csv"
else:
    print("Unknown dataset.")
    sys.exit(1)

# ----------------------------
# 2. Load Data dan Latih Model
# ----------------------------
@st.cache_data
def load_and_train_model(file_path, top_n=8):
    top_n = int(top_n)  # pastikan top_n adalah integer
    df = pd.read_csv(file_path)

    X = df.drop(columns=["actual_rating"])
    y = df["actual_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.001,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    df["predicted_rating"] = np.round(model.predict(X), 1)

    top_n_recommendation = (
        df.sort_values(by=["userId", "predicted_rating"], ascending=[True, False])
          .groupby("userId")
          .head(top_n)
          .reset_index(drop=True)
    )

    top_n_recommendation.to_csv("top_n_recommendation.csv", index=False, float_format="%.1f")
    # top_n_recommendation.to_excel("top_n_recommendation.xlsx", index=False, float_format="%.1f")

    y_pred_all = model.predict(X_test)
    mask = y_test != 0.0
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred_all[mask]

    rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_filtered))
    r2 = r2_score(y_test_filtered, y_pred_filtered)

    scores = {
        "RMSE": rmse,
        "R2": r2
    }

    return top_n_recommendation, scores, df, y_test_filtered, y_pred_filtered


# ----------------------------
# 3. UI Streamlit
# ----------------------------
st.title("📊 Sistem Rekomendasi dengan XGBoost")
st.info(f"Menggunakan dataset: `{dataset_choice}`")

top_n_recommendation, scores, df_all, y_test, y_pred = load_and_train_model(file_path)

st.success("✅ Model dilatih & rekomendasi berhasil dihasilkan!")

st.subheader("📥 Top-N Rekomendasi")
user_ids = df_all['userId'].unique()
selected_user = st.selectbox("Pilih User ID", user_ids)

# Hindari warning SettingWithCopy
display_df = top_n_recommendation[top_n_recommendation['userId'] == selected_user][['itemId', 'predicted_rating']].copy()
display_df['predicted_rating'] = display_df['predicted_rating'].apply(lambda x: f"{x:.1f}")
st.table(display_df)

st.subheader("📈 Evaluasi Model")
st.write(f"**RMSE** : {scores['RMSE']:.4f}")
st.write(f"**R²**   : {scores['R2']:.4f}")

# Visualisasi
st.subheader("📊 Plot: Actual vs Predicted Ratings")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.02, edgecolors='k')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Rating")
ax.set_ylabel("Predicted Rating")
ax.set_title("Actual vs Predicted Ratings")
st.pyplot(fig)

# Uncomment untuk mengaktifkan fitur download Excel
# st.download_button(
#     label="📤 Download Rekomendasi (Excel)",
#     data=open("top_n_recommendation.xlsx", "rb").read(),
#     file_name="top_n_recommendation.xlsx"
# )
