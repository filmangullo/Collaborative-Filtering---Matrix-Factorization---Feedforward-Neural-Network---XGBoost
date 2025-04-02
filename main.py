import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Concatenate, Dense

# 1. Load dataset MovieLens
ratings = pd.read_csv('ratings.csv')  # file dari MovieLens dataset
print(ratings.head())

# 2. Encode userId dan movieId menjadi index numerik
user_enc = LabelEncoder()
movie_enc = LabelEncoder()

ratings['user'] = user_enc.fit_transform(ratings['userId'])
ratings['movie'] = movie_enc.fit_transform(ratings['movieId'])

num_users = ratings['user'].nunique()
num_movies = ratings['movie'].nunique()

# 3. Split data
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# 4. Matrix Factorization model
def build_mf_model(n_users, n_movies, n_latent=10):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    
    user_embed = Embedding(n_users, n_latent)(user_input)
    movie_embed = Embedding(n_movies, n_latent)(movie_input)
    
    user_vec = Flatten()(user_embed)
    movie_vec = Flatten()(movie_embed)
    
    dot = Dot(axes=1)([user_vec, movie_vec])
    
    model = Model(inputs=[user_input, movie_input], outputs=dot)
    model.compile(optimizer='adam', loss='mse')
    return model

mf_model = build_mf_model(num_users, num_movies)
mf_model.summary()

# 5. Latih MF
mf_model.fit([train['user'], train['movie']], train['rating'], 
             epochs=10, batch_size=256, verbose=1)

# 6. Ekstrak fitur laten
user_layer = mf_model.get_layer(index=2)
movie_layer = mf_model.get_layer(index=3)

user_latent = user_layer.get_weights()[0]
movie_latent = movie_layer.get_weights()[0]

# 7. Gabungkan fitur laten untuk training MLP
def build_dataset(df):
    user_feats = user_latent[df['user'].values]
    movie_feats = movie_latent[df['movie'].values]
    features = np.hstack([user_feats, movie_feats])
    return features

X_train_mlp = build_dataset(train)
X_test_mlp = build_dataset(test)
y_train = train['rating'].values
y_test = test['rating'].values

# 8. MLP Model
mlp = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train_mlp.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

mlp.compile(optimizer='adam', loss='mse')
mlp.fit(X_train_mlp, y_train, epochs=10, batch_size=128, verbose=1)

# 9. Ekstrak prediksi MLP untuk digunakan di XGBoost
train['mlp_pred'] = mlp.predict(X_train_mlp)
test['mlp_pred'] = mlp.predict(X_test_mlp)

# 10. Buat fitur akhir untuk XGBoost
def final_features(df, X_latent, mlp_preds):
    df_final = df[['user', 'movie']].copy()
    df_final['mlp_pred'] = mlp_preds.flatten()
    return df_final

X_train_xgb = final_features(train, X_train_mlp, train['mlp_pred'].values)
X_test_xgb = final_features(test, X_test_mlp, test['mlp_pred'].values)

# 11. Latih model XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train_xgb, y_train)

# 12. Evaluasi model akhir
from sklearn.metrics import mean_squared_error

preds = xgb_model.predict(X_test_xgb)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE Hybrid Model: {rmse:.4f}')
