import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. SETUP DAGSHUB & MLFLOW
# Menggunakan link .mlflow sesuai konfigurasi DagsHub
MLFLOW_TRACKING_URI = "https://dagshub.com/harianja961352/Eksperimen_SML_Harianja.mlflow"
dagshub.init(repo_owner='harianja961352', repo_name='Eksperimen_SML_Harianja', mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Eksperimen_Bike_Sharing_Harianja")

# --- KRITERIA 2: MENGAKTIFKAN AUTOLOG (WAJIB) ---
# Ini akan otomatis mencatat parameter, metrics (RMSE, R2), model, dan artefak
mlflow.autolog()
# --------------------------------------------

# 2. LOAD DATA
# Mengarahkan ke folder data_preprocessing sesuai saran struktur folder reviewer
# export di Notebook agar file ini tersedia
dataset_path = 'data_preprocessing/hour_cleaned.csv'
df = pd.read_csv(dataset_path)
X = df.drop(columns=['cnt'])
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING MODEL
# Dengan autologging, kita tidak perlu lagi menulis mlflow.log_param atau mlflow.log_metric secara manual
with mlflow.start_run():
    # Parameter model
    n_estimators = 100
    max_depth = 10
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    print("Training selesai! Semua parameter, metrik, dan model telah dicatat otomatis oleh MLflow Autolog.")

# 4. SIMPAN MODEL UNTUK STREAMLIT
# Memastikan folder Dashboard ada sebelum menyimpan
dashboard_path = '../Dashboard'
if not os.path.exists(dashboard_path):
    os.makedirs(dashboard_path)

joblib.dump(model, os.path.join(dashboard_path, 'bike_model.joblib'))
print(f"Model berhasil disimpan ke {dashboard_path}/bike_model.joblib untuk kebutuhan Dashboard!")