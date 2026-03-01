import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. SETUP MLFLOW LOKAL (PENTING)
# Menggunakan folder /tmp agar sukses dijalankan di GitHub Actions (Linux)
#mlflow.set_tracking_uri("file:///tmp/mlruns")
#mlflow.set_experiment("Eksperimen_Bike_Sharing_Harianja")

# --- KRITERIA 2: MENGAKTIFKAN AUTOLOG (WAJIB) ---
# Memastikan semua parameter, metrik, dan model dicatat otomatis tanpa log manual
mlflow.sklearn.autolog()
# --------------------------------------------

# 2. LOAD DATA
# Pastikan terminal berada di folder yang sama dengan file CSV ini
dataset_path = 'hour_cleaned.csv'
df = pd.read_csv(dataset_path)

# Memisahkan fitur dan target
X = df.drop(columns=['cnt'])
y = df['cnt']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING MODEL
# Dengan autologging, kita HANYA perlu memanggil .fit()
with mlflow.start_run():
    # Inisialisasi model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    # Proses Training
    # Begitu .fit() dipanggil, MLflow Autolog akan mencatat semuanya secara otomatis
    model.fit(X_train, y_train)
    
    print("Training selesai! Semua parameter, metrik, dan model telah dicatat otomatis oleh MLflow Autolog.")

# 4. SIMPAN MODEL UNTUK DASHBOARD
# Memastikan folder Dashboard tersedia di luar folder MLProject
dashboard_path = '../Dashboard'
if not os.path.exists(dashboard_path):
    os.makedirs(dashboard_path)

joblib.dump(model, os.path.join(dashboard_path, 'bike_model.joblib'))
print(f"Model berhasil disimpan ke {dashboard_path}/bike_model.joblib!")