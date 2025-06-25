# train_xgb_only.py
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from training import read_tles_from_file, generate_sequences_from_tles
from training import auto_generate_flyby_neo_czml

# Generate a random NEO flyby CZML file for Cesium
auto_generate_flyby_neo_czml()

def main():
    # 1) load the same sequences used for LSTM/GRU
    tles = read_tles_from_file("gp.txt", num_samples=20000)
    seqs = generate_sequences_from_tles(tles, num_points=30, step_sec=10)

    # 2) flatten & scale
    N, T, F = seqs.shape
    flat = seqs.reshape(-1, F)
    scaler = MinMaxScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(N, T, F)

    # 3) features = all but last step; labels = last step XYZ
    X = scaled[:, :-1, :]
    y = scaled[:, -1, :3]
    X_flat = X.reshape(N, -1)

    # 4) train XGB regressor
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)
    xgb.fit(X_flat, y)

    # 5) save model + scaler under the exact names as loader expects
    joblib.dump(xgb, "xgb_xgb_model.joblib")
    joblib.dump(scaler, "xgb_scaler.pkl")
    print("✅ Trained and saved XGBoost model.")

if __name__ == "__main__":
    main()
