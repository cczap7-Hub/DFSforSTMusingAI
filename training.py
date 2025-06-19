import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import jday
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, Input
import requests

# --- Download TLEs ---
def download_tles(
    url: str = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    num_samples: int = 100
):
    lines = requests.get(url).text.strip().splitlines()
    tles = []
    for i in range(0, len(lines) - 2, 3):
        # skip any header lines
        if lines[i].startswith("1 ") or lines[i].startswith("2 "):
            continue
        tles.append((lines[i], lines[i+1], lines[i+2]))
        if len(tles) >= num_samples:
            break
    return tles

# --- Generate position/velocity/eccentricity sequences ---
def generate_sequences_from_tles(
    tles,
    num_points: int = 20,
    step_sec: int = 10
):
    sequences = []
    for name, tle1, tle2 in tles:
        sat = Satrec.twoline2rv(tle1, tle2)
        start_time = datetime.now(timezone.utc)
        pos_seq = []
        for i in range(num_points):
            dt = start_time + timedelta(seconds=i*step_sec)
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second)
            err, r, v = sat.sgp4(jd, fr)
            if err != 0 or not np.all(np.isfinite(r)):
                print(f"SGP4 error {err} for {name} at step {i}")
                break
            pos_seq.append(r)
        if len(pos_seq) == num_points:
            pos_seq = np.array(pos_seq)              # shape (T,3)
            vel_seq = np.gradient(pos_seq, axis=0)   # shape (T,3)
            ecc = sat.ecco                            # scalar eccentricity
            ecc_seq = np.full((num_points, 1), ecc)  # shape (T,1)
            # concatenate into (T,7): x,y,z,vx,vy,vz,ecco
            seq = np.concatenate([pos_seq, vel_seq, ecc_seq], axis=1)
            sequences.append(seq)
    print(f"Generated {len(sequences)} valid sequences out of {len(tles)} TLEs.")
    return np.array(sequences)


# --- Build models expecting 7 features per timestep ---
def build_lstm_model():
    model = keras.Sequential([
        Input(shape=(19, 7)),
        layers.Bidirectional(layers.LSTM(512, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(512)),
        layers.Dense(64, activation="relu"),
        layers.Dense(3)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru_model():
    model = keras.Sequential([
        Input(shape=(19, 7)),
        layers.Bidirectional(layers.GRU(512, return_sequences=True)),
        layers.Bidirectional(layers.GRU(512)),
        layers.Dense(64, activation="relu"),
        layers.Dense(3)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# --- Main training and evaluation ---
def train_and_evaluate(
    model_builder,
    sequences: np.ndarray,
    epochs: int = 1000,
    batch_size: int = 64,
    model_name: str = "model"
):
    scaler = MinMaxScaler()
    N, T, F = sequences.shape  # F is now 7
    data = sequences.reshape(-1, F)
    scaled = scaler.fit_transform(data).reshape(N, T, F)

    X = scaled[:, :-1, :]      # all but last timestep
    y = scaled[:, -1, :3]      # target is final x,y,z

    model = model_builder()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)

    # Predict & inverse-transform
    pred_scaled = model.predict(X)
    last_vels   = sequences[:, -1, 3:6]                 # (N,3)
    pad         = np.zeros((N, 1))                      # for ecc dim
    pred_padded = np.hstack([pred_scaled, last_vels, pad])  # (N,7)
    pred_unscaled = scaler.inverse_transform(pred_padded)[:, :3]

    true_unscaled = sequences[:, -1, :3]
    errors = np.linalg.norm(true_unscaled - pred_unscaled, axis=1)
    mags   = np.linalg.norm(true_unscaled, axis=1)
    error_pct = np.mean(errors / mags) * 100
    print(f"Mean error: {np.mean(errors):.2f} km, Mean error %: {error_pct:.2f}%")

    # --- SAVE MODEL AND SCALER ---
    model.save(f"{model_name}_full_model.keras")
    import joblib
    joblib.dump(scaler, f"{model_name}_scaler.pkl")
    print(f"Saved model to {model_name}_full_model.keras and scaler to {model_name}_scaler.pkl")

    return model, scaler, error_pct


if __name__ == "__main__":
    print("Downloading TLEs…")
    tles = download_tles(num_samples=10000)

    print("Generating sequences…")
    sequences = generate_sequences_from_tles(tles, num_points=20)

    print(f"Got {len(sequences)} valid satellite sequences.\n")

    print("Training LSTM…")
    lstm_model, lstm_scaler, lstm_err = train_and_evaluate(
        build_lstm_model, sequences, model_name="lstm"
    )

    print("\nTraining GRU…")
    gru_model, gru_scaler, gru_err = train_and_evaluate(
        build_gru_model, sequences, model_name="gru"
    )

    if lstm_err < 5 and gru_err < 5:
        print("\nBoth models achieved <5% mean error!")
    else:
        print("\nTry increasing data, epochs, or model size for better accuracy.")


################################################################################
#                    Helper functions for STMS_m7.ipynb inference             #
################################################################################

def read_tles_from_file(filename, num_samples=None):
    lines = open(filename, "r").read().strip().splitlines()
    tles = []
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
        tles.append((lines[i], lines[i+1], lines[i+2]))
        if num_samples and len(tles) >= num_samples:
            break
    return tles

def generate_benchmark_positions(tles, num_points=30, step_sec=20):
    # reuse your updated sequence generator
    full_seq = generate_sequences_from_tles(
        tles=tles, num_points=num_points, step_sec=step_sec
    )
    return full_seq[:, :, :3]

def load_xgboost_model(model_name: str):
    import joblib
    model  = joblib.load(f"{model_name}_xgb_model.joblib")
    scaler = joblib.load(f"{model_name}_scaler.pkl")
    return model, scaler

def predict_xgboost_sequence(model, scaler, sequences):
    N, T, F = sequences.shape
    flat    = sequences.reshape(-1, F)
    scaled  = scaler.transform(flat).reshape(N, T, F)
    X_flat  = scaled[:, :-1, :].reshape(N, (T-1)*F)
    preds_sc= model.predict(X_flat)
    pad     = np.zeros((N, F-3))
    inv     = scaler.inverse_transform(np.hstack([preds_sc, pad]))
    return inv[:, :3]

def load_trained_model(model_name: str):
    model  = keras.models.load_model(f"{model_name}_full_model.keras")
    import joblib
    scaler = joblib.load(f"{model_name}_scaler.pkl")
    return model, scaler

def predict_sequence(model, scaler, sequences):
    N, T, F = sequences.shape
    flat    = sequences.reshape(-1, F)
    scaled  = scaler.transform(flat).reshape(N, T, F)
    X       = scaled[:, :-1, :]
    preds_sc= model.predict(X, verbose=0)
    last_vels = sequences[:, -1, 3:6]
    pad       = np.zeros((N, 1))
    padded    = np.hstack([preds_sc, last_vels, pad])
    inv       = scaler.inverse_transform(padded)
    return inv[:, :3]

def save_predictions_for_cesium_with_actual(predictions, benchmark_positions, model_name="Model"):
    filename = f"predictions_{model_name}.json"
    output = []
    for idx, pt in enumerate(predictions):
        orbit = [
            {"x": float(x)*1000, "y": float(y)*1000, "z": float(z)*1000}
            for x,y,z in benchmark_positions[idx]
        ]
        output.append({
            "id": f"Satellite {idx+1} – {model_name}",
            "orbit": orbit,
            "predicted": {
                "x": pt[0]*1000,
                "y": pt[1]*1000,
                "z": pt[2]*1000
            }
        })
    with open(filename, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved CesiumJSON → {filename}")
