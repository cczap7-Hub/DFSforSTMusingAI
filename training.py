import json
import matplotlib.pyplot as plt
import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import jday
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, Input
import requests
import random
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import requests
import random
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler


# Download TLEs
def download_tles(
    url: str = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    num_samples: int = 10000
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

# Generate position/velocity/eccentricity sequences
def generate_sequences_from_tles(
    tles,
    num_points: int = 30,
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


# Building models expecting 7 features per timestep
def build_lstm_model(num_timesteps, num_features):
    model = keras.Sequential([
        Input(shape=(num_timesteps, num_features)),
        layers.Bidirectional(layers.LSTM(512, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(512)),
        layers.Dense(64, activation="relu"),
        layers.Dense(3)
    ])
    model.compile(optimizer="adam", loss="mae")
    return model

def build_gru_model(num_timesteps, num_features):
    model = keras.Sequential([
        Input(shape=(num_timesteps, num_features)),
        layers.Bidirectional(layers.GRU(512, return_sequences=True)),
        layers.Bidirectional(layers.GRU(512)),
        layers.Dense(64, activation="relu"),
        layers.Dense(3)
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


# Main training and evaluation
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
    last_vels   = sequences[:, -1, 3:6] # (N,3)
    pad         = np.zeros((N, 1)) # for ecc dim
    pred_padded = np.hstack([pred_scaled, last_vels, pad]) # (N,7)
    pred_unscaled = scaler.inverse_transform(pred_padded)[:, :3]

    true_unscaled = sequences[:, -1, :3]
    errors = np.linalg.norm(true_unscaled - pred_unscaled, axis=1)
    mags   = np.linalg.norm(true_unscaled, axis=1)
    error_pct = np.mean(errors / mags) * 100
    print(f"Mean error: {np.mean(errors):.2f} km, Mean error %: {error_pct:.2f}%")

    # SAVE MODEL AND SCALER
    model.save(f"{model_name}_full_model.keras")
    import joblib
    joblib.dump(scaler, f"{model_name}_scaler.pkl")
    print(f"Saved model to {model_name}_full_model.keras and scaler to {model_name}_scaler.pkl")

    return model, scaler, error_pct

# Main execution
if __name__ == "__main__":
    # Prompt user for flexibility
    num_timesteps = int(input("Enter number of timesteps (e.g., 29): "))
    num_features = int(input("Enter number of features (e.g., 7): "))
    num_samples = int(input("Enter number of TLE samples (e.g., 1000): "))

    print("Downloading TLEs…")
    tles = download_tles(num_samples=num_samples)

    print("Generating sequences…")
    # num_points = num_timesteps + 1 (since you predict the last from the previous)
    sequences = generate_sequences_from_tles(tles, num_points=num_timesteps + 1)

    print(f"Got {len(sequences)} valid satellite sequences.\n")

    print("Training LSTM…")
    lstm_model, lstm_scaler, lstm_err = train_and_evaluate(
        lambda: build_lstm_model(num_timesteps, num_features),
        sequences,
        model_name="lstm"
    )

    print("\nTraining GRU…")
    gru_model, gru_scaler, gru_err = train_and_evaluate(
        lambda: build_gru_model(num_timesteps, num_features),
        sequences,
        model_name="gru"
    )

    if lstm_err < 5 and gru_err < 5:
        print("\nBoth models achieved <5% mean error!")
    else:
        print("\nTry increasing data, epochs, or model size for better accuracy.")

### Helper Functions ###

def read_tles_from_file(filename, num_samples=None, shuffle=True):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    tles = []
    for i in range(0, len(lines) - 2, 3):
        if lines[i].startswith('1 ') or lines[i].startswith('2 '): continue
        tles.append((lines[i], lines[i+1], lines[i+2]))
    if shuffle:
        random.shuffle(tles)
    if num_samples is not None:
        tles = tles[:num_samples]

    return tles

def generate_benchmark_positions(tles, num_points=30, step_sec=30):
    # reusing updated sequence generator
    full_seq = generate_sequences_from_tles(
        tles=tles, num_points=num_points, step_sec=step_sec
    )
    return full_seq[:, :, :3]

def fit_scaler_for_class(sequences, class_name="satellite"):
    """
    Fit and save a MinMaxScaler for a specific object class.
    """
    scaler = MinMaxScaler()
    N, T, F = sequences.shape
    scaler.fit(sequences.reshape(-1, F))
    import joblib
    joblib.dump(scaler, f"{class_name}_scaler.pkl")
    print(f"Saved scaler for {class_name} to {class_name}_scaler.pkl")
    return scaler

def load_xgboost_model(model_name: str):
    import joblib
    model  = joblib.load(f"{model_name}_xgb_model.joblib")
    scaler = joblib.load(f"{model_name}_scaler.pkl")
    return model, scaler

def predict_xgboost_sequence(model, scaler, sequences):
    N, T, F = sequences.shape
    scaler_n_features = scaler.n_features_in_
    if F != scaler_n_features:
        sequences = sequences[..., :scaler_n_features]
        F = scaler_n_features
    flat = sequences.reshape(-1, F)
    scaled = scaler.transform(flat).reshape(N, T, F)
    X = scaled[:, :-1, :].reshape(N, -1) # flatten time for XGBoost
    preds_scaled = model.predict(X) # shape (N, 3)
    # Used the **scaled** last velocities/ecc for inverse_transform:
    last_scaled = scaled[:, -1, 3:] # shape (N, 3) or (N, 4)
    to_inverse = np.hstack([preds_scaled, last_scaled])
    preds = scaler.inverse_transform(to_inverse)[:, :3]
    return preds

def load_trained_model(model_name: str):
    model  = keras.models.load_model(f"{model_name}_full_model.keras")
    import joblib
    scaler = joblib.load(f"{model_name}_scaler.pkl")
    return model, scaler

def predict_sequence(model, scaler, sequences):
    """
    Runs a Keras model on sequences of shape (N, T, F),
    where F is either 6 (pos+vel) or 7 (pos+vel+ecc).
    Inverts the MinMaxScaler back to km and returns an (N,3) array.
    """
    N, T, F = sequences.shape

    # Match scaler's expected feature count
    scaler_n_features = scaler.n_features_in_
    if F != scaler_n_features:
        sequences = sequences[..., :scaler_n_features]
        F = scaler_n_features

    # flatten & scale
    flat        = sequences.reshape(-1, F)
    scaled_flat = scaler.transform(flat)
    scaled_seq  = scaled_flat.reshape(N, T, F)

    # predict on all but last step
    X            = scaled_seq[:, :-1, :]
    preds_scaled = model.predict(X, verbose=0) #(N, 3)

    # re-assemble a full feature vector for inverse_transform:
    last_vels = sequences[:, -1, 3:6] #(N, 3)

    if F == 6:
        to_inverse = np.hstack([preds_scaled, last_vels])
    elif F == 7:
        ecc = sequences[:, -1, 6:7] # (N, 1)
        to_inverse = np.hstack([preds_scaled, last_vels, ecc])
    else:
        raise ValueError(f"predict_sequence: unsupported feature count F={F}")

    # inverse‐scale and return only the XYZ dims
    inv = scaler.inverse_transform(to_inverse)
    return inv[:, :3]

def save_predictions_for_cesium_with_actual(predictions, benchmark_positions, model_name="Model", object_type="satellite", source="TLE"):
    filename = f"predictions_{model_name}.json"
    output = []
    for idx, (pred, real) in enumerate(zip(predictions, benchmark_positions)):
        orbit = [{"x": float(x), "y": float(y), "z": float(z)} for x,y,z in real]
        predicted = {"x": float(pred[0]), "y": float(pred[1]), "z": float(pred[2])}
        output.append({
            "id": f"Satellite {idx}",
            "orbit": orbit,
            "predicted": predicted,
            "metadata": {
                "object_type": object_type,
                "source": source
            }
        })
    with open(filename, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved CesiumJSON → {filename}")

def evaluate_model_performance(preds, actual, object_type="satellite"):
    errors_km  = np.linalg.norm(preds - actual, axis=1)
    radii      = np.linalg.norm(actual, axis=1)
    errors_pct = errors_km / radii * 100
    print(f"{object_type} Mean Error: {np.mean(errors_km):.2f} km, Mean % Error: {np.mean(errors_pct):.2f}%")
    plt.figure(figsize=(16,6))  # <-- Wider plot
    plt.subplot(1,2,1)
    plt.stem(errors_pct, linefmt='b-', markerfmt='o', basefmt=' ')
    plt.title(f"{object_type} Prediction Error (%)")
    plt.xlabel("Index")
    plt.ylabel("Error (%)")
    plt.subplot(1,2,2)
    plt.stem(errors_km, linefmt='r-', markerfmt='x', basefmt=' ')
    plt.title(f"{object_type} Prediction Distance Missed (km)")
    plt.xlabel("Index")
    plt.ylabel("Distance Missed (km)")
    plt.tight_layout()
    plt.show()

def get_random_flyby_neo(start_date="2025-01-01", end_date="2025-12-31", max_dist_au=0.01):
    """
    Query NASA CNEOS API for NEOs with close approaches to Earth.
    Returns a random NEO designation and its close approach date.
    """
    url = "https://ssd-api.jpl.nasa.gov/cad.api"
    params = {
        "date-min": start_date,
        "date-max": end_date,
        "dist-max": max_dist_au,  # AU-->0.01 AU ~ 1.5 million km
        "body": "Earth",
        "sort": "date",
        "limit": 100
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if "data" not in data or not data["data"]:
        raise RuntimeError("No close approaches found in this window.")
    # Each entry is: [des, orbit_id, jd, cd, dist, dist_min, dist_max, v_rel, v_inf, t_sigma_f, h, diameter, diameter_sigma, fullname]
    entry = random.choice(data["data"])
    designation = entry[0]
    close_approach_date = entry[3]  # Ex: '2025-Jun-24 03:20'
    return designation, close_approach_date

def fetch_flyby_neo_positions(neo_name, center_time, window_days=2, step='1h'):
    """
    Fetches the NEO's position relative to Earth (geocentric) from JPL Horizons.
    """
    try:
        dt = datetime.datetime.strptime(center_time, "%Y-%b-%d %H:%M")
        iso_center_time = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        iso_center_time = center_time  # fallback if already in ISO

    # Center window on close approach
    t0 = Time(iso_center_time)
    start_time = (t0 - window_days/2).iso.split()[0]
    stop_time  = (t0 + window_days/2).iso.split()[0]
    obj = Horizons(id=neo_name, location='500',  # 500 = geocenter
                   epochs={'start': start_time, 'stop': stop_time, 'step': step},
                   id_type='designation')
    eph = obj.vectors()
    times = eph['datetime_jd']

    # Convert MaskedColumn to plain numpy arrays, filling masked with nan
    xs = eph['x'].filled(np.nan) if hasattr(eph['x'], 'filled') else np.array(eph['x'], float)
    ys = eph['y'].filled(np.nan) if hasattr(eph['y'], 'filled') else np.array(eph['y'], float)
    zs = eph['z'].filled(np.nan) if hasattr(eph['z'], 'filled') else np.array(eph['z'], float)

    xs = xs * 1.496e8  # AU to km
    ys = ys * 1.496e8
    zs = zs * 1.496e8

    return np.array(times), np.vstack([xs, ys, zs]).T

def fetch_neo_positions(spk_id, epochs_jd):
    """
    Returns an (T,3) array of x,y,z (km) for a small body designation.
    """
    obj = Horizons(
        id=spk_id,
        location='@sun',
        epochs=epochs_jd,
        id_type='designation'
    )
    vec = obj.vectors()

    # Convert MaskedColumn → plain numpy arrays (filling masked with nan):
    x = vec['x'].filled(np.nan) if hasattr(vec['x'], 'filled') else np.array(vec['x'], float)
    y = vec['y'].filled(np.nan) if hasattr(vec['y'], 'filled') else np.array(vec['y'], float)
    z = vec['z'].filled(np.nan) if hasattr(vec['z'], 'filled') else np.array(vec['z'], float)

    pos_au = np.vstack((x, y, z)).T  # shape (T,3)
    return pos_au * 1.4959787e8       # Converts AU → km

def generate_neo_sequences(spk_ids, start_jd, num_points=30, duration_days=1.0):
    """
    For each SPK ID, fetch a num_points‐long 3D track and return
    an array of shape (len(spk_ids), num_points, 7)
    with columns [x,y,z,vx,vy,vz,ecc].
    """
    jds = np.linspace(start_jd, start_jd + duration_days, num_points)
    all_seqs = []
    for spk in spk_ids:
        pos = fetch_neo_positions(spk, jds)   # (num_points, 3)
        # finite‐difference velocity (km/day → km/sec):
        dt = (jds[1] - jds[0]) * 86400
        vel = np.vstack((
            np.gradient(pos[:,0], dt),
            np.gradient(pos[:,1], dt),
            np.gradient(pos[:,2], dt)
        )).T
        ecc = np.full((num_points,1), np.nan)  # placeholder
        seq7 = np.hstack((pos, vel, ecc))
        all_seqs.append(seq7)
    return np.stack(all_seqs)  # shape (len(spk_ids), num_points, 7)

def auto_generate_flyby_neo_czml():
    """
    Selects a random close-approach NEO, fetches its flyby ephemeris,
    and writes a CZML file for Cesium visualization.
    """
    neo_name, ca_date = get_random_flyby_neo()
    print(f"Selected NEO: {neo_name} (close approach: {ca_date})")
    times_jd, positions_km = fetch_flyby_neo_positions(neo_name, ca_date)
    write_flyby_neo_czml(neo_name, times_jd, positions_km, filename="neos.czml")

def write_flyby_neo_czml(neo_name, times_jd, positions_km, filename="neos.czml"):
    cart = []
    for jd, (x, y, z) in zip(times_jd, positions_km):
        t_iso = Time(jd, format="jd").iso.replace(' ', 'T') + "Z"
        cart += [t_iso, x * 1000, y * 1000, z * 1000]  # km → m

    czml = [{
        "id": "document",
        "name": "NEO CZML",
        "version": "1.0"
    }, {
        "id": f"neo_{neo_name}",
        "availability": f"{cart[0]}/{cart[-4]}",
        "position": {
            "interpolationAlgorithm": "LAGRANGE",
            "interpolationDegree": 5,
            "referenceFrame": "INERTIAL",
            "cartesian": cart
        },
        "point": {
            "pixelSize": 16,
            "color": {"rgba": [255, 0, 255, 255]}
        },
        "path": {
            "material": {"solidColor": {"color": {"rgba": [255, 0, 255, 200]}}},
            "width": 3,
            "leadTime": 0,
            "trailTime": 86400.0,
            "resolution": 120
        },
        "label": {
            "text": f"NEO {neo_name}",
            "font": "14px sans-serif",
            "fillColor": {"rgba": [255, 255, 255, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 255]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "horizontalOrigin": "CENTER",
            "verticalOrigin": "CENTER"
        }
    }]
    import json
    with open(filename, "w") as f:
        json.dump(czml, f, indent=2)
    print(f"Wrote flyby NEO {neo_name} to {filename}")