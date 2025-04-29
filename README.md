# DFSforSTMusingAI
**Data Fusion System for Space Traffic Management Using Artificial Intelligence**

This repository contains the source code and results for my Master's Capstone project at Full Sail University. The project aims to develop an AI-powered system that predicts future satellite positions based on real orbital telemetry data.

By leveraging Long Short-Term Memory (LSTM) neural networks and Two-Line Element (TLE) datasets, this system forecasts satellite trajectories and visualizes predictions using both 2D and interactive 3D plots. The final goal is to support global space traffic management and reduce the risk of satellite collisions in orbit.

---

## 📌 Features

- 🔭 Ingests real TLE satellite data (e.g., from CelesTrak)
- 🧠 Trains an LSTM model using sliding windows and noise augmentation
- 🎯 Predicts next-step satellite positions (X, Y, Z)
- 📉 Calculates and visualizes prediction error (in % and km)
- 🌍 Renders satellite orbits in 3D using Plotly
- 📁 Saves results to `trackingresults.csv` and JSON for Cesium integration

---

## 📂 Files

- `STMS_m1.ipynb` – Main Jupyter notebook with full data processing and model training
- `lstm_weights.weights.h5` – Trained LSTM model weights
- `trackingresults.csv` – Benchmark vs predicted positions and error metrics
- `predictions.json` – Output format for external 3D viewers (e.g., CesiumJS)

---

## 🚀 Getting Started

### Requirements
- Python 3.10+
- Jupyter Notebook
- TensorFlow / Keras
- Pandas / NumPy
- SGP4
- Plotly
- Basemap (optional for 2D visualization)

### How to Run
1. Load the notebook in Jupyter
2. Choose a TLE data file (e.g., `gp.txt`)
3. Input number of satellites, epochs, and batch size
4. Run training + prediction
5. Visualize the predictions and error charts

---

## 🎓 Author

**Corban Czap**  
Computer Science Master's Program  
Full Sail University  
Contact: cjczap@student.fullsail.edu

---

## 📜 License

This project is for academic use under the Full Sail University capstone guidelines. Do not redistribute without permission.

