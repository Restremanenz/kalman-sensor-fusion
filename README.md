# Speed-Climbing Sensor Fusion (ESKF)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Evaluation%20Phase-success)
![Hardware](https://img.shields.io/badge/Sensors-IMU%20%2B%20Barometer-orange)

This repository contains the code for : A high-precision, dead-reckoning based climbing tracking system. It uses a **15/18-State Error-State Kalman Filter (ESKF)** to calculate the 3D trajectory of a speed climber exclusively using IMU and barometer data—completely without GPS.

---

##  Key Features & Mathematical Concepts

This project implements industry standards for Inertial Navigation Systems (INS):

* **ESKF Architecture:** Estimation of position, velocity, orientation error (quaternions), accelerometer bias, and gyroscope bias.
* **ZUPT & ZARU (Zero Velocity / Angular Rate Updates):** Heuristic stillness detection to eliminate accumulated IMU drift by evaluating sensor data during the brief rest phases between climbing moves.
* **Numerical Stability:** Implementation of the **Joseph form** for the covariance update and continuous matrix symmetrization to prevent floating-point rounding errors.
* **Dynamic Bias Calibration:** The *run-to-run / turn-on bias* of the gyroscope is dynamically recalculated during the static initialization phase before every single run.
* **Boundary-Condition Smoother:** A backward smoother (post-processing) that enforces physical boundary conditions (e.g., start and end points on the wall) and eliminates systematic scale errors (*gravity leakage*).

---

## 📂 Software Architecture (Clean Architecture)

The codebase is highly modular and pipeline-driven:

* `config.py` - Central configuration file. All noise parameters (R, Q matrices) are defined here based on physical hardware datasheet values.
* `preprocessing.py` - Hardware synchronization, resampling (Pandas `merge_asof`), zero-phase Butterworth filtering (barometer), and static start detection.
* `filters.py` - The mathematical core. Contains the `FilterpyESKF` class with prediction and robust update steps.
* `Kalman.py` - The main pipeline for processing individual datasets, including 3D trajectory visualization.
* `batch_eval.py` - The statistical evaluation tool for the thesis (Return-to-Origin tests, 1D/3D translation tests) including automated plot generation (lollipop plots, target / scatter metrics).

---

## 🛠 Hardware & Sensors

The algorithms are optimized for STMicroelectronics sensors:
* **IMU:** LSM6DSV16X (6-axis, Accelerometer & Gyroscope)
* **Barometer:** LPS22DF (High-Precision, 24-bit)

---

## 🚀 Installation & Usage

1. **Install Dependencies:**
   Ensure Python 3.8+ is installed. Install the required packages via pip:
   ```bash
   pip install numpy pandas matplotlib scipy filterpy
