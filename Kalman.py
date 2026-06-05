import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import config  
from st_log_reader import STLogReader
from preprocessing import IMUPreprocessor
from postprocessing import PostProcessor
from visualization import TrajectoryVisualizer
from filters import FilterpyESKF15


class IMUCalibration:
    """Lädt die ST-Kalibrierungsdaten und wendet sie an."""
    def __init__(self, acc_json_path, gyro_json_path):
        with open(acc_json_path, 'r') as f:
            acc_data = json.load(f)
            self.acc_bias = np.array(acc_data["offset_b"])
            self.acc_M = np.array(acc_data["matrix_M"])
            
        with open(gyro_json_path, 'r') as f:
            gyro_data = json.load(f)
            self.gyro_bias = np.array(gyro_data["offset_b"])
            self.gyro_noise_std = np.array(gyro_data["noise_std"]) * (np.pi / 180.0)

    def calibrate_acc(self, raw_acc):
        calibrated = self.acc_M @ (raw_acc - self.acc_bias)
        return calibrated * 9.81

    def calibrate_gyro(self, raw_gyro):
        calibrated_dps = raw_gyro - self.gyro_bias
        return calibrated_dps * (np.pi / 180.0)


def run_eskf_pipeline(df_imu, q_init, init_idx, calib, fs_dynamisch):
    """Führt die komplette Koppelnavigation (Prediction & Update) durch."""
    print("Starte Error-State Kalman Filter...")
    
    eskf = FilterpyESKF15( 
        initial_pos = [0.0, 0.0, 0.0], 
        initial_q = q_init, 
        gyro_noise_std = calib.gyro_noise_std,
        accel_noise = config.ACCEL_NOISE_DENSITY,
        bg_rw = config.GYRO_BIAS_RW, 
        ba_rw = config.ACCEL_BIAS_RW, 
        grav_unc = config.GRAVITY_UNCERTAINTY,
        zupt_unc = config.ZUPT_UNCERTAINTY,
        baro_unc = config.BARO_UNCERTAINTY
    )
    
    positions, orientations, velocities, times_plot = [], [], [], []
    times = df_imu["Time"].values
    last_baro_time = -1.0 
    
    for i in range(init_idx, len(df_imu)):
        dt = times[i] - times[i-1]
        if dt <= 0: continue
            
        row = df_imu.iloc[i]
        
        # 1. Daten holen & kalibrieren
        raw_acc = np.array([row['A_x [g]'], row['A_y [g]'], row['A_z [g]']]) 
        raw_gyro = np.array([row['G_x [dps]'], row['G_y [dps]'], row['G_z [dps]']]) 
        acc_calib = calib.calibrate_acc(raw_acc)
        gyro_calib = calib.calibrate_gyro(raw_gyro)
        
        # 2. ESKF Prediction (Koppelnavigation)
        eskf.predict(acc_calib, gyro_calib, dt)
        
        # 3. Barometer Update (Asynchroner Trigger)
        current_baro_time = row['Baro_Time']
        if pd.notna(current_baro_time) and current_baro_time != last_baro_time:
            eskf.update_barometer(row['Altitude_filt [m]'])
            last_baro_time = current_baro_time

        # 4. ZUPT & Gravity Update
        acc_world = eskf.q.apply(acc_calib)
        acc_magnitude = np.linalg.norm(acc_world + eskf.g)
        
        if getattr(config, 'USE_ZUPT', False):
            if acc_magnitude < config.ZUPT_THRESHOLD_MS2:
                eskf.update_zupt()
                eskf.update_gravity(acc_calib)

        # 5. Speichern
        positions.append(eskf.p.copy())
        orientations.append(eskf.q)
        velocities.append(eskf.v.copy())
        times_plot.append(times[i])

    return np.array(positions), np.array(velocities), orientations, np.array(times_plot), eskf


def main():
    # ==============================================================
    # 1. SETUP & DATEN LADEN
    # ==============================================================
    reader = STLogReader(config.LOG_FOLDER)
    calib = IMUCalibration(config.ACCEL_CALIB_FILE, config.GYRO_CALIB_FILE)
    preprocessor = IMUPreprocessor(config)
    
    df_imu, fs_dynamisch = preprocessor.load_and_merge_data(reader)

    # ==============================================================
    # 2. PRE-PROCESSING (Initialisierung & Barometer)
    # ==============================================================
    if config.USE_AUTO_INIT:
        df_init, init_idx = preprocessor.find_initial_stillness(df_imu, fs_dynamisch)
        q_init, P0 = preprocessor.initialize_run(df_init, calib)
    else:
        print("WARNUNG: Auto-Init deaktiviert. Starte unkalibriert.")
        init_idx, P0, q_init = 1, df_imu['P [hPa]'].iloc[0], R.from_quat([0,0,0,1])

    # Höhenberechnung, Zero-Phase Filter, Tare & Data-Cropping
    df_imu = preprocessor.process_barometer_and_crop(df_imu, P0, init_idx, fs_dynamisch)

    # ==============================================================
    # 3. KOPPELNAVIGATION (15-State ESKF)
    # ==============================================================
    positions, velocities, orientations, times, eskf = run_eskf_pipeline(
        df_imu, q_init, init_idx, calib, fs_dynamisch
    )

    # ==============================================================
    # 4. POST-PROCESSING (Boundary Condition Smoother)
    # ==============================================================
    if getattr(config, 'USE_SMOOTHER', False):
        postprocessor = PostProcessor(config)
        positions, velocities = postprocessor.apply_smoother(
            positions, velocities, times, df_imu
        )

    # ==============================================================
    # 5. VISUALISIERUNG
    # ==============================================================
    print("Bereite finale 3D-Plots vor...")
    vis = TrajectoryVisualizer(animation_fps=config.ANIMATION_FPS)
    
    vis.plot_static_trajectory(positions)
    vis.plot_animated_trajectory(positions, orientations, fs_dynamisch)

    if getattr(config, 'SHOW_VELOCITY', False):
        vis.plot_velocity(times, velocities)
    if getattr(config, 'SHOW_RAW_SENSOR_DATA', False):
        vis.plot_raw_sensor_data(df_imu)
    if getattr(config, 'SHOW_ALTITUDE', False):
        vis.plot_altitude(df_imu)
    if getattr(config, 'SHOW_FILTER_TUNING', False) and getattr(config, 'USE_PRE_FILTER', False):
        vis.plot_raw_vs_filtered(df_imu, axis='z', zoom_start=0.0, zoom_end=10.0)

    vis.show_all()
    
    print(f"\n[INFO] Finaler In-Run Bias Schätzwert:")
    print(f"Gyro Bias (bg): {eskf.bg}")
    print(f"Accel Bias (ba): {eskf.ba}")

if __name__ == "__main__":
    main()