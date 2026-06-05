import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

import config  
from st_log_reader import STLogReader
from visualization import TrajectoryVisualizer
from filters import FilterpyESKF15

class IMUCalibration:
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

def apply_zero_phase_filter(data, cutoff, fs, order=4):
    """Wendet einen Zero-Phase Butterworth Low-Pass Filter an, optimiert für IMU-Transienten."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Verhindert das "Ringing" am Anfang und Ende des Datensatzes
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    return filtfilt(b, a, data, padtype='even', padlen=padlen)

def main():
    # 1. Daten laden (Pfade kommen aus der Config)
    reader = STLogReader(config.LOG_FOLDER)
    calib = IMUCalibration(config.ACCEL_CALIB_FILE, config.GYRO_CALIB_FILE)
    
    gyro_info = reader.get_sensor_info("lsm6dsv16x_gyro")
    if gyro_info is None:
        print("FEHLER: Konnte Gyro-Metadaten nicht auslesen.")
        sys.exit(1)
        
    fs_dynamisch = gyro_info.get("measured_odr_hz")
    if fs_dynamisch is None:
        fs_dynamisch = 960.0
    
    df_acc = reader.get_sensor_data("lsm6dsv16x_acc")
    df_gyro = reader.get_sensor_data("lsm6dsv16x_gyro")

    # 1.5 Barometer Daten laden
    baro_info = reader.get_sensor_info("lps22df_press")
    fs_baro = baro_info.get("measured_odr_hz", 25.0) if baro_info else 25.0
    print(fs_baro)
    
# Alle Sensordaten laden und direkt nach Zeit sortieren
    df_acc = reader.get_sensor_data("lsm6dsv16x_acc").sort_values("Time")
    df_gyro = reader.get_sensor_data("lsm6dsv16x_gyro").sort_values("Time")
    df_baro = reader.get_sensor_data("lps22df_press").sort_values("Time")

    # 1.5 Barometer Setup
    baro_info = reader.get_sensor_info("lps22df_press")
    fs_baro = baro_info.get("measured_odr_hz", 25.0) if baro_info else 25.0
    print(f"Barometer ODR erkannt: {fs_baro} Hz")

    # Automatische Spaltenerkennung (Löst das Problem mit 'PRESS' vs 'Press')
    original_press_col = [col for col in df_baro.columns if col != 'Time'][0]
    df_baro = df_baro.rename(columns={original_press_col: 'PRESS [hPa]'})
    
    # Trigger-Timestamp für den Kalman Filter speichern
    df_baro['Baro_Time'] = df_baro['Time'] 
    
    # --- MERGING (Streng nacheinander!) ---
    # 1. Accel und Gyro synchronisieren
    df_imu = pd.merge_asof(df_acc, df_gyro, on="Time", direction="nearest")
    
    # 2. Barometer asynchron anhängen
    df_imu = pd.merge_asof(df_imu, df_baro, on="Time", direction="backward")
    df_imu = df_imu.reset_index(drop=True)
    # --------------------------------------
    
    # 2. Initialisierung (Steuerung über Config)
    if config.USE_AUTO_INIT:
        print("Suche nach initialer Ruhephase...")
        
        MIN_STILL_SAMPLES = int(fs_dynamisch * config.MIN_STILL_SECONDS) 
        
        gyro_mag = np.sqrt(df_imu['G_x [dps]']**2 + df_imu['G_y [dps]']**2 + df_imu['G_z [dps]']**2)
        is_still = gyro_mag < config.STILLNESS_THRESHOLD
        block_ids = (is_still != is_still.shift()).cumsum()
        
        df_init = None
        init_end_idx = 0
        
        for block_id, group in df_imu[is_still].groupby(block_ids):
            if len(group) >= MIN_STILL_SAMPLES:
                margin = int(len(group) * 0.1)
                core_group = group.iloc[margin:-margin]
                
                df_init = core_group
                init_end_idx = core_group.index[-1]
                break
                
        if df_init is None:
            print("FEHLER: Keine ausreichend lange Ruhephase am Start gefunden.")
            sys.exit(1)

        plot_end_idx = min(len(df_imu), int(init_end_idx + 2 * fs_dynamisch))
        
        visualizer_init = TrajectoryVisualizer()
        visualizer_init.plot_auto_init(
            df_plot=df_imu.iloc[:plot_end_idx],
            gyro_mag_plot=gyro_mag.iloc[:plot_end_idx],
            df_init=df_init,
            gyro_mag_init=gyro_mag.loc[df_init.index],
            threshold=config.STILLNESS_THRESHOLD
        )

        raw_gyros_init = df_init[['G_x [dps]', 'G_y [dps]', 'G_z [dps]']].values
        calib.gyro_bias = np.mean(raw_gyros_init, axis=0)

        raw_accs_init = df_init[['A_x [g]', 'A_y [g]', 'A_z [g]']].values
        accs_init_calib = np.array([calib.calibrate_acc(a) for a in raw_accs_init])
        mean_acc = np.mean(accs_init_calib, axis=0)
        
        v1 = mean_acc / np.linalg.norm(mean_acc)
        v2 = np.array([0.0, 0.0, 1.0])
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            q_init = R.from_rotvec(axis * angle)
        else:
            q_init = R.from_quat([0,0,0,1])

    # Basisdruck P0 aus der Ruhephase berechnen
        P0 = df_init['PRESS [hPa]'].mean()
        print(f"Kalibrierter Basisdruck P0: {P0:.2f} hPa")

    else:
        print("Automatische Initialisierung DEAKTIVIERT. Starte ab Beginn.")
        init_end_idx = 1 
        q_init = R.from_quat([0, 0, 0, 1]) 
        P0 = df_imu['PRESS [hPa]'].iloc[0] 
        
    # =========================================================
    # BAROMETER: HÖHENBERECHNUNG & ZERO-PHASE FILTER
    # =========================================================
    # Internationale barometrische Höhenformel (relativ zu P0)
    df_imu['Altitude [m]'] = 44330.0 * (1.0 - (df_imu['PRESS [hPa]'] / P0)**(1 / 5.255))
    
    # Filtern des Höhensignals (Extrem wichtig für saubere Z-Updates!)
    if config.USE_BARO_PRE_FILTER:
        print(f"Wende Zero-Phase Filter auf Barometer an (Cutoff: {config.BARO_CUTOFF_HZ}Hz)...")
        # Wir filtern direkt die resultierende Höhenkurve
        df_imu['Altitude_filt [m]'] = apply_zero_phase_filter(
            df_imu['Altitude [m]'].values, 
            cutoff=config.BARO_CUTOFF_HZ, 
            fs=fs_dynamisch, 
            order=2
        )
    else:
        df_imu['Altitude_filt [m]'] = df_imu['Altitude [m]']

    # Wir zwingen die Starthöhe am Beginn des Koppelnavigations-Loops exakt auf 0.0m
    start_altitude = df_imu['Altitude_filt [m]'].iloc[init_end_idx]
    df_imu['Altitude_filt [m]'] -= start_altitude
    df_imu['Altitude [m]'] -= start_altitude # (Optional: auch Rohdaten für den Plot nullen)

    if getattr(config, 'MAX_PROCESS_TIME', None) is not None:
        start_time_sec = df_imu['Time'].iloc[init_end_idx]
        max_time_sec = start_time_sec + config.MAX_PROCESS_TIME
        
        # Schneidet alles ab, was zeitlich nach unserer Ziellänge kommt
        df_imu = df_imu[df_imu['Time'] <= max_time_sec]


    # Kalman Filter Start 
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
    
    positions = []
    orientations = [] 
    times = df_imu["Time"].values
    velocities = []
    times_plot = []
    
# 4. Koppelnavigation
    last_baro_time = -1.0 # Trigger-Variable
    
    for i in range(init_end_idx, len(df_imu)):
        dt = times[i] - times[i-1]
        if dt <= 0: continue
            
        row = df_imu.iloc[i]
        
        raw_acc = np.array([row['A_x [g]'], row['A_y [g]'], row['A_z [g]']]) 
        raw_gyro = np.array([row['G_x [dps]'], row['G_y [dps]'], row['G_z [dps]']]) 
        
        acc_calib = calib.calibrate_acc(raw_acc)
        gyro_calib = calib.calibrate_gyro(raw_gyro)
        
        # 1. Koppelnavigation (Prediction)
        eskf.predict(acc_calib, gyro_calib, dt)
        
        # 2. BAROMETER UPDATE (Asynchroner Trigger)
        current_baro_time = row['Baro_Time']
        if pd.notna(current_baro_time) and current_baro_time != last_baro_time:
            baro_z = row['Altitude_filt [m]']
            eskf.update_barometer(baro_z)
            last_baro_time = current_baro_time

        # 3. ZUPT & GRAVITY UPDATE
        acc_world = eskf.q.apply(acc_calib)
        linear_acc = acc_world + eskf.g
        acc_magnitude = np.linalg.norm(linear_acc)
        
        if config.USE_ZUPT:
            if acc_magnitude < config.ZUPT_THRESHOLD_MS2:
                eskf.update_zupt()
                eskf.update_gravity(acc_calib)

        # 4. Daten für Plotting speichern
        positions.append(eskf.p.copy())
        orientations.append(eskf.q)
        velocities.append(eskf.v.copy())
        times_plot.append(times[i])

    # ---------------------------------------------------------
    #  BOUNDARY CONDITION SMOOTHER 
    # ---------------------------------------------------------
    if getattr(config, 'USE_SMOOTHER', False):
        print("Wende physikalischen Rückwärts-Smoother an...")
        num_samples = len(velocities)
        
        # Lineare Faktoren von 0.0 (Start) bis 1.0 (Ende)
        drift_factors = np.linspace(0.0, 1.0, num_samples)[:, np.newaxis]

        smoothed_velocities = velocities.copy()
        smoothed_positions = positions.copy()

        # --- A. GESCHWINDIGKEIT (Nur für komplette Runs!) ---
        if getattr(config, 'FORCE_V_END_ZERO', False):
            print(" -> Korrigiere End-Geschwindigkeit auf 0.0 m/s")
            v_end_error = velocities[-1] - np.array([0.0, 0.0, 0.0])
            smoothed_velocities = velocities - (drift_factors * v_end_error)
            
            # Da sich die Geschwindigkeit geändert hat, müssen wir die Route neu integrieren
            smoothed_positions[0] = positions[0]
            dt_array = np.diff(times_plot)
            for i in range(1, num_samples):
                smoothed_positions[i] = smoothed_positions[i-1] + smoothed_velocities[i] * dt_array[i-1]

        # --- B. POSITION (Für alle Runs, zieht Track physikalisch korrekt an die Wand) ---
        target_pos = smoothed_positions[-1].copy() # Default: Nichts ändern

        if getattr(config, 'SMOOTH_XY_TO_ZERO', False):
            target_pos[0] = getattr(config, 'TARGET_X_M', 0.0)
            target_pos[1] = getattr(config, 'TARGET_Y_M', 0.0)
            print(f" -> Korrigiere End-Position X/Y auf [{target_pos[0]:.2f}, {target_pos[1]:.2f}]")
        
        if getattr(config, 'SMOOTH_TO_BARO_Z', False):
            # Holt sich dynamisch die absolut letzte Barometer-Höhe aus dem Datensatz!
            baro_end_z = df_imu['Altitude_filt [m]'].iloc[-1]
            target_pos[2] = baro_end_z
            print(f" -> Korrigiere End-Position Z auf Barometer-Höhe: {baro_end_z:.2f}m")

        # Differenz (Fehler) zwischen ESKF-Ende und unserem physikalischen Ziel berechnen
        pos_end_error = smoothed_positions[-1] - target_pos
        
        # Fehler linear über den gesamten Datensatz rückwärts abziehen
        smoothed_positions -= (drift_factors * pos_end_error)

        # Arrays für das finale Plotting überschreiben
        positions = smoothed_positions
        velocities = smoothed_velocities
    # ---------------------------------------------------------   
        
    # 5. Visualisierung
    positions = np.array(positions)
    velocities = np.array(velocities) 
    times_plot = np.array(times_plot) 

    print("Bereite finale 3D-Plots vor...")
    
    visualizer_main = TrajectoryVisualizer(animation_fps=config.ANIMATION_FPS)
    visualizer_main.plot_static_trajectory(positions)
    visualizer_main.plot_animated_trajectory(positions, orientations, fs_dynamisch)

    if config.SHOW_VELOCITY:
        visualizer_main.plot_velocity(times_plot, velocities)

    if config.SHOW_RAW_SENSOR_DATA:
        visualizer_main.plot_raw_sensor_data(df_imu)

    if getattr(config, 'SHOW_ALTITUDE', False):
        visualizer_main.plot_altitude(df_imu)

    visualizer_main.show_all()
    print(eskf.bg)
    print(eskf.ba)

if __name__ == "__main__":
    main()