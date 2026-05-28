import os
import sys
import json
import numpy as np
import pandas as pd
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

    df_acc = df_acc.sort_values("Time")
    df_gyro = df_gyro.sort_values("Time")
    df_imu = pd.merge_asof(df_acc, df_gyro, on="Time", direction="nearest")
    df_imu = df_imu.reset_index(drop=True)
    
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
            
    else:
        print("Automatische Initialisierung DEAKTIVIERT. Starte ab Beginn.")
        init_end_idx = 1 
        q_init = R.from_quat([0, 0, 0, 1]) 

    # Kalman Filter Start 
    eskf = FilterpyESKF15( 
        initial_pos = [0.0, 0.0, 0.0], 
        initial_q = q_init, 
        gyro_noise_std = calib.gyro_noise_std,
        accel_noise = config.ACCEL_NOISE_DENSITY,
        bg_rw = config.GYRO_BIAS_RW, 
        ba_rw = config.ACCEL_BIAS_RW, 
        grav_unc = config.GRAVITY_UNCERTAINTY,
        zupt_unc = config.ZUPT_UNCERTAINTY
    )
    
    positions = []
    orientations = [] 
    times = df_imu["Time"].values
    velocities = []
    times_plot = []
    
    # 4. Koppelnavigation
    for i in range(init_end_idx, len(df_imu)):
        dt = times[i] - times[i-1]
        if dt <= 0: continue
            
        row = df_imu.iloc[i]
        
        raw_acc = np.array([row['A_x [g]'], row['A_y [g]'], row['A_z [g]']]) 
        raw_gyro = np.array([row['G_x [dps]'], row['G_y [dps]'], row['G_z [dps]']]) 
        
        acc_calib = calib.calibrate_acc(raw_acc)
        gyro_calib = calib.calibrate_gyro(raw_gyro)
        
        eskf.predict(acc_calib, gyro_calib, dt)
        
        acc_world = eskf.q.apply(acc_calib)
        linear_acc = acc_world + eskf.g
        acc_magnitude = np.linalg.norm(linear_acc)
        
        if acc_magnitude < config.ZUPT_THRESHOLD_MS2:
            eskf.update_zupt()
            eskf.update_gravity(acc_calib)
            
        positions.append(eskf.p.copy())
        orientations.append(eskf.q)
        velocities.append(eskf.v.copy())
        times_plot.append(times[i])
        
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

    visualizer_main.show_all()
    print(eskf.bg)
    print(eskf.ba)

if __name__ == "__main__":
    main()