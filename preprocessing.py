import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from visualization import TrajectoryVisualizer

def apply_zero_phase_filter(data, cutoff, fs, order=4, btype='low'):
    """Wendet einen Zero-Phase Butterworth Filter (Low-Pass oder High-Pass) an."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False) 
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    return filtfilt(b, a, data, padtype='even', padlen=padlen)


class IMUPreprocessor:
    """Kapselt das gesamte Data-Wrangling und die Initialisierung vor dem Kalman Filter."""
    
    def __init__(self, config):
        self.config = config

    def load_and_merge_data(self, reader):
        """Lädt alle Sensoren, gleicht die Frequenzen an und merged die DataFrames."""
        print("Lade und synchronisiere Sensordaten...")
        df_acc = reader.get_sensor_data("lsm6dsv16x_acc").sort_values("Time")
        df_gyro = reader.get_sensor_data("lsm6dsv16x_gyro").sort_values("Time")
        df_baro = reader.get_sensor_data("lps22df_press").sort_values("Time")
        df_mag = reader.get_sensor_data("lis2mdl_mag").sort_values("Time")

        # Auslesen der echten Frequenzen
        gyro_info = reader.get_sensor_info("lsm6dsv16x_gyro")
        fs_dynamisch = gyro_info.get("measured_odr_hz", 960.0) if gyro_info else 960.0

        baro_info = reader.get_sensor_info("lps22df_press")
        fs_baro = baro_info.get("measured_odr_hz", 25.0) if baro_info else 25.0
        
        mag_info = reader.get_sensor_info("lis2mdl_mag")
        fs_mag = mag_info.get("measured_odr_hz", 100.0) if mag_info else 100.0
        
        print(f" -> Sensor-Raten: IMU @ {fs_dynamisch:.1f} Hz | Baro @ {fs_baro:.1f} Hz | Mag @ {fs_mag:.1f} Hz")

        # Automatische Barometer-Spaltenerkennung
        original_press_col = [col for col in df_baro.columns if col != 'Time'][0]
        df_baro = df_baro.rename(columns={original_press_col: 'P [hPa]'})
        df_baro['Baro_Time'] = df_baro['Time'] 

        df_mag = reader.get_sensor_data("lis2mdl_mag").sort_values("Time")
        df_mag['Mag_Time'] = df_mag['Time'] 

        # Pandas Merge-Magie (Dual-Path Synchronisation)
        df_imu = pd.merge_asof(df_acc, df_gyro, on="Time", direction="nearest")
        df_imu = pd.merge_asof(df_imu, df_baro, on="Time", direction="backward")
        df_imu = pd.merge_asof(df_imu, df_mag, on="Time", direction="nearest")

        # Schließt Lücken am Rand des Datensatzes vor dem Filtern
        df_imu = df_imu.ffill().bfill()
        
        return df_imu.reset_index(drop=True), fs_dynamisch

    def find_initial_stillness(self, df_imu, fs_dynamisch):
        """Sucht die erste absolute Ruhephase des Sensors zur Kalibrierung."""
        print("Suche nach initialer Ruhephase...")
        min_still_samples = int(fs_dynamisch * self.config.MIN_STILL_SECONDS) 
        
        gyro_mag = np.sqrt(df_imu['G_x [dps]']**2 + df_imu['G_y [dps]']**2 + df_imu['G_z [dps]']**2)
        is_still = gyro_mag < self.config.STILLNESS_THRESHOLD
        block_ids = (is_still != is_still.shift()).cumsum()
        
        df_init = None
        init_end_idx = 0
        
        for block_id, group in df_imu[is_still].groupby(block_ids):
            if len(group) >= min_still_samples:
                margin = int(len(group) * 0.1)
                core_group = group.iloc[margin:-margin]
                df_init = core_group
                init_end_idx = core_group.index[-1]
                break
                
        if df_init is None:
            print("FEHLER: Keine ausreichend lange Ruhephase am Start gefunden.")
            sys.exit(1)

        if getattr(self.config, 'SHOW_INIT_PLOT', True):
            plot_end_idx = min(len(df_imu), int(init_end_idx + 2 * fs_dynamisch))
            visualizer_init = TrajectoryVisualizer()
            visualizer_init.plot_auto_init(
                df_plot=df_imu.iloc[:plot_end_idx],
                gyro_mag_plot=gyro_mag.iloc[:plot_end_idx],
                df_init=df_init,
                gyro_mag_init=gyro_mag.loc[df_init.index],
                threshold=self.config.STILLNESS_THRESHOLD
            )
        
        return df_init, init_end_idx

    def initialize_run(self, df_init, calib):
        # 1. Gyro Bias berechnen
        raw_gyros_init = df_init[['G_x [dps]', 'G_y [dps]', 'G_z [dps]']].values
        calib.gyro_bias = np.mean(raw_gyros_init, axis=0)

        # 2. Beschleunigung auswerten (Schwerkraft-Vektor)
        raw_accs_init = df_init[['A_x [g]', 'A_y [g]', 'A_z [g]']].values
        accs_init_calib = np.array([calib.calibrate_acc(a) for a in raw_accs_init])
        mean_acc = np.mean(accs_init_calib, axis=0)
        v_acc_body = mean_acc / np.linalg.norm(mean_acc)

        
        if self.config.USE_MAGNETOMETER and 'M_x [G]' in df_init.columns:
            print(" -> Initialisierung: Nutze Beschleunigung & Magnetometer für echtes Start-Heading.")
            raw_mags_init = df_init[['M_x [G]', 'M_y [G]', 'M_z [G]']].values
            mags_init_calib = np.array([calib.calibrate_mag(m) for m in raw_mags_init])
            mean_mag = np.mean(mags_init_calib, axis=0)
            v_mag_body = mean_mag / np.linalg.norm(mean_mag)

            g_global_expected = np.array([0.0, 0.0, 1.0])
            mag_global_expected = np.array(self.config.GLOBAL_MAG_REF)
            mag_global_expected /= np.linalg.norm(mag_global_expected)

            # Wahba's Problem lösen (Kombination aus Acc und Mag)
            q_init, _ = R.align_vectors(
                a=[g_global_expected, mag_global_expected],
                b=[v_acc_body, v_mag_body],
                weights=[1.0, 0.2]
            )
        else:
            print(" -> Initialisierung: Magnetometer DEAKTIVIERT. Nutze reine Wasserwaage (Yaw = 0).")
            # Reine Schwerkraft-Ausrichtung (Exakt wie in deinem allerersten Code!)
            v1 = v_acc_body
            v2 = np.array([0.0, 0.0, 1.0])
            axis = np.cross(v1, v2)
            axis_norm = np.linalg.norm(axis)
            
            if axis_norm > 1e-6:
                axis = axis / axis_norm
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                q_init = R.from_rotvec(axis * angle)
            else:
                q_init = R.from_quat([0,0,0,1])

        # 5. Barometer Basisdruck
        P0 = df_init['P [hPa]'].mean()
        print(f" -> Kalibrierter Basisdruck P0: {P0:.2f} hPa")
        
        return q_init, P0

    def process_barometer_and_crop(self, df_imu, P0, init_end_idx, fs_dynamisch):
        """Berechnet die Höhe, wendet Zero-Phase Filter an, tariert und kürzt ggf. den Datensatz."""
        # 1. Höhe berechnen
        df_imu['Altitude [m]'] = 44330.0 * (1.0 - (df_imu['P [hPa]'] / P0)**(1 / 5.255))
        
        # 2. Filtern
        if self.config.USE_BARO_PRE_FILTER:
            df_imu['Altitude_filt [m]'] = apply_zero_phase_filter(
                df_imu['Altitude [m]'].values, 
                cutoff=self.config.BARO_CUTOFF_HZ, 
                fs=fs_dynamisch, 
                order=2
            )
        else:
            df_imu['Altitude_filt [m]'] = df_imu['Altitude [m]']

        # 3. Tare (Nullabgleich am Startpunkt)
        start_altitude = df_imu['Altitude_filt [m]'].iloc[init_end_idx]
        df_imu['Altitude_filt [m]'] -= start_altitude
        df_imu['Altitude [m]'] -= start_altitude 
        
        # 4. Datensatz abschneiden (Debugging)
        if getattr(self.config, 'MAX_PROCESS_TIME', None) is not None:
            start_time_sec = df_imu['Time'].iloc[init_end_idx]
            max_time_sec = start_time_sec + self.config.MAX_PROCESS_TIME
            df_imu = df_imu[df_imu['Time'] <= max_time_sec]
            print(f" -> DEBUG-MODUS: Datensatz auf {self.config.MAX_PROCESS_TIME}s gekürzt!")

        # ROBUSTE OFFLINE-STILLSTANDSERKENNUNG   
        # 1. Betrag der Beschleunigung berechnen
        acc_mag = np.sqrt(df_imu['A_x [g]']**2 + df_imu['A_y [g]']**2 + df_imu['A_z [g]']**2)
        
        # 2. Hochpass-Filter (nutzt nun die Variable aus config)
        acc_hp = apply_zero_phase_filter(
            acc_mag.values, 
            cutoff=self.config.OFFLINE_ZUPT_HP_CUTOFF, 
            fs=fs_dynamisch, 
            order=1, 
            btype='high'
        )
        
        # 3. Absolutbetrag bilden
        acc_rectified = np.abs(acc_hp)
        
        # 4. Tiefpass-Filter (nutzt nun die Variable aus config)
        acc_smoothed = apply_zero_phase_filter(
            acc_rectified, 
            cutoff=self.config.OFFLINE_ZUPT_LP_CUTOFF, 
            fs=fs_dynamisch, 
            order=1, 
            btype='low'
        )
        
        # 5. Schwellenwert anwenden (nutzt nun die Variable aus config)
        df_imu['Stationary'] = acc_smoothed < self.config.OFFLINE_ZUPT_THRESHOLD
        return df_imu