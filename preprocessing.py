import sys
import numpy as np
import pandas as pd
import ahrs
from scipy.signal import butter, filtfilt
import scipy.interpolate
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
        """Lädt alle Sensoren, interpoliert weich und merged die DataFrames."""
        print("Lade und synchronisiere Sensordaten...")
        df_acc = reader.get_sensor_data("lsm6dsv16x_acc").sort_values("Time")
        df_gyro = reader.get_sensor_data("lsm6dsv16x_gyro").sort_values("Time")
        df_baro = reader.get_sensor_data("lps22df_press").sort_values("Time")

        # Auslesen der echten Frequenzen
        gyro_info = reader.get_sensor_info("lsm6dsv16x_gyro")
        fs_dynamisch = gyro_info.get("measured_odr_hz", 960.0) if gyro_info else 960.0

        baro_info = reader.get_sensor_info("lps22df_press")
        fs_baro = baro_info.get("measured_odr_hz", 25.0) if baro_info else 25.0
        
        # Automatische Barometer-Spaltenerkennung
        original_press_col = [col for col in df_baro.columns if col != 'Time'][0]
        df_baro = df_baro.rename(columns={original_press_col: 'P [hPa]'})

        # =========================================================
        # 1. Master-Zeitachse aus Accel und Gyro bilden 
        # (Beide laufen auf ~960Hz, hier reicht Nearest-Merge)
        # =========================================================
        df_imu = pd.merge_asof(df_acc, df_gyro, on="Time", direction="nearest")
        master_time = df_imu['Time'].values

        # =========================================================
        # 2. Barometer: Lineare Interpolation + Trigger-Erhalt
        # =========================================================
        # WICHTIG: Originale Zeitstempel behalten, damit der Kalman-Filter nur 25x pro Sekunde updatet!
        df_baro_trigger = pd.DataFrame({'Time': df_baro['Time'], 'Baro_Time': df_baro['Time']})
        df_imu = pd.merge_asof(df_imu, df_baro_trigger, on="Time", direction="backward")
        
        # Echte lineare Interpolation der Messwerte (verhindert Treppenstufen für den Filter)
        f_baro = scipy.interpolate.interp1d(df_baro['Time'], df_baro['P [hPa]'], kind='linear', bounds_error=False, fill_value="extrapolate")
        df_imu['P [hPa]'] = f_baro(master_time)

        # =========================================================
        # 3. Magnetometer: Lineare Interpolation (falls vorhanden)
        # =========================================================
        try:
            df_mag = reader.get_sensor_data("lis2mdl_mag").sort_values("Time")
            mag_info = reader.get_sensor_info("lis2mdl_mag")
            fs_mag = mag_info.get("measured_odr_hz", 100.0) if mag_info else 100.0
            print(f" -> Sensor-Raten: IMU @ {fs_dynamisch:.1f} Hz | Baro @ {fs_baro:.1f} Hz | Mag @ {fs_mag:.1f} Hz")
            
            # Originale Zeitstempel als Trigger behalten (100 Hz Updates)
            df_mag_trigger = pd.DataFrame({'Time': df_mag['Time'], 'Mag_Time': df_mag['Time']})
            df_imu = pd.merge_asof(df_imu, df_mag_trigger, on="Time", direction="backward")
            
            # Interpolation der Magnetwerte in alle drei Dimensionen
            f_mag_x = scipy.interpolate.interp1d(df_mag['Time'], df_mag['M_x [G]'], kind='linear', bounds_error=False, fill_value="extrapolate")
            f_mag_y = scipy.interpolate.interp1d(df_mag['Time'], df_mag['M_y [G]'], kind='linear', bounds_error=False, fill_value="extrapolate")
            f_mag_z = scipy.interpolate.interp1d(df_mag['Time'], df_mag['M_z [G]'], kind='linear', bounds_error=False, fill_value="extrapolate")
            
            df_imu['M_x [G]'] = f_mag_x(master_time)
            df_imu['M_y [G]'] = f_mag_y(master_time)
            df_imu['M_z [G]'] = f_mag_z(master_time)
            
        except ValueError:
            print(f" -> Sensor-Raten: IMU @ {fs_dynamisch:.1f} Hz | Baro @ {fs_baro:.1f} Hz | Mag nicht gefunden")

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

    def get_initial_alignment(self, df_imu, calib, fs_dynamisch):
        fs = fs_dynamisch
        
        if getattr(self.config, 'USE_DYNAMIC_ALIGNMENT', False):
            print("🚀 Nutze Retrograde Dynamic Alignment (Madgwick)...")
            
            # ====================================================================
            # 🔥 FIX 1: Gyro-Bias retten!
            # Wir suchen erst die ECHTE Ruhephase, ABER NUR um das Gyro zu nullen!
            # ====================================================================
            df_still, _ = self.find_initial_stillness(df_imu, fs)
            raw_gyros_still = df_still[['G_x [dps]', 'G_y [dps]', 'G_z [dps]']].values
            calib.gyro_bias = np.mean(raw_gyros_still, axis=0)
            print(f" -> Gyro-Bias erfolgreich aus echter Ruhephase berechnet.")

            # 1. Start-Explosion finden 
            # 🔥 FIX 2: Schwellenwert senken, damit wir nicht schon mitten im Klettern sind!
            acc_mag = np.sqrt(df_imu['A_x [g]']**2 + df_imu['A_y [g]']**2 + df_imu['A_z [g]']**2).values
            threshold = getattr(self.config, 'START_PEAK_THRESHOLD_G', 1.2) # <--- Von 1.5 auf 1.2 gesenkt!
            peaks = np.where(acc_mag > threshold)[0]
            
            if len(peaks) == 0:
                print(" -> WARNUNG: Kein Start-Peak gefunden! Falle auf Static Leveling zurück.")
                df_init, init_idx = self.find_initial_stillness(df_imu, fs)
                q_init, P0, _ = self._static_leveling_alignment(df_init, calib)
                return q_init, P0, init_idx
            
            start_idx = peaks[0] 
            
            # 🔥 FIX 3: Den Sicherheitsabstand (Buffer) erhöhen!
            # Wir gehen 1.0 Sekunden vor dem Peak zurück (statt 0.3s), 
            # damit wir GANZ SICHER vor der Muskelanspannung des Kletterers landen.
            buffer_samples = int(getattr(self.config, 'WARMUP_BUFFER_SEC', 1.0) * fs)
            warmup_samples = int(getattr(self.config, 'WARMUP_WINDOW_SEC', 3.0) * fs)
            
            warmup_end_idx = start_idx - buffer_samples
            warmup_start_idx = warmup_end_idx - warmup_samples
            
            # ... AB HIER GEHT ES EXAKT SO WEITER WIE IN DEINEM AKTUELLEN CODE ...
            # df_warmup = df_imu.iloc[warmup_start_idx:warmup_end_idx]
            # ...
            
            # 3. Warm-Up Daten extrahieren
            df_warmup = df_imu.iloc[warmup_start_idx:warmup_end_idx]
            
            # Wasserwaage für das erste Frame des Warmups
            q_base, P0, _ = self._static_leveling_alignment(df_warmup.head(int(0.5 * fs)), calib)
            q_base_scipy = q_base.as_quat()
            q_base_ahrs = np.array([q_base_scipy[3], q_base_scipy[0], q_base_scipy[1], q_base_scipy[2]])
            
            gyro_data = df_warmup[['G_x [dps]', 'G_y [dps]', 'G_z [dps]']].values
            acc_data = df_warmup[['A_x [g]', 'A_y [g]', 'A_z [g]']].values
            
            gyro_rad = np.deg2rad(gyro_data - calib.gyro_bias)
            acc_calib = np.array([calib.calibrate_acc(a) for a in acc_data])
            
            # 4. Madgwick vorschalten
            madgwick = ahrs.filters.Madgwick(gain=0.05, frequency=fs)
            Q = np.zeros((len(gyro_rad), 4))
            Q[0] = q_base_ahrs 
            
            for i in range(1, len(gyro_rad)):
                Q[i] = madgwick.updateIMU(Q[i-1], gyr=gyro_rad[i], acc=acc_calib[i])
                
            final_q_madgwick = Q[-1] 
            q_init = R.from_quat([final_q_madgwick[1], final_q_madgwick[2], final_q_madgwick[3], final_q_madgwick[0]])
            
            print(f" -> Dynamisches Alignment beendet. ESKF übernimmt bei {start_idx/fs:.2f}s")
            return q_init, P0, start_idx
            
        else:
            print("⚓ Nutze Static Leveling Alignment (Wasserwaage)...")
            # ====================================================================
            # 🔥 HIER IST DIE KORREKTUR:
            # Wir rufen deine originale Suchfunktion auf. Das repariert folgendes:
            # 1. Der Init-Plot wird wieder gezeichnet.
            # 2. Es wird eine *echte* Ruhephase gesucht (maximale Genauigkeit).
            # ====================================================================
            df_init, init_idx = self.find_initial_stillness(df_imu, fs)
            
            # Jetzt berechnen wir das static leveling mit den echten, sauberen Ruhedaten
            q_init, P0, _ = self._static_leveling_alignment(df_init, calib)
            
            # Wir geben den originalen init_idx zurück, damit der Kalman-Filter 
            # exakt nach der gefundenen Ruhephase startet (wie in deinem alten Code!)
            return q_init, P0, init_idx
    def _static_leveling_alignment(self, df_init, calib):
        """Deine vorherige, perfekte Wasserwaagen-Methode als Helper-Funktion ausgelagert."""
        # 1. Gyro Bias berechnen (Darf hier live berechnet werden, weil Sensor echt ruht!)
        raw_gyros_init = df_init[['G_x [dps]', 'G_y [dps]', 'G_z [dps]']].values
        calib.gyro_bias = np.mean(raw_gyros_init, axis=0)

        # 2. Beschleunigung auswerten (Wasserwaage)
        raw_accs_init = df_init[['A_x [g]', 'A_y [g]', 'A_z [g]']].values
        accs_init_calib = np.array([calib.calibrate_acc(a) for a in raw_accs_init])
        mean_acc = np.mean(accs_init_calib, axis=0)
        v_acc_body = mean_acc / np.linalg.norm(mean_acc)

        v2 = np.array([0.0, 0.0, 1.0])
        axis = np.cross(v_acc_body, v2)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(v_acc_body, v2), -1.0, 1.0))
            q_level = R.from_rotvec(axis * angle)
        else:
            q_level = R.from_quat([0,0,0,1])

        q_init = q_level 
        
        P0 = df_init['P [hPa]'].mean()
        init_idx = len(df_init) 
        
        return q_init, P0, init_idx
    
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