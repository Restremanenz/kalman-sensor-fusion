import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import config  
from st_log_reader import STLogReader
from preprocessing import IMUPreprocessor
from visualization import TrajectoryVisualizer
from filters import FilterpyESKF
from smoother import ESKFSmoother  


class IMUCalibration:
    """Lädt die ST-Kalibrierungsdaten aus einer kombinierten JSON-Datei und wendet sie an."""
    def __init__(self, calib_json_path):
        with open(calib_json_path, 'r') as f:
            data = json.load(f)
            
            # 1. Accelerometer Parameter laden
            self.acc_bias = np.array(data["acc"]["offset_b"])
            self.acc_M = np.array(data["acc"]["matrix_M"])
            
            # 2. Gyroskop Parameter laden
            self.gyro_bias = np.array(data["gyro"]["offset_b"])
            self.gyro_noise_std = np.array(data["gyro"]["noise_std"]) * (np.pi / 180.0)

            # 3. Magnetometer Parameter laden
            self.mag_bias = np.array(data["mag"]["offset_b"])
            self.mag_M = np.array(data["mag"]["matrix_M"])

    def calibrate_acc(self, raw_acc):
        calibrated = self.acc_M @ (raw_acc - self.acc_bias)
        return calibrated * 9.81

    def calibrate_gyro(self, raw_gyro):
        calibrated_dps = raw_gyro - self.gyro_bias
        return calibrated_dps * (np.pi / 180.0)
    
    def calibrate_mag(self, raw_mag):
        """Wendet Hard-Iron (Bias) und Soft-Iron (Matrix) Korrektur an."""
        return self.mag_M @ (raw_mag - self.mag_bias)


def run_eskf_pipeline(df_imu, q_init, init_idx, calib, fs_dynamisch):
    mag_in_run_active = config.USE_MAGNETOMETER and config.USE_18_STATE_ESKF
    print(f"Starte {'18-State' if mag_in_run_active else '15-State'} Error-State Kalman Filter...")
    
    eskf = FilterpyESKF(
        initial_pos = [0.0, 0.0, 0.0], 
        initial_q = q_init, 
        gyro_noise_std = calib.gyro_noise_std,
        accel_noise = config.ACCEL_NOISE_DENSITY,
        bg_rw = config.GYRO_BIAS_RW, 
        ba_rw = config.ACCEL_BIAS_RW, 
        grav_unc = config.GRAVITY_UNCERTAINTY,
        zupt_unc = config.ZUPT_UNCERTAINTY,
        baro_unc = config.BARO_UNCERTAINTY,
        zaru_unc = config.ZARU_UNCERTAINTY,
        use_18_state = mag_in_run_active,
        mag_rw = config.MAG_BIAS_RW,
        mag_unc = config.MAG_UNCERTAINTY
    )
    
    # 1. INITIALISIERE DEN NEUEN SMOOTHER
    smoother = ESKFSmoother(use_18_state=mag_in_run_active)
    smoother.save_initial_state(
        eskf.p, eskf.v, eskf.q, eskf.ba, eskf.bg, 
        eskf.bm if mag_in_run_active else np.zeros(3), eskf.kf.P
    )

    times_plot = []
    times = df_imu["Time"].values
    last_baro_time = -1.0 
    last_mag_time = -1.0
    
    is_climbing = True
    peak_altitude = 0.0
    final_baro_z = 0.0

    # =================================================================
    # 🏁 PRE-COMPUTE END INDEX (Sensor-Fusion: Baro + Accel)
    # =================================================================
    end_idx = len(df_imu)
    
    if getattr(config, 'USE_END_DETECTION', False):
        print("🔍 Suche nach Ziel-Buzzer (Baro + Accel Fusion)...")
        # 1. Grobe Suche: Höchster Punkt laut gefiltertem Barometer (nach dem Start)
        altitudes = df_imu['Altitude_filt [m]'].values
        baro_peak_idx = init_idx + np.argmax(altitudes[init_idx:])
        
        # 2. Feine Suche: Finde den Moment des Loslassens (Freier Fall) 
        # Wir suchen in einem Zeitfenster von 2 Sekunden nach dem Baro-Peak
        search_start = baro_peak_idx - int(0.5 * fs_dynamisch) # 0.5s Toleranz vor dem Peak
        search_end = min(len(df_imu), baro_peak_idx + int(2.5 * fs_dynamisch)) 
        
        acc_mag = np.sqrt(df_imu['A_x [g]']**2 + df_imu['A_y [g]']**2 + df_imu['A_z [g]']**2).values
        
        # Suchen, wann die Beschleunigung unter den Schwellenwert (z.B. 0.6g) fällt
        freefall_indices = np.where(acc_mag[search_start:search_end] < getattr(config, 'FREEFALL_THRESHOLD_G', 0.6))[0]
        
        if len(freefall_indices) > 0:
            # Der exakte Index kurz bevor der freie Fall beginnt
            end_idx = search_start + freefall_indices[0]
            print(f" -> 🎯 Ziel exakt erkannt! Baro-Peak bei {times[baro_peak_idx]:.2f}s | Freier Fall ins Seil beginnt bei {times[end_idx]:.2f}s")
            final_baro_z = altitudes[baro_peak_idx] # Für das Smoother-Boundary Update
        else:
            # Fallback, falls der Kletterer langsam abklettert (kein freier Fall)
            end_idx = baro_peak_idx
            print(f" -> 🎯 Ziel via Baro-Peak bei {times[end_idx]:.2f}s erkannt (Kein freier Fall detektiert).")
            final_baro_z = altitudes[baro_peak_idx]
    # =================================================================
    # ⏱️ LAUFZEIT BERECHNEN (Speedklettern)
    # =================================================================
    if getattr(config, 'USE_END_DETECTION', False):
        start_time = times[init_idx]
        # Falls end_idx das Ende des Arrays erreicht hat, nehmen wir den letzten gültigen Index (-1)
        valid_end_idx = end_idx if end_idx < len(times) else len(times) - 1
        end_time = times[valid_end_idx]
        
        run_time = end_time - start_time
        
        print("\n" + "="*55)
        print("⏱️ SPEEDKLETTERN - LAUFZEIT-ANALYSE")
        print("="*55)
        print(f" -> Start-Zeitpunkt: {start_time:.3f} s (Index {init_idx})")
        print(f" -> Ziel-Zeitpunkt:  {end_time:.3f} s (Index {valid_end_idx})")
        print(f" -> NETTO-LAUFZEIT:  {run_time:.3f} Sekunden")
        print("="*55 + "\n")        

    for i in range(1, end_idx):
        dt = times[i] - times[i-1]
        if dt <= 0: continue
            
        row = df_imu.iloc[i]
        
        # =================================================================
        # 🛡️ PRE-FLIGHT MASKING: Vor dem Startpunkt wird NICHT integriert!
        # =================================================================
        if i < init_idx:
            # Damit Baro/Mag-Timer synchron bleiben, wenn der Start ertönt:
            current_baro_time = row['Baro_Time']
            if pd.notna(current_baro_time): last_baro_time = current_baro_time
            if mag_in_run_active:
                current_mag_time = row['Mag_Time']
                if pd.notna(current_mag_time): last_mag_time = current_mag_time

            # Wir füttern den Smoother künstlich mit Nullen (v=0, p=0, Orientierung=q_init)
            smoother.save_predict(eskf.kf.F, eskf.kf.P) 
            smoother.save_update(
                np.zeros(3),        # Position bleibt hart auf [0,0,0]
                np.zeros(3),        # Geschwindigkeit bleibt hart auf [0,0,0]
                q_init,             # Orientierung ist exakt die Start-Ausrichtung
                eskf.ba, eskf.bg, eskf.bm if mag_in_run_active else np.zeros(3), 
                eskf.kf.P
            )
            times_plot.append(times[i])
            continue  # <-- Springt zur nächsten Zeile, der ESKF wird komplett ignoriert!
        
        # =================================================================
        # AB HIER: NORMALE ESKF KOPPELNAVIGATION (i >= init_idx)
        # =================================================================
        
        raw_acc = np.array([row['A_x [g]'], row['A_y [g]'], row['A_z [g]']]) 
        raw_gyro = np.array([row['G_x [dps]'], row['G_y [dps]'], row['G_z [dps]']]) 
        raw_mag = np.array([row['M_x [G]'], row['M_y [G]'], row['M_z [G]']])

        acc_calib = calib.calibrate_acc(raw_acc)
        gyro_calib = calib.calibrate_gyro(raw_gyro)
        mag_calib = calib.calibrate_mag(raw_mag)

        # 2. PREDICT UND SPEICHERN
        eskf.predict(acc_calib, gyro_calib, dt)
        smoother.save_predict(eskf.kf.F, eskf.kf.P) # Speichert P_prior
        
        # --------------------------------------------------
        # Peak-Erkennung und harter Schleifen-Abbruch
        # --------------------------------------------------
        current_alt = row['Altitude_filt [m]']
        if current_alt > peak_altitude:
            peak_altitude = current_alt  # Höchsten Punkt merken
            
        # Wenn wir z.B. 50cm vom Höchstpunkt abfallen -> Kletterer hängt im Seil!
        if (peak_altitude - current_alt) > getattr(config, 'DESCENT_DETECTION_THRESHOLD', 0.5):
            print(f" -> Info: Abseilen erkannt bei {current_alt:.1f}m (Peak war {peak_altitude:.1f}m). Beende Koppelnavigation!")
            final_baro_z = current_alt  # Höhe festhalten
            break  # <--- Bricht die For-Schleife SOFORT ab!
            
        final_baro_z = current_alt  # Wird aktualisiert, falls wir regulär ans Dateiende kommen

        # --------------------------------------------------
        # Barometer & Wand Update
        # --------------------------------------------------
        current_baro_time = row['Baro_Time']
        if pd.notna(current_baro_time) and current_baro_time != last_baro_time:
            eskf.update_barometer(row['Altitude_filt [m]'])
            
            if getattr(config, 'USE_WALL_CONSTRAINT', False):
                eskf.update_wall_constraint(
                    normal_xy=config.WALL_NORMAL_XY, 
                    inclination_deg=config.WALL_INCLINATION_DEG, 
                    uncertainty=config.WALL_UNCERTAINTY
                )
            last_baro_time = current_baro_time

        mag_magnitude = np.linalg.norm(mag_calib)
        if mag_in_run_active:
            current_mag_time = row['Mag_Time']
            if pd.notna(current_mag_time) and current_mag_time != last_mag_time and abs(mag_magnitude - 1.0) < 0.1:
                eskf.update_mag(mag_calib)
                last_mag_time = current_mag_time

        if getattr(config, 'USE_ZUPT', False):
            acc_world = eskf.q.apply(acc_calib)
            acc_magnitude = np.linalg.norm(acc_world + eskf.g)
            gyro_magnitude = np.linalg.norm(gyro_calib)
            zaru_threshold_rads = getattr(config, 'STILLNESS_THRESHOLD', 3.0) * (np.pi / 180.0)
            
            if row['Stationary'] == True and gyro_magnitude < zaru_threshold_rads and acc_magnitude < 0.4:
                eskf.update_zupt()             
                eskf.update_zaru(gyro_calib)   
                eskf.update_gravity(acc_calib) 

        # 3. UPDATE SPEICHERN
        smoother.save_update(
            eskf.p, eskf.v, eskf.q, eskf.ba, eskf.bg, 
            eskf.bm if mag_in_run_active else np.zeros(3), eskf.kf.P
        )
        times_plot.append(times[i])

    # ==========================================================
    # 4. BOUNDARY CONDITIONS ALS KALMAN-UPDATES & RTS SMOOTHING
    # ==========================================================
    if getattr(config, 'USE_SMOOTHER', False):
        applied_boundary = False
        
        # Zwinge Velocity exakt auf 0 (Nur wenn komplett ausgelaufen)
        if getattr(config, 'FORCE_V_END_ZERO', False) and getattr(config, 'MAX_PROCESS_TIME', None) is None:
            eskf.update_zupt() 
            applied_boundary = True

        # Zwinge Position auf Startpunkt-Linie (z.B. für RTO)
        if getattr(config, 'SMOOTH_XY_TO_ZERO', False):
            target_xy = np.array([getattr(config, 'TARGET_X_M', 0.0), getattr(config, 'TARGET_Y_M', 0.0)])
            eskf.update_position_xy(target_xy, uncertainty=0.001) # 1mm Unsicherheit -> Sehr hartes Update
            applied_boundary = True

        # Zwinge Z-Position auf exakte letzte Barometer-Messung
        if getattr(config, 'SMOOTH_TO_BARO_Z', False):
            # Nutzt die Höhe aus dem exakten Moment des Schleifen-Abbruchs
            eskf.update_position_z(final_baro_z, uncertainty=0.001)
            applied_boundary = True

        # Falls Bedingungen angewendet wurden, überschreiben wir den allerletzten Zustand im Smoother
        if applied_boundary:
            smoother.overwrite_last_update(
                eskf.p, eskf.v, eskf.q, eskf.ba, eskf.bg, 
                eskf.bm if mag_in_run_active else np.zeros(3), eskf.kf.P
            )

        # Starte den echten RTS-Rückwärtsdurchlauf
        positions, velocities, orientations = smoother.smooth()
    else:
        # Falls Smoother aus ist, nutze die ungeschönten Vorwärts-Werte
        positions, velocities, orientations = smoother.get_forward_states()

    return positions, velocities, orientations, np.array(times_plot), eskf


def main():
    # ==============================================================
    # 1. SETUP & DATEN LADEN
    # ==============================================================
    reader = STLogReader(config.LOG_FOLDER)
    calib = IMUCalibration(config.IMU_CALIB_FILE)
    preprocessor = IMUPreprocessor(config)
    
    df_imu, fs_dynamisch = preprocessor.load_and_merge_data(reader)

    # ==============================================================
    # 2. PRE-PROCESSING (Initialisierung & Barometer)
    # ==============================================================
    if getattr(config, 'USE_AUTO_INIT', True):
        q_init, P0, init_idx = preprocessor.get_initial_alignment(df_imu, calib, fs_dynamisch)

        # ==============================================================
        # AUSGABE FÜR DEN VERGLEICH 
        # ==============================================================
        euler_deg = q_init.as_euler('xyz', degrees=True) # Umrechnung in Grad
        gravity_body = q_init.inv().apply([0.0, 0.0, 1.0]) # Berechnung des Vektors

        print("\n" + "="*55)
        print("🎯 INITIAL ALIGNMENT VERGLEICH")
        print("="*55)
        print(f"Methode:      {'Dynamisch (Madgwick)' if getattr(config, 'USE_DYNAMIC_ALIGNMENT', False) else 'Statisch (Wasserwaage)'}")
        print(f"Start-Index:  {init_idx}")
        print(f"Winkel:       Roll: {euler_deg[0]:.2f}° | Pitch: {euler_deg[1]:.2f}° | Yaw: {euler_deg[2]:.2f}°")
        print(f"Schwerkraft:  [X: {gravity_body[0]:.4f}g | Y: {gravity_body[1]:.4f}g | Z: {gravity_body[2]:.4f}g]")
        print("="*55 + "\n")
    else:
        print("WARNUNG: Auto-Init deaktiviert. Starte unkalibriert.")
        init_idx = 1
        P0 = df_imu['P [hPa]'].iloc[0]
        q_init = R.from_quat([0, 0, 0, 1])

    # Höhenberechnung, Zero-Phase Filter, Tare & Data-Cropping
    df_imu = preprocessor.process_barometer_and_crop(df_imu, P0, init_idx, fs_dynamisch)

    # ==============================================================
    # 3. KOPPELNAVIGATION (15-State ESKF)
    # ==============================================================
    positions, velocities, orientations, times, eskf = run_eskf_pipeline(
        df_imu, q_init, init_idx, calib, fs_dynamisch
    )


    # ==============================================================
    # 5. VISUALISIERUNG
    # ==============================================================
    print("Bereite finale 3D-Plots vor...")
    vis = TrajectoryVisualizer(animation_fps=config.ANIMATION_FPS)
    
    vis.plot_static_trajectory(positions)
    # vis.plot_animated_trajectory(positions, orientations, fs_dynamisch)

    if getattr(config, 'SHOW_VELOCITY', False):
        vis.plot_velocity(times, velocities)
    if getattr(config, 'SHOW_RAW_SENSOR_DATA', False):
        vis.plot_raw_sensor_data(df_imu)
    if getattr(config, 'SHOW_ALTITUDE', False):
        vis.plot_altitude(df_imu)

    vis.show_all()
    
    print(f"\n[INFO] Finaler In-Run Bias Schätzwert:")
    print(f"Gyro Bias (bg): {eskf.bg}")
    print(f"Accel Bias (ba): {eskf.ba}")
    print(f"Mag Bias (bm): {eskf.bm}")

if __name__ == "__main__":
    main()