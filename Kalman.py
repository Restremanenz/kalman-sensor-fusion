import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar

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
            self.gyro_noise_std = np.array(config.GYRO_NOISE_STD)

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


def run_eskf_pipeline(df_imu, q_init, process_start_idx, true_start_idx, calib, fs_dynamisch):
    mag_in_run_active = config.USE_MAGNETOMETER and config.USE_18_STATE_ESKF
    print(f"Starte {'18-State' if mag_in_run_active else '15-State'} Error-State Kalman Filter...")
    
    eskf = FilterpyESKF(
        initial_pos = [0.0, 0.0, 0.0], 
        initial_q = q_init, 
        gyro_noise_std = calib.gyro_noise_std,
        accel_noise = config.ACCEL_NOISE_DENSITY,
        bg_rw = np.array(config.GYRO_BIAS_RW), 
        ba_rw = config.ACCEL_BIAS_RW, 
        grav_unc = config.GRAVITY_UNCERTAINTY,
        zupt_unc = config.ZUPT_UNCERTAINTY,
        baro_unc = config.BARO_UNCERTAINTY,
        zaru_unc = config.ZARU_UNCERTAINTY,
        use_18_state = mag_in_run_active,
        mag_rw = config.MAG_BIAS_RW,
        mag_unc = config.MAG_UNCERTAINTY
    )
    
    smoother = ESKFSmoother(use_18_state=mag_in_run_active)
    smoother_started = False
    
    times_plot = []
    times = df_imu["Time"].values
    last_baro_time = -1.0 
    last_mag_time = -1.0
    
    peak_altitude = 0.0
    final_baro_z = 0.0

    end_idx = len(df_imu)
    if getattr(config, 'USE_END_DETECTION', False):
        altitudes = df_imu['Altitude_filt [m]'].values
        # Suche nach Ziel erst AB dem echten Start
        baro_peak_idx = true_start_idx + np.argmax(altitudes[true_start_idx:])
        search_start = baro_peak_idx - int(0.5 * fs_dynamisch) 
        search_end = min(len(df_imu), baro_peak_idx + int(2.5 * fs_dynamisch)) 
        
        acc_mag = np.sqrt(df_imu['A_x [g]']**2 + df_imu['A_y [g]']**2 + df_imu['A_z [g]']**2).values
        freefall_indices = np.where(acc_mag[search_start:search_end] < getattr(config, 'FREEFALL_THRESHOLD_G', 0.6))[0]
        
        if len(freefall_indices) > 0:
            end_idx = search_start + freefall_indices[0]
            final_baro_z = altitudes[baro_peak_idx] 
        else:
            end_idx = baro_peak_idx
            final_baro_z = altitudes[baro_peak_idx]

    # Wir beginnen die Schleife bei 1 für die dt-Berechnung
    for i in range(1, end_idx):
        dt = times[i] - times[i-1]
        if dt <= 0: continue
            
        row = df_imu.iloc[i]
        current_baro_time = row['Baro_Time']
        current_mag_time = row['Mag_Time'] if mag_in_run_active else None
        
        # Ignoriere alles VOR dem definierten Verarbeitungspunkt (spart Rechenzeit)
        if i < process_start_idx:
            if pd.notna(current_baro_time): last_baro_time = current_baro_time
            if pd.notna(current_mag_time): last_mag_time = current_mag_time
            continue 

        raw_acc = np.array([row['A_x [g]'], row['A_y [g]'], row['A_z [g]']]) 
        raw_gyro = np.array([row['G_x [dps]'], row['G_y [dps]'], row['G_z [dps]']]) 
        raw_mag = np.array([row['M_x [G]'], row['M_y [G]'], row['M_z [G]']])

        acc_calib = calib.calibrate_acc(raw_acc)
        gyro_calib = calib.calibrate_gyro(raw_gyro)
        mag_calib = calib.calibrate_mag(raw_mag)

        # =================================================================
        # 🛠️ WARM-UP PHASE (Nur bei Methode 'ESKF' aktiv)
        # =================================================================
        if i < true_start_idx:
            eskf.predict(acc_calib, gyro_calib, dt)
            
            # Zwinge den Filter an Ort und Stelle: Er MUSS Orientierung & Biases optimieren!
            eskf.update_zupt()             
            eskf.update_zaru(gyro_calib)   
            eskf.update_gravity(acc_calib)
            
            if pd.notna(current_baro_time): last_baro_time = current_baro_time
            if pd.notna(current_mag_time): last_mag_time = current_mag_time
            continue # Smoother wird hier noch NICHT gefüttert!

        # =================================================================
        # 🚀 "CLEAN SLATE" RESET (Exakt am Moment des Start-Peaks)
        # =================================================================
        elif i == true_start_idx:
            # Orientierung (q) und Biases (ba, bg) behalten wir aus dem Warm-Up!
            # Position und Geschwindigkeit hart auf [0,0,0] setzen:
            eskf.p = np.zeros(3)
            eskf.v = np.zeros(3)
            
            # Mache die Kovarianz für Position & Geschwindigkeit sehr "sicher", 
            # da wir den Ursprung des Koordinatensystems exakt hier definieren.
            eskf.kf.P[0:6, 0:6] = 1e-6 

            smoother.save_initial_state(
                eskf.p, eskf.v, eskf.q, eskf.ba, eskf.bg, 
                eskf.bm if mag_in_run_active else np.zeros(3), eskf.kf.P
            )
            smoother_started = True

        # =================================================================
        # 🏃‍♂️ AB HIER: NORMALE ESKF KOPPELNAVIGATION (i >= true_start_idx)
        # =================================================================
        eskf.predict(acc_calib, gyro_calib, dt)
        if smoother_started:
            smoother.save_predict(eskf.kf.F, eskf.kf.P) 
        
        current_alt = row['Altitude_filt [m]']
        if current_alt > peak_altitude:
            peak_altitude = current_alt  
            
        if (peak_altitude - current_alt) > getattr(config, 'DESCENT_DETECTION_THRESHOLD', 0.5):
            print(f" -> Info: Abseilen erkannt bei {current_alt:.1f}m. Beende Koppelnavigation!")
            final_baro_z = current_alt 
            break  
            
        final_baro_z = current_alt

        if pd.notna(current_baro_time) and current_baro_time != last_baro_time:
            eskf.update_barometer(row['Altitude_filt [m]'])
            if getattr(config, 'USE_WALL_CONSTRAINT', False):
                eskf.update_wall_constraint(normal_xy=config.WALL_NORMAL_XY, inclination_deg=config.WALL_INCLINATION_DEG, uncertainty=config.WALL_UNCERTAINTY)
            last_baro_time = current_baro_time

        if mag_in_run_active and pd.notna(current_mag_time) and current_mag_time != last_mag_time:
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

        if smoother_started:
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
            xy_unc = getattr(config, 'TARGET_XY_UNCERTAINTY', 0.3)
            eskf.update_position_xy(target_xy, uncertainty=xy_unc) 
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

def compute_optimal_yaw_correction(positions, velocities, times, target_x, target_y, 
                                   ignore_start_sec=1.5, alpha=1.0, beta=2.0):
    """
    Findet die optimale Rigid-Body-Rotation (Yaw) für die XY-Ebene, 
    indem die Quergeschwindigkeiten minimiert und der Endpunkt verankert wird.
    
    Kostenfunktion: J(θ) = β * mean(v_perp(θ)^2) + α * ||p_N(θ) - d||^2
    """
    # 1. Ziel-Vektor d (Die ideale Linie von Start zu Ziel)
    d = np.array([target_x, target_y])
    d_norm = np.linalg.norm(d)
    
    if d_norm < 1e-6:
        print("Warnung: Zielvektor ist 0. Keine Ausrichtung möglich.")
        return 0.0
        
    # Hauptbewegungsrichtung (u) und orthogonale Richtung (n, Querbewegung)
    u = d / d_norm  
    n = np.array([-u[1], u[0]]) 
    
    # 2. Startphase verwerfen (Biomechanik: Erstes Eindrehen ignorieren)
    # Geht davon aus, dass 'times' in Sekunden ist und bei 0 (oder ähnlich) startet
    t_rel = times - times[0]
    valid_idx = t_rel >= ignore_start_sec
    
    # Falls die Trajektorie zu kurz ist, Fallback auf alle Daten
    if np.sum(valid_idx) < 100: 
        valid_idx = np.ones_like(t_rel, dtype=bool)
        print("Warnung: Trajektorie zu kurz, nutze alle Daten für PCA.")
        
    v_valid = velocities[valid_idx, :2] # Nur X/Y Geschwindigkeiten ab Sekunde X
    
    # Endposition (unkorrigiert)
    p_N = positions[-1, :2]
    
    # 3. Kostenfunktion definieren
    def cost_function(theta):
        # 2D Rotationsmatrix für Winkel theta
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([
            [c, -s], 
            [s,  c]
        ])
        
        # A. Geschwindigkeiten rotieren und auf Querachse projizieren
        v_rot = (R_z @ v_valid.T).T
        v_perp = v_rot @ n
        # Querbewegungsterm (höher gewichtet, da Form robuster ist als Drift)
        term_beta = beta * np.mean(v_perp**2) 
        
        # B. Endposition rotieren und Abstand zum echten Ziel messen
        p_N_rot = R_z @ p_N
        term_alpha = alpha * np.sum((p_N_rot - d)**2)
        
        return term_beta + term_alpha

    # 4. Eindimensionale Optimierung (Suche im Bereich von -180° bis +180°)
    # minimize_scalar ist perfekt und rasend schnell für 1-DOF Probleme
    res = minimize_scalar(cost_function, bounds=(-np.pi, np.pi), method='bounded')
    
    if res.success:
        return res.x # Der optimale Winkel theta (in Radiant)
    else:
        print("Warnung: Optimierung fehlgeschlagen, nutze 0°.")
        return 0.0
    
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
        q_init, P0, process_start_idx, true_start_idx = preprocessor.get_initial_alignment(df_imu, calib, fs_dynamisch)

        # ==============================================================
        # AUSGABE FÜR DEN VERGLEICH 
        # ==============================================================
        euler_deg = q_init.as_euler('xyz', degrees=True) # Umrechnung in Grad
        gravity_body = q_init.inv().apply([0.0, 0.0, 1.0]) # Berechnung des Vektors

        print("\n" + "="*55)
        print("INITIAL ALIGNMENT VERGLEICH")
        print("="*55)
        print(f"Methode:      {getattr(config, 'ALIGNMENT_METHOD', 'STATIC')}")
        print(f"Start-Indizes: Process: {process_start_idx} | True Start: {true_start_idx}")
        print(f"Winkel:       Roll: {euler_deg[0]:.2f}° | Pitch: {euler_deg[1]:.2f}° | Yaw: {euler_deg[2]:.2f}°")
        print(f"Schwerkraft:  [X: {gravity_body[0]:.4f}g | Y: {gravity_body[1]:.4f}g | Z: {gravity_body[2]:.4f}g]")
        print("="*55 + "\n")
    else:
        print("WARNUNG: Auto-Init deaktiviert. Starte unkalibriert.")
        process_start_idx = 1
        true_start_idx = 1
        P0 = df_imu['P [hPa]'].iloc[0]
        q_init = R.from_quat([0, 0, 0, 1])

    # Höhenberechnung, Zero-Phase Filter, Tare & Data-Cropping
    df_imu = preprocessor.process_barometer_and_crop(df_imu, P0, true_start_idx, fs_dynamisch)

    # ==============================================================
    # 3. HIERARCHISCHE SCHÄTZUNG: SCOUT-PASS (Vorwärts-Filter)
    # ==============================================================
    use_yaw_correction = getattr(config, 'USE_YAW_CORRECTION', True)

    if use_yaw_correction:
        print("\n" + "="*55)
        print("🧭 DURCHLAUF 1: SCOUT-PASS (Erkennung der echten Kletterachse)")
        print("="*55)
        
        # WICHTIG: Smoother für den Scout-Pass abschalten!
        original_smoother = getattr(config, 'USE_SMOOTHER', False)
        original_wall_constraint = getattr(config, 'USE_WALL_CONSTRAINT', False)

        config.USE_SMOOTHER = False 
        config.USE_WALL_CONSTRAINT = False
        
        positions_scout, velocities_scout, _, times_scout, _ = run_eskf_pipeline(
            df_imu, q_init, process_start_idx, true_start_idx, calib, fs_dynamisch
        )
        
        # Lese physikalische Zielkoordinaten aus der Config
        target_x = getattr(config, 'TARGET_X_M', -1.2)
        target_y = getattr(config, 'TARGET_Y_M', -0.2)
        
        # ==============================================================
        # 4. OPTIMIERUNG: 1-DOF RIGID BODY ROTATION FINDEN
        # ==============================================================
        # Gewichte können getunt werden: 
        # beta=2.0 (Querbewegung stark bestrafen), alpha=1.0 (Endpunkt sanft anziehen)
        theta_opt = compute_optimal_yaw_correction(
            positions_scout, velocities_scout, times_scout, target_x, target_y,
            ignore_start_sec=0.5, alpha=1.0, beta=2.0
        )
        
        print(f" -> 🎯 Optimierung abgeschlossen!")
        print(f" -> Berechnete Fehlstellung (Yaw): {np.degrees(theta_opt):.2f}°")
        
        # Drehe das initiale Quaternion als starren Körper um die Z-Achse
        q_yaw = R.from_rotvec([0, 0, theta_opt])
        q_init_corrected = q_yaw * q_init
    
        # ==============================================================
        # 5. FINALER DURCHLAUF: VORWÄRTS + RTS SMOOTHER
        # ==============================================================
        print("\n" + "="*55)
        print("DURCHLAUF 2: FINALER PASS (Rigid-Body-Corrected + Smoother)")
        print("="*55)
        
        # Smoother wieder aktivieren, damit echter Rausch-Drift geglättet wird
        config.USE_SMOOTHER = original_smoother
        config.USE_WALL_CONSTRAINT = original_wall_constraint
        
        # Finale Trajektorie mit korrigiertem Start-Heading berechnen
        positions, velocities, orientations, times, eskf = run_eskf_pipeline(
            df_imu, q_init_corrected, process_start_idx, true_start_idx, calib, fs_dynamisch
        )
        
    else:
        print("\n" + "="*55)
        print("🧭 YAW-KORREKTUR DEAKTIVIERT (Originaler 1-Pass)")
        print("="*55)
        
        # Dein originaler, direkter Aufruf ohne Vorab-Drehung
        positions, velocities, orientations, times, eskf = run_eskf_pipeline(
            df_imu, q_init, process_start_idx, true_start_idx, calib, fs_dynamisch
        )

    # LAUFZEIT BERECHNEN UND AUSGEBEN 
    if len(times) > 1:
        run_duration = times[-1] - times[0]
        print("\n" + "="*55)
        print(f"BERECHNETE LAUFZEIT: {run_duration:.3f} Sekunden")
        print("="*55)
    else:
        print("\n[WARNUNG] Laufzeit konnte nicht berechnet werden (zu wenig Daten).")

    # ==============================================================
    # VIDEO-DATEN LADEN (Für die Visualisierung)
    # ==============================================================
    video_positions = None
    if getattr(config, 'USE_VIDEO_DATA', False):
        try:
            with open(config.VIDEO_DATA_FILE, 'r') as f:
                v_data = json.load(f)
            
            # JSON PositionX -> Y-Achse (Breite), PositionY -> Z-Achse (Höhe)
            raw_y = np.array(v_data["COG"]["COG_PositionX"]["Raw"]) + config.VIDEO_OFFSET_Y
            raw_z = np.array(v_data["COG"]["COG_PositionY"]["Raw"]) + config.VIDEO_OFFSET_Z
            raw_x = np.zeros_like(raw_y) # Tiefe ist 0, da wir nur 2D Video haben
            
            # Als Nx3 Array stapeln wie deine positions
            video_positions = np.column_stack((raw_x, raw_y, raw_z))
            print(f"[INFO] Videodaten erfolgreich geladen ({len(video_positions)} Frames).")
        except Exception as e:
            print(f"[WARNUNG] Konnte Videodaten nicht laden: {e}")

    # ==============================================================
    # 6. VISUALISIERUNG
    # ==============================================================
    print("Bereite finale 3D-Plots vor...")
    
    vis = TrajectoryVisualizer(config=config)
    
    # Plots basierend auf der config.py steuern
    if getattr(config, 'SHOW_3D_TRAJECTORY', True):
        vis.plot_static_trajectory(positions)
    if getattr(config, 'SHOW_ANIMATED_TRAJECTORY', False):
        vis.plot_animated_trajectory(positions, orientations, fs_dynamisch)
    if getattr(config, 'SHOW_2D_FRONT', True):
        vis.plot_2d_wall_with_trajectory(positions, velocities) 
    if getattr(config, 'SHOW_2D_FRONT_VIDEO', True):
        vis.plot_2d_wall_with_trajectory(positions, velocities, video_positions)        
    if getattr(config, 'SHOW_2D_SIDE', True):
        vis.plot_2d_side_view_with_trajectory(positions, velocities)
    if getattr(config, 'SHOW_VELOCITY', False):
        vis.plot_velocity(times, velocities)
    if getattr(config, 'SHOW_RAW_SENSOR_DATA', False):
        vis.plot_raw_sensor_data(df_imu)
    if getattr(config, 'SHOW_ALTITUDE', False):
        vis.plot_altitude(df_imu)
    if getattr(config, 'SHOW_HIP_ROTATION', True):
        vis.plot_hip_rotation(times, orientations)
    if getattr(config, 'SHOW_2D_FRONT_YAW', True):
        vis.plot_2d_wall_with_yaw(positions, orientations)

    vis.show_all()
    
    print(f"\n[INFO] Finaler In-Run Bias Schätzwert:")
    print(f"Gyro Bias (bg): {eskf.bg}")
    print(f"Accel Bias (ba): {eskf.ba}")
    print(f"Mag Bias (bm): {eskf.bm}")

if __name__ == "__main__":
    main()