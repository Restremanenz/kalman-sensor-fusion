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


def compute_optimal_time_shift(imu_t, imu_vz, vid_t, vid_vz):
    """
    Führt das Signal-Alignment zwischen IMU- und Video-Geschwindigkeit durch.
    Kombiniert Kreuzkorrelation (grob) mit Least-Squares (fein).
    """
    from scipy.signal import butter, filtfilt, correlate, correlation_lags
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize
    
    # 1. Tiefpassfilter (3 Hz) um hochfrequentes Rauschen / Wackeln zu ignorieren
    fs_imu = 1.0 / np.mean(np.diff(imu_t))
    b_i, a_i = butter(2, 3.0 / (0.5 * fs_imu), btype='low')
    imu_vz_filt = filtfilt(b_i, a_i, imu_vz)
    
    fs_vid = 1.0 / np.mean(np.diff(vid_t))
    b_v, a_v = butter(2, 3.0 / (0.5 * fs_vid), btype='low')
    vid_vz_filt = filtfilt(b_v, a_v, vid_vz)
    
    # 2. Spline-Interpolation der Video-Daten (für sub-frame Genauigkeit)
    vid_spline = CubicSpline(vid_t, vid_vz_filt)
    
    # 3. Grobe Synchronisation (Cross-Correlation)
    imu_t_rel = imu_t - imu_t[0]
    vid_resampled = np.zeros_like(imu_t_rel)
    
    valid = (imu_t_rel >= vid_t[0]) & (imu_t_rel <= vid_t[-1])
    vid_resampled[valid] = vid_spline(imu_t_rel[valid])
    
    corr = correlate(imu_vz_filt, vid_resampled, mode='full')
    lags = correlation_lags(len(imu_vz_filt), len(vid_resampled), mode='full')
    
    coarse_lag = lags[np.argmax(corr)]
    dt = np.mean(np.diff(imu_t_rel))
    coarse_offset = (coarse_lag * dt) + imu_t[0]
    
    # 4. Feinabstimmung (Least Squares Optimization)
    def cost_func(shift):
        offset = shift[0]
        t_vid_eval = imu_t - offset
        
        mask = (t_vid_eval >= vid_t[0]) & (t_vid_eval <= vid_t[-1])
        if np.sum(mask) < (1.0 * fs_imu): 
            return 1e6
            
        v_vid_eval = vid_spline(t_vid_eval[mask])
        v_imu_eval = imu_vz_filt[mask]
        
        return np.mean((v_imu_eval - v_vid_eval)**2)
        
    res = minimize(cost_func, x0=[coarse_offset], method='Nelder-Mead')
    return res.x[0] if res.success else coarse_offset


def compute_optimal_yaw_correction(positions, velocities, times, target_x, target_y, 
                                   ignore_start_sec=1.5, alpha=1.0, beta=2.0):
    """Findet die optimale Rigid-Body-Rotation (Yaw) für die XY-Ebene."""
    d = np.array([target_x, target_y])
    d_norm = np.linalg.norm(d)
    
    if d_norm < 1e-6:
        return 0.0
        
    u = d / d_norm  
    n = np.array([-u[1], u[0]]) 
    
    t_rel = times - times[0]
    valid_idx = t_rel >= ignore_start_sec
    
    if np.sum(valid_idx) < 100: 
        valid_idx = np.ones_like(t_rel, dtype=bool)
        
    v_valid = velocities[valid_idx, :2] 
    p_N = positions[-1, :2]
    
    def cost_function(theta):
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s], [s, c]])
        
        v_rot = (R_z @ v_valid.T).T
        v_perp = v_rot @ n
        term_beta = beta * np.mean(v_perp**2) 
        
        p_N_rot = R_z @ p_N
        term_alpha = alpha * np.sum((p_N_rot - d)**2)
        
        return term_beta + term_alpha

    res = minimize_scalar(cost_function, bounds=(-np.pi, np.pi), method='bounded')
    return res.x if res.success else 0.0


def align_initial_orientation_to_wall(q_initial, base_yaw_deg, start_pose_yaw_deg=0.0):
    """Richtet die Anfangsorientierung auf das feste Wandkoordinatensystem aus.

    Das Wandkoordinatensystem ist rechtshändig definiert als:
    +X von der Wand weg, +Y nach rechts und +Z nach oben.
    Die Transformation wird vor ESKF und RTS angewendet, damit auch alle
    Messbedingungen und Endpunktbedingungen dieselben Achsen verwenden.
    """
    total_yaw_deg = float(base_yaw_deg) + float(start_pose_yaw_deg)
    wall_rotation = R.from_euler('z', total_yaw_deg, degrees=True)
    return wall_rotation * q_initial, total_yaw_deg


def run_eskf_pipeline(
        df_imu,
        q_init,
        process_start_idx,
        true_start_idx,
        calib,
        fs_dynamisch,
        wall_start_y=0.0,
        apply_lateral_corridor=True
):
    mag_in_run_active = config.USE_MAGNETOMETER and config.USE_18_STATE_ESKF
    
    eskf = FilterpyESKF(
        initial_pos = [0.0, 0.0, 0.0], 
        initial_q = q_init, 
        gyro_noise_density = np.array(config.GYRO_NOISE_DENSITY),
        accel_noise_density = config.ACCEL_NOISE_DENSITY,
        bg_rw_density = np.array(config.GYRO_BIAS_RW_DENSITY),
        ba_rw_density = config.ACCEL_BIAS_RW_DENSITY,
        grav_unc = config.GRAVITY_UNCERTAINTY,
        zupt_unc = config.ZUPT_UNCERTAINTY,
        baro_unc = config.BARO_UNCERTAINTY,
        zaru_unc = config.ZARU_UNCERTAINTY,
        use_18_state = mag_in_run_active,
        mag_rw_density = config.MAG_BIAS_RW,
        mag_unc = config.MAG_UNCERTAINTY
    )
    
    smoother = ESKFSmoother(use_18_state=mag_in_run_active)
    smoother_started = False
    
    times_plot = []
    times = df_imu["Time"].values
    last_baro_time = -1.0 
    last_mag_time = -1.0

    corridor_enabled = (
        apply_lateral_corridor
        and getattr(config, 'USE_LATERAL_CORRIDOR', False)
    )
    corridor_hits = 0
    last_corridor_update_time = -np.inf

    if corridor_enabled:
        corridor_y_min = float(getattr(config, 'CORRIDOR_Y_MIN_M', -1.5)) - wall_start_y
        corridor_y_max = float(getattr(config, 'CORRIDOR_Y_MAX_M', 1.5)) - wall_start_y
        corridor_uncertainty = float(getattr(config, 'CORRIDOR_UNCERTAINTY', 0.2))
        corridor_update_hz = float(getattr(config, 'CORRIDOR_UPDATE_HZ', 20.0))
        if corridor_update_hz <= 0.0:
            raise ValueError("CORRIDOR_UPDATE_HZ muss größer als 0 sein.")
        corridor_update_interval = 1.0 / corridor_update_hz
    
    peak_altitude = 0.0
    final_baro_z = 0.0

    end_idx = len(df_imu)
    if getattr(config, 'USE_END_DETECTION', False):
        altitudes = df_imu['Altitude_filt [m]'].values
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

    for i in range(1, end_idx):
        dt = times[i] - times[i-1]
        if dt <= 0: continue
            
        row = df_imu.iloc[i]
        current_baro_time = row['Baro_Time']
        current_mag_time = row['Mag_Time'] if mag_in_run_active else None
        
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

        # WARM-UP PHASE
        if i < true_start_idx:
            eskf.predict(acc_calib, gyro_calib, dt)
            eskf.update_zupt()             
            eskf.update_zaru(gyro_calib)   
            eskf.update_gravity(acc_calib)
            if pd.notna(current_baro_time): last_baro_time = current_baro_time
            if pd.notna(current_mag_time): last_mag_time = current_mag_time
            continue 

        # "CLEAN SLATE" RESET (Lokal bei 0,0,0)
        elif i == true_start_idx:
            eskf.p = np.zeros(3)
            eskf.v = np.zeros(3)
            eskf.kf.P[0:6, 0:6] = 1e-6 

            smoother.save_initial_state(
                eskf.p, eskf.v, eskf.q, eskf.ba, eskf.bg, 
                eskf.bm if mag_in_run_active else np.zeros(3), eskf.kf.P
            )
            smoother_started = True

        # ESKF KOPPELNAVIGATION
        eskf.predict(acc_calib, gyro_calib, dt)
        if smoother_started:
            smoother.save_predict(
                eskf.p,
                eskf.v,
                eskf.q,
                eskf.ba,
                eskf.bg,
                eskf.bm if mag_in_run_active else np.zeros(3),
                eskf.kf.F,
                eskf.kf.P
            )
        
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

        # Video nur dann als Messung einspielen, wenn die Filterfusion explizit
        # aktiviert ist. Im Vergleichsmodus bleiben IMU und Video unabhängig.
        use_video_in_filter = (
            getattr(config, 'USE_VIDEO_DATA', False)
            and getattr(config, 'USE_VIDEO_IN_FILTER', False)
        )
        if (
            use_video_in_filter
            and 'Video_Y' in df_imu.columns
            and 'Video_Z' in df_imu.columns
            and pd.notna(row['Video_Y'])
            and pd.notna(row['Video_Z'])
        ):
            unc = getattr(config, 'VIDEO_UNCERTAINTY', 0.1)
            eskf.update_video_2d(row['Video_Y'], row['Video_Z'], uncertainty=unc)

        if (
            corridor_enabled
            and (times[i] - last_corridor_update_time) >= corridor_update_interval
        ):
            last_corridor_update_time = times[i]
            corridor_hits += int(
                eskf.update_lateral_corridor(
                    corridor_y_min,
                    corridor_y_max,
                    uncertainty=corridor_uncertainty
                )
            )

        if smoother_started:
            smoother.save_update(
                eskf.p, eskf.v, eskf.q, eskf.ba, eskf.bg, 
                eskf.bm if mag_in_run_active else np.zeros(3), eskf.kf.P
            )
            times_plot.append(times[i])

    if corridor_enabled:
        print(
            f" -> Y-Korridor absolut "
            f"[{getattr(config, 'CORRIDOR_Y_MIN_M', -1.5):+.2f}, "
            f"{getattr(config, 'CORRIDOR_Y_MAX_M', 1.5):+.2f}] m | "
            f"lokal [{corridor_y_min:+.2f}, {corridor_y_max:+.2f}] m | "
            f"Eingriffe: {corridor_hits}"
        )

    # BOUNDARY CONDITIONS & RTS SMOOTHING
    if getattr(config, 'USE_SMOOTHER', False):
        applied_boundary = False
        if getattr(config, 'FORCE_V_END_ZERO', False) and getattr(config, 'MAX_PROCESS_TIME', None) is None:
            eskf.update_zupt() 
            applied_boundary = True
        if getattr(config, 'SMOOTH_XY_TO_ZERO', False):
            target_xy = np.array([getattr(config, 'TARGET_X_M', 0.0), getattr(config, 'TARGET_Y_M', 0.0)])
            xy_unc = getattr(config, 'TARGET_XY_UNCERTAINTY', 0.3)
            eskf.update_position_xy(target_xy, uncertainty=xy_unc) 
            applied_boundary = True
        if getattr(config, 'SMOOTH_TO_BARO_Z', False):
            eskf.update_position_z(final_baro_z, uncertainty=0.001)
            applied_boundary = True

        if applied_boundary:
            smoother.overwrite_last_update(
                eskf.p, eskf.v, eskf.q, eskf.ba, eskf.bg, 
                eskf.bm if mag_in_run_active else np.zeros(3), eskf.kf.P
            )
        positions, velocities, orientations = smoother.smooth()
    else:
        positions, velocities, orientations = smoother.get_forward_states()

    # Die Filtertrajektorie ist relativ zum physischen Start definiert.
    # Der Smoother darf diesen lokalen Koordinatenursprung nicht verschieben.
    if len(positions) > 0:
        positions = positions - positions[0]

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
    # 2. PRE-PROCESSING
    # ==============================================================
    if getattr(config, 'USE_AUTO_INIT', True):
        q_init, P0, process_start_idx, true_start_idx = preprocessor.get_initial_alignment(df_imu, calib, fs_dynamisch)
        euler_deg = q_init.as_euler('xyz', degrees=True) 
        gravity_body = q_init.inv().apply([0.0, 0.0, 1.0]) 

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

    df_imu = preprocessor.process_barometer_and_crop(df_imu, P0, true_start_idx, fs_dynamisch)

    # Die Ausrichtung erfolgt vor Scout-Pass, ESKF und RTS. Dadurch liegen auch
    # Wandbedingung und RTS-Endpunkt von Beginn an im Wandkoordinatensystem.
    base_yaw_deg = getattr(config, 'WALL_FRAME_BASE_YAW_DEG', 180.0)
    start_pose_yaw_deg = getattr(config, 'START_POSE_YAW_CORRECTION_DEG', 0.0)
    q_init_wall, total_wall_yaw_deg = align_initial_orientation_to_wall(
        q_init,
        base_yaw_deg,
        start_pose_yaw_deg
    )

    print(
        f" -> Wandkoordinatensystem: Basis={base_yaw_deg:+.2f}°, "
        f"Startkorrektur={start_pose_yaw_deg:+.2f}°, "
        f"Gesamt={total_wall_yaw_deg:+.2f}°"
    )
    
    # Neue Spalten initialisieren
    df_imu['Video_Y'] = np.nan
    df_imu['Video_Z'] = np.nan
    
    # Absolute Basis-Offsets initialisieren (werden später berechnet)
    true_video_start_y = 0.0
    true_video_start_z = 0.0
    video_positions = None
    
    # ==============================================================
    # 3. DURCHLAUF 1: SCOUT-PASS (Immer in lokal [0,0,0])
    # ==============================================================
    print("\n" + "="*55)
    print("🧭 DURCHLAUF 1: SCOUT-PASS (IMU-Referenz)")
    print("="*55)
    
    original_smoother = getattr(config, 'USE_SMOOTHER', False)
    original_wall_constraint = getattr(config, 'USE_WALL_CONSTRAINT', False)
    config.USE_SMOOTHER = False 
    config.USE_WALL_CONSTRAINT = False
    
    positions_scout, velocities_scout, _, times_scout, _ = run_eskf_pipeline(
        df_imu,
        q_init_wall,
        process_start_idx,
        true_start_idx,
        calib,
        fs_dynamisch,
        apply_lateral_corridor=False
    )
    
    q_init_corrected = q_init_wall
    if getattr(config, 'USE_YAW_CORRECTION', True):
        target_x = getattr(config, 'TARGET_X_M', -1.2)
        target_y = getattr(config, 'TARGET_Y_M', -0.2)
        theta_opt = compute_optimal_yaw_correction(
            positions_scout, velocities_scout, times_scout, target_x, target_y,
            ignore_start_sec=0.5, alpha=1.0, beta=2.0
        )
        print(f" -> 🎯 Yaw-Optimierung abgeschlossen! Fehlstellung: {np.degrees(theta_opt):.2f}°")
        q_init_corrected = R.from_rotvec([0, 0, theta_opt]) * q_init_wall

    # ==============================================================
    # 4. KINEMATISCHER SYNC (Advanced Signal Alignment)
    # ==============================================================
    if getattr(config, 'USE_VIDEO_DATA', False):
        print("\n" + "="*55)
        print("🎬 FÜHRE ERWEITERTEN VIDEO-SYNC DURCH (Cross-Correlation)")
        print("="*55)
        try:
            with open(config.VIDEO_DATA_FILE, 'r') as f:
                v_data = json.load(f)
            
            fps = float(v_data.get("FPS", 60.0))
            startgriff_y = getattr(config, 'STARTGRIFF_Y_M', 0.75)
            startgriff_z = getattr(config, 'STARTGRIFF_Z_M', 1.6875)
            
            # StartOffsets aus JSON holen
            start_pos_x = v_data["StartPos"][1] # Breite
            start_pos_y = v_data["StartPos"][2] # Höhe
            
            true_video_start_y = startgriff_y + start_pos_x 
            true_video_start_z = startgriff_z - start_pos_y 
            
            # Absolute Welt-Daten aufbauen (nur für finale Visualisierung)
            raw_y = np.array(v_data["COG"]["COG_PositionX"]["Raw"]) + true_video_start_y
            raw_z = np.array(v_data["COG"]["COG_PositionY"]["Raw"]) + true_video_start_z
            video_positions = np.column_stack((np.zeros_like(raw_y), raw_y, raw_z))
            
            # Geschwindigkeit für Alignment berechnen
            video_t = np.arange(len(raw_z)) / fps
            video_vz = np.gradient(raw_z, video_t)
            
            imu_t = times_scout
            imu_vz = velocities_scout[:, 2] 
            
            # Algorithmus anwenden
            optimal_offset = compute_optimal_time_shift(imu_t, imu_vz, video_t, video_vz)
            v_times_absolute = video_t + optimal_offset
            
            # Exakte Video-Startzeit berechnen ---
            # imu_t[0] ist der exakte physikalische Startpunkt (true_start_idx)
            video_start_sec = imu_t[0] - optimal_offset
            video_start_frame = int(video_start_sec * fps)
            
            # Rastern und WICHTIG: nur die RELATIVEN Bewegungen an den Filter geben
            nearest_indices = np.searchsorted(df_imu['Time'].values, v_times_absolute)
            valid_mask = nearest_indices < len(df_imu)
            nearest_indices = nearest_indices[valid_mask]
            
            df_imu.loc[nearest_indices, 'Video_Y'] = raw_y[valid_mask] - true_video_start_y
            df_imu.loc[nearest_indices, 'Video_Z'] = raw_z[valid_mask] - true_video_start_z
            
            print(f"[INFO] Sync erfolgreich! Optimaler Zeitversatz (Offset): {optimal_offset:.3f} Sekunden")
            print(f"[INFO] Der Start (IMU-Peak) passiert exakt bei Video-Sekunde: {video_start_sec:.3f}s (Frame ~{video_start_frame})")
            print(
                f"[INFO] Absolute Video-Startposition: "
                f"Y={true_video_start_y:.3f}m, Z={true_video_start_z:.3f}m."
            )
        except Exception as e:
            print(f"[WARNUNG] Videodaten-Sync fehlgeschlagen: {e}")

    # ==============================================================
    # 5. FINALER DURCHLAUF (Lokal bei 0,0,0)
    # ==============================================================
    print("\n" + "="*55)
    print("DURCHLAUF 2: FINALER PASS")
    print("="*55)
    
    config.USE_SMOOTHER = original_smoother
    config.USE_WALL_CONSTRAINT = original_wall_constraint

    sensor_start_wall = np.asarray(
        getattr(config, 'SENSOR_START_POSITION_WALL_M', [0.0, 0.0, 0.0]),
        dtype=float
    )
    if sensor_start_wall.shape != (3,) or not np.all(np.isfinite(sensor_start_wall)):
        raise ValueError(
            "SENSOR_START_POSITION_WALL_M muss drei endliche Werte [X, Y, Z] enthalten."
        )
    
    positions, velocities, orientations, times, eskf = run_eskf_pipeline(
        df_imu,
        q_init_corrected,
        process_start_idx,
        true_start_idx,
        calib,
        fs_dynamisch,
        wall_start_y=sensor_start_wall[1],
        apply_lateral_corridor=True
    )

    # ==============================================================
    # 6. ABSOLUTER WELT-OFFSET & VISUALISIERUNG
    # ==============================================================
    # Die Filtertrajektorie bleibt lokal. Nur eine separate Plot-Kopie wird mit
    # der festen Sensorstartposition in absolute Wandkoordinaten verschoben.
    positions_plot = positions + sensor_start_wall

    if len(times) > 1:
        print("\n" + "="*55)
        print(f"BERECHNETE LAUFZEIT: {(times[-1] - times[0]):.3f} Sekunden")
        print("="*55)

    print("Bereite finale 3D-Plots vor...")
    vis = TrajectoryVisualizer(config=config)
    
    if getattr(config, 'SHOW_3D_TRAJECTORY', True):
        vis.plot_static_trajectory(positions_plot)
    if getattr(config, 'SHOW_ANIMATED_TRAJECTORY', False):
        vis.plot_animated_trajectory(positions_plot, orientations, fs_dynamisch)
    if getattr(config, 'SHOW_2D_FRONT', True):
        vis.plot_2d_wall_with_trajectory(positions_plot, velocities)
    if getattr(config, 'SHOW_2D_FRONT_VIDEO', True) and video_positions is not None:
        vis.plot_2d_wall_with_trajectory(positions_plot, velocities, video_positions)
    if getattr(config, 'SHOW_2D_SIDE', True):
        vis.plot_2d_side_view_with_trajectory(positions_plot, velocities)
    if getattr(config, 'SHOW_VELOCITY', False):
        vis.plot_velocity(times, velocities)
    if getattr(config, 'SHOW_RAW_SENSOR_DATA', False):
        vis.plot_raw_sensor_data(df_imu)
    if getattr(config, 'SHOW_ALTITUDE', False):
        vis.plot_altitude(df_imu)
    if getattr(config, 'SHOW_HIP_ROTATION', True):
        vis.plot_hip_rotation(times, orientations)
    if getattr(config, 'SHOW_2D_FRONT_YAW', True):
        vis.plot_2d_wall_with_yaw(positions_plot, orientations)

    vis.show_all()
    
    print(f"\n[INFO] Finaler In-Run Bias Schätzwert:")
    print(f"Gyro Bias (bg): {eskf.bg}")
    print(f"Accel Bias (ba): {eskf.ba}")
    print(f"Mag Bias (bm): {eskf.bm}")

if __name__ == "__main__":
    main()
