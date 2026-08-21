import json
from dataclasses import replace

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
from pipeline_variants import get_pipeline_options


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


def rotate_initial_covariance(initial_covariance, world_rotation):
    """Dreht globale ESKF-Fehlerzustände in einen neuen Weltframe."""
    if initial_covariance is None:
        return None

    covariance = np.asarray(initial_covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "Die Startkovarianz muss eine quadratische Matrix sein."
        )
    if covariance.shape[0] < 9:
        raise ValueError(
            "Die Startkovarianz enthält keine vollständigen ESKF-Zustände."
        )

    transform = np.eye(covariance.shape[0])
    rotation_matrix = world_rotation.as_matrix()

    # Position, Geschwindigkeit und der links-multiplikative
    # Orientierungsfehler sind im globalen Koordinatensystem definiert.
    transform[0:3, 0:3] = rotation_matrix
    transform[3:6, 3:6] = rotation_matrix
    transform[6:9, 6:9] = rotation_matrix

    rotated = transform @ covariance @ transform.T
    return 0.5 * (rotated + rotated.T)


def apply_initial_covariance(eskf, initial_covariance):
    """Überträgt eine 15- oder 18-State-Kovarianz auf den ESKF."""
    if initial_covariance is None:
        return

    source = np.asarray(initial_covariance, dtype=float)
    if source.shape == (eskf.dim_x, eskf.dim_x):
        covariance = source.copy()
    elif source.shape == (15, 15) and eskf.dim_x == 18:
        covariance = eskf.kf.P.copy()
        covariance[:15, :15] = source
    else:
        raise ValueError(
            f"Startkovarianz {source.shape} passt nicht zum "
            f"{eskf.dim_x}-State-ESKF."
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError("Die Startkovarianz enthält ungültige Werte.")

    covariance = 0.5 * (covariance + covariance.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue < -1e-8:
        raise ValueError(
            "Die Warm-up-Kovarianz ist nicht positiv semidefinit."
        )
    if minimum_eigenvalue < 0.0:
        covariance += np.eye(eskf.dim_x) * (
            -minimum_eigenvalue + 1e-12
        )

    eskf.kf.P = covariance
    eskf.kf.x = np.zeros(eskf.dim_x)


def run_eskf_pipeline(
        df_imu,
        q_init,
        process_start_idx,
        true_start_idx,
        calib,
        fs_dynamisch,
        options,
        wall_start_y=0.0,
        initial_accel_bias_mps2=None,
        initial_gyro_bias_radps=None,
        initial_covariance=None,
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
    if initial_accel_bias_mps2 is not None:
        eskf.ba = np.asarray(
            initial_accel_bias_mps2,
            dtype=float,
        ).copy()
    if initial_gyro_bias_radps is not None:
        eskf.bg = np.asarray(
            initial_gyro_bias_radps,
            dtype=float,
        ).copy()
    apply_initial_covariance(eskf, initial_covariance)
    
    smoother = ESKFSmoother(use_18_state=mag_in_run_active)
    smoother_started = False
    
    times_plot = []
    times = df_imu["Time"].values
    last_baro_time = -1.0 
    last_mag_time = -1.0

    corridor_enabled = options.use_lateral_corridor
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

        # VORSTARTPROPAGATION
        # Die Orientierung wird ab dem ausgewählten Gravity-Fenster bis zum
        # Bewegungsstart ausschließlich mit dem kalibrierten Gyroskop
        # fortgeschrieben. Position und Geschwindigkeit werden am Start ohnehin
        # zurückgesetzt. Unbedingte Gravity-/ZARU-/ZUPT-Updates würden echte
        # Vorstartbewegungen fälschlich als Sensorfehler interpretieren.
        if i < true_start_idx:
            eskf.predict(acc_calib, gyro_calib, dt)
            if pd.notna(current_baro_time): last_baro_time = current_baro_time
            if pd.notna(current_mag_time): last_mag_time = current_mag_time
            continue 

        # Lokaler Trajektorienstart bei [0, 0, 0].
        elif i == true_start_idx:
            eskf.p = np.zeros(3)
            eskf.v = np.zeros(3)
            eskf.kf.x = np.zeros(eskf.dim_x)

            position_variance = float(getattr(
                config,
                'START_POSITION_STD_M',
                0.001,
            )) ** 2
            velocity_variance = float(getattr(
                config,
                'START_VELOCITY_STD_MPS',
                0.05,
            )) ** 2
            if position_variance <= 0.0 or velocity_variance <= 0.0:
                raise ValueError(
                    "Die Startunsicherheiten müssen größer als null sein."
                )

            start_covariance = eskf.kf.P.copy()

            # Position und Geschwindigkeit definieren am Start ein neues
            # lokales Koordinatensystem. Nur ihre Zustände und Kopplungen
            # werden zurückgesetzt. Die im Warm-up bestimmten Lage- und
            # Bias-Kovarianzen samt ihrer gegenseitigen Kopplungen bleiben.
            start_covariance[0:6, :] = 0.0
            start_covariance[:, 0:6] = 0.0
            start_covariance[0:3, 0:3] = (
                np.eye(3) * position_variance
            )
            start_covariance[3:6, 3:6] = (
                np.eye(3) * velocity_variance
            )
            eskf.kf.P = 0.5 * (
                start_covariance + start_covariance.T
            )

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
            if options.use_barometer:
                eskf.update_barometer(row['Altitude_filt [m]'])
            if options.use_wall_constraint:
                eskf.update_wall_constraint(
                    normal_xy=config.WALL_NORMAL_XY,
                    inclination_deg=config.WALL_INCLINATION_DEG,
                    uncertainty=config.WALL_UNCERTAINTY
                )
            last_baro_time = current_baro_time

        if mag_in_run_active and pd.notna(current_mag_time) and current_mag_time != last_mag_time:
            eskf.update_mag(mag_calib)
            last_mag_time = current_mag_time

        if options.use_zupt:
            acc_world = eskf.q.apply(acc_calib)
            acc_magnitude = np.linalg.norm(acc_world + eskf.g)
            gyro_magnitude = np.linalg.norm(gyro_calib)
            zaru_threshold_rads = (
                getattr(config, 'ZARU_GYRO_THRESHOLD_DPS', 4.0)
                * (np.pi / 180.0)
            )
            
            if row['Stationary'] == True and gyro_magnitude < zaru_threshold_rads and acc_magnitude < 0.4:
                eskf.update_zupt()             
                eskf.update_zaru(gyro_calib)   
                eskf.update_gravity(acc_calib) 

        # Video nur dann als Messung einspielen, wenn die Filterfusion explizit
        # aktiviert ist. Im Vergleichsmodus bleiben IMU und Video unabhängig.
        use_video_in_filter = (
            getattr(config, 'USE_VIDEO_DATA', False)
            and options.use_video_in_filter
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
    if options.use_rts:
        applied_boundary = False
        if getattr(config, 'FORCE_V_END_ZERO', False) and getattr(config, 'MAX_PROCESS_TIME', None) is None:
            eskf.update_zupt() 
            applied_boundary = True
        if options.use_endpoint:
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


def _initial_attitude_cost(
        positions,
        times,
        eskf,
        df_imu,
        roll_correction_deg,
        pitch_correction_deg,
        wall_start_y,
):
    """Bewertet eine Startorientierung ohne Verwendung der Videoreferenz."""
    if len(positions) < 2 or len(times) != len(positions):
        return np.inf, {'invalid': True}

    baro_reference = np.interp(
        times,
        df_imu['Time'].to_numpy(dtype=float),
        df_imu['Altitude_filt [m]'].to_numpy(dtype=float),
    )
    baro_std = max(float(config.INITIAL_OPT_BARO_STD_M), 1e-6)
    baro_cost = float(np.mean(
        ((positions[:, 2] - baro_reference) / baro_std) ** 2
    ))

    wall_normal = np.asarray(config.WALL_NORMAL_XY, dtype=float)
    wall_normal_norm = np.linalg.norm(wall_normal)
    if wall_normal_norm <= 1e-9:
        raise ValueError("WALL_NORMAL_XY darf kein Nullvektor sein.")
    wall_normal /= wall_normal_norm
    wall_residual = (
        positions[:, :2] @ wall_normal
        - positions[:, 2] * np.tan(np.radians(config.WALL_INCLINATION_DEG))
    )
    wall_std = max(float(config.WALL_UNCERTAINTY), 1e-6)
    wall_cost = float(np.mean((wall_residual / wall_std) ** 2))

    corridor_y_min = float(config.CORRIDOR_Y_MIN_M) - float(wall_start_y)
    corridor_y_max = float(config.CORRIDOR_Y_MAX_M) - float(wall_start_y)
    y_local = positions[:, 1]
    corridor_excess = np.where(
        y_local < corridor_y_min,
        corridor_y_min - y_local,
        np.where(y_local > corridor_y_max, y_local - corridor_y_max, 0.0),
    )
    corridor_std = max(float(config.CORRIDOR_UNCERTAINTY), 1e-6)
    corridor_cost = float(np.mean((corridor_excess / corridor_std) ** 2))

    target_x = float(config.TARGET_X_M)
    target_y = float(config.TARGET_Y_M)
    target_x_std = max(float(config.INITIAL_OPT_TARGET_X_STD_M), 1e-6)
    target_y_std = max(float(config.INITIAL_OPT_TARGET_Y_STD_M), 1e-6)
    endpoint_cost = float(
        ((positions[-1, 0] - target_x) / target_x_std) ** 2
        + ((positions[-1, 1] - target_y) / target_y_std) ** 2
    )

    z_peak = max(float(np.max(positions[:, 2])), 1e-6)
    z_progress = np.clip(positions[:, 2] / z_peak, 0.0, 1.0)
    expected_y = target_y * z_progress
    lateral_shape_std = max(
        float(config.INITIAL_OPT_LATERAL_SHAPE_STD_M),
        1e-6,
    )
    lateral_shape_cost = float(np.mean(
        ((positions[:, 1] - expected_y) / lateral_shape_std) ** 2
    ))

    attitude_prior_std = max(
        float(config.INITIAL_ATTITUDE_PRIOR_STD_DEG),
        1e-6,
    )
    attitude_prior_cost = float(
        (roll_correction_deg / attitude_prior_std) ** 2
        + (pitch_correction_deg / attitude_prior_std) ** 2
    )

    accel_bias_std = max(
        float(config.INITIAL_OPT_ACCEL_BIAS_STD_MPS2),
        1e-6,
    )
    gyro_bias_std = max(
        float(config.INITIAL_OPT_GYRO_BIAS_STD_RADPS),
        1e-6,
    )
    bias_cost = float(
        np.sum((eskf.ba / accel_bias_std) ** 2)
        + np.sum((eskf.bg / gyro_bias_std) ** 2)
    )

    components = {
        'barometer': baro_cost,
        'wall': wall_cost,
        'corridor': corridor_cost,
        'endpoint': endpoint_cost,
        'lateral_shape': lateral_shape_cost,
        'attitude_prior': attitude_prior_cost,
        'bias': bias_cost,
    }
    # Die geometrischen Rohkosten können bei einer durch Gravity-Leakage stark
    # driftenden Trajektorie sehr groß werden. log1p erhält ihre Rangfolge,
    # verhindert aber, dass eine einzelne unsichere Annahme (insbesondere der
    # geschätzte Endpunkt) die gesamte Optimierung dominiert. Der Winkel-Prior
    # bleibt quadratisch: Große nachträgliche Korrekturen sind damit nur bei
    # klarer Evidenz zulässig.
    total_cost = float(
        config.INITIAL_OPT_BARO_WEIGHT * np.log1p(baro_cost)
        + config.INITIAL_OPT_WALL_WEIGHT * np.log1p(wall_cost)
        + config.INITIAL_OPT_CORRIDOR_WEIGHT * np.log1p(corridor_cost)
        + config.INITIAL_OPT_ENDPOINT_WEIGHT * np.log1p(endpoint_cost)
        + config.INITIAL_OPT_LATERAL_SHAPE_WEIGHT
        * np.log1p(lateral_shape_cost)
        + config.INITIAL_OPT_ATTITUDE_PRIOR_WEIGHT * attitude_prior_cost
        + config.INITIAL_OPT_BIAS_WEIGHT * np.log1p(bias_cost)
    )
    if not np.isfinite(total_cost):
        return np.inf, components
    return total_cost, components


def optimize_initial_attitude(
        df_imu,
        candidate_alignments,
        true_start_idx,
        calib,
        fs_dynamisch,
        sync_reference_options,
        base_yaw_deg,
        start_pose_yaw_deg,
        wall_start_y,
        fixed_solution=None,
):
    """Wählt Fenster sowie kleine Roll-/Pitch-Korrektur deterministisch aus."""
    if not candidate_alignments:
        raise ValueError("Keine Startorientierungskandidaten vorhanden.")

    candidate_by_id = {
        candidate['candidate_id']: candidate
        for candidate in candidate_alignments
    }
    evaluation_cache = {}

    def evaluate(candidate, roll_deg, pitch_deg):
        key = (
            candidate['candidate_id'],
            round(float(roll_deg), 6),
            round(float(pitch_deg), 6),
        )
        if key in evaluation_cache:
            return evaluation_cache[key]

        q_wall, total_wall_yaw_deg = align_initial_orientation_to_wall(
            candidate['q_initial'],
            base_yaw_deg,
            start_pose_yaw_deg,
        )
        tilt_correction = R.from_euler(
            'xy',
            [float(roll_deg), float(pitch_deg)],
            degrees=True,
        )
        q_test = tilt_correction * q_wall
        wall_rotation = R.from_euler(
            'z',
            total_wall_yaw_deg,
            degrees=True,
        )
        complete_frame_rotation = tilt_correction * wall_rotation
        initial_covariance_wall = rotate_initial_covariance(
            candidate.get('initial_covariance'),
            complete_frame_rotation,
        )
        positions, velocities, orientations, times, eskf = run_eskf_pipeline(
            df_imu,
            q_test,
            candidate['process_start_idx'],
            true_start_idx,
            calib,
            fs_dynamisch,
            options=sync_reference_options,
            wall_start_y=wall_start_y,
            initial_accel_bias_mps2=candidate.get(
                'initial_accel_bias_mps2'
            ),
            initial_gyro_bias_radps=candidate.get(
                'initial_gyro_bias_radps'
            ),
            initial_covariance=initial_covariance_wall,
        )
        total_cost, components = _initial_attitude_cost(
            positions,
            times,
            eskf,
            df_imu,
            float(roll_deg),
            float(pitch_deg),
            wall_start_y,
        )
        result = {
            'candidate_id': candidate['candidate_id'],
            'roles': list(candidate['roles']),
            'roll_correction_deg': float(roll_deg),
            'pitch_correction_deg': float(pitch_deg),
            'total_cost': total_cost,
            'cost_components': components,
            'q_initial_wall': q_test,
            'process_start_idx': int(candidate['process_start_idx']),
            'initial_accel_bias_mps2': np.asarray(candidate.get(
                'initial_accel_bias_mps2',
                np.zeros(3),
            ), dtype=float),
            'initial_gyro_bias_radps': np.asarray(candidate.get(
                'initial_gyro_bias_radps',
                np.zeros(3),
            ), dtype=float),
            'initial_covariance': (
                None
                if initial_covariance_wall is None
                else initial_covariance_wall.copy()
            ),
            'positions': positions,
            'velocities': velocities,
            'orientations': orientations,
            'times': times,
        }
        evaluation_cache[key] = result
        return result

    if fixed_solution is not None:
        candidate_id = fixed_solution['candidate_id']
        if candidate_id not in candidate_by_id:
            raise ValueError(
                f"Fixierter Initialisierungskandidat {candidate_id!r} "
                "ist im aktuellen Lauf nicht verfügbar."
            )
        selected = evaluate(
            candidate_by_id[candidate_id],
            fixed_solution['roll_correction_deg'],
            fixed_solution['pitch_correction_deg'],
        )
        # In den folgenden Validierungsvarianten steht die Entscheidung bereits
        # fest. Erneute Vergleichsläufe aller Fenster wären redundant und
        # könnten die gemeinsame Initialisierung nicht mehr verändern.
        baseline_evaluations = [selected]
        adaptive_baseline = selected
        selection_source = 'fixed_validation_solution'
    else:
        baseline_evaluations = [
            evaluate(candidate, 0.0, 0.0)
            for candidate in candidate_alignments
        ]
        adaptive_baseline = next(
            (
                result
                for result in baseline_evaluations
                if 'longest_valid' in result['roles']
            ),
            baseline_evaluations[0],
        )
        # Die Kandidatenwahl verwendet ausschließlich Vorstartdaten. Kosten der
        # späteren Klettertrajektorie werden nur diagnostisch gespeichert.
        candidate = min(
            candidate_alignments,
            key=lambda item: (
                item['candidate_id'] != str(getattr(
                    config,
                    'INITIAL_ATTITUDE_SOURCE',
                    'STABILIZED_RECENT_WARMUP',
                )),
                item.get('selection_priority', 1),
                item['selection_score'],
                item['quality_score'],
                item['gyro_p95_dps'],
                -item['duration_s'],
            ),
        )
        quality_baseline = evaluate(candidate, 0.0, 0.0)

        if not bool(getattr(
            config,
            'USE_INITIAL_ATTITUDE_FINE_TUNING',
            False,
        )):
            selected = quality_baseline
            adaptive_baseline = selected
            candidate_diagnostics = []
            for result in baseline_evaluations:
                source = candidate_by_id[result['candidate_id']]
                candidate_diagnostics.append({
                    'candidate_id': result['candidate_id'],
                    'roles': list(result['roles']),
                    'selected': (
                        result['candidate_id'] == selected['candidate_id']
                    ),
                    'duration_s': float(source['duration_s']),
                    'quality_score': float(source['quality_score']),
                    'gyro_mean_dps': float(source['gyro_mean_dps']),
                    'gyro_p95_dps': float(source['gyro_p95_dps']),
                    'prestart_score': float(source['prestart_score']),
                    'short_window_penalty': float(
                        source['short_window_penalty']
                    ),
                    'selection_score': float(source['selection_score']),
                    'warmup_report': dict(source['warmup_report']),
                    'trajectory_cost': float(result['total_cost']),
                    'cost_components': dict(result['cost_components']),
                    'endpoint_local_m': result['positions'][-1].tolist(),
                    'maximum_abs_lateral_m': float(np.max(np.abs(
                        result['positions'][:, 1]
                    ))),
                })
            report = {
                'enabled': True,
                'selection_source': 'configured_initial_attitude_source',
                'accepted': True,
                'rejection_reasons': [],
                'candidate_id': selected['candidate_id'],
                'candidate_roles': list(selected['roles']),
                'roll_correction_deg': 0.0,
                'pitch_correction_deg': 0.0,
                'total_cost': selected['total_cost'],
                'cost_components': dict(selected['cost_components']),
                'adaptive_baseline_candidate_id': selected['candidate_id'],
                'adaptive_baseline_cost': selected['total_cost'],
                'selected_cost_improvement': 0.0,
                'quality_baseline_candidate_id': selected['candidate_id'],
                'quality_baseline_cost': selected['total_cost'],
                'proposed_candidate_id': selected['candidate_id'],
                'proposed_roll_correction_deg': 0.0,
                'proposed_pitch_correction_deg': 0.0,
                'proposed_cost': selected['total_cost'],
                'proposed_relative_improvement': 0.0,
                'proposed_roll_at_search_boundary': False,
                'proposed_pitch_at_search_boundary': False,
                'roll_at_search_boundary': False,
                'pitch_at_search_boundary': False,
                'evaluation_count': int(len(evaluation_cache)),
                'uses_video_reference': False,
                'candidate_selection_method': (
                    'configured_source_then_prestart_consistency'
                ),
                'candidate_baselines': candidate_diagnostics,
            }
            solution = {
                'candidate_id': selected['candidate_id'],
                'roll_correction_deg': 0.0,
                'pitch_correction_deg': 0.0,
            }
            return selected, adaptive_baseline, report, solution

        roll_min = float(config.INITIAL_ROLL_SEARCH_MIN_DEG)
        roll_max = float(config.INITIAL_ROLL_SEARCH_MAX_DEG)
        pitch_min = float(config.INITIAL_PITCH_SEARCH_MIN_DEG)
        pitch_max = float(config.INITIAL_PITCH_SEARCH_MAX_DEG)
        coarse_step = float(config.INITIAL_ATTITUDE_COARSE_STEP_DEG)
        fine_step = float(config.INITIAL_ATTITUDE_FINE_STEP_DEG)
        fine_radius = float(config.INITIAL_ATTITUDE_FINE_RADIUS_DEG)
        if coarse_step <= 0.0 or fine_step <= 0.0 or fine_radius < 0.0:
            raise ValueError(
                "Schrittweiten der Startorientierungsoptimierung sind ungültig."
            )

        roll_values = np.arange(
            roll_min,
            roll_max + 0.5 * coarse_step,
            coarse_step,
        )
        pitch_values = np.arange(
            pitch_min,
            pitch_max + 0.5 * coarse_step,
            coarse_step,
        )
        # Deterministische Koordinatensuche: Roll beeinflusst primär die Y-Z-
        # Ebene und wird deshalb zuerst bewertet. Anschließend wird Pitch bei
        # festem bestem Roll optimiert. Das benötigt wesentlich weniger volle
        # ESKF-Durchläufe als ein kartesisches 2D-Raster.
        coarse_roll_results = [
            evaluate(candidate, roll_deg, 0.0)
            for roll_deg in roll_values
        ]
        coarse_roll_best = min(
            coarse_roll_results,
            key=lambda result: result['total_cost'],
        )
        coarse_pitch_results = [
            evaluate(
                candidate,
                coarse_roll_best['roll_correction_deg'],
                pitch_deg,
            )
            for pitch_deg in pitch_values
        ]
        coarse_best = min(
            coarse_pitch_results,
            key=lambda result: result['total_cost'],
        )

        fine_roll_values = np.arange(
            max(roll_min, coarse_best['roll_correction_deg'] - fine_radius),
            min(roll_max, coarse_best['roll_correction_deg'] + fine_radius)
            + 0.5 * fine_step,
            fine_step,
        )
        fine_roll_results = [
            evaluate(
                candidate,
                roll_deg,
                coarse_best['pitch_correction_deg'],
            )
            for roll_deg in fine_roll_values
        ]
        fine_roll_best = min(
            fine_roll_results,
            key=lambda result: result['total_cost'],
        )
        fine_pitch_values = np.arange(
            max(
                pitch_min,
                fine_roll_best['pitch_correction_deg'] - fine_radius,
            ),
            min(
                pitch_max,
                fine_roll_best['pitch_correction_deg'] + fine_radius,
            ) + 0.5 * fine_step,
            fine_step,
        )
        fine_pitch_results = [
            evaluate(
                candidate,
                fine_roll_best['roll_correction_deg'],
                pitch_deg,
            )
            for pitch_deg in fine_pitch_values
        ]
        proposed = min(
            coarse_roll_results
            + coarse_pitch_results
            + fine_roll_results
            + fine_pitch_results,
            key=lambda result: result['total_cost'],
        )
        proposed_roll_at_boundary = bool(
            np.isclose(proposed['roll_correction_deg'], roll_min)
            or np.isclose(proposed['roll_correction_deg'], roll_max)
        )
        proposed_pitch_at_boundary = bool(
            np.isclose(proposed['pitch_correction_deg'], pitch_min)
            or np.isclose(proposed['pitch_correction_deg'], pitch_max)
        )
        baseline_cost = float(quality_baseline['total_cost'])
        relative_improvement = (
            (baseline_cost - float(proposed['total_cost']))
            / max(abs(baseline_cost), 1e-9)
        )
        minimum_improvement = float(getattr(
            config,
            'INITIAL_ATTITUDE_MIN_RELATIVE_IMPROVEMENT',
            0.05,
        ))
        reject_boundary = bool(getattr(
            config,
            'INITIAL_ATTITUDE_REJECT_BOUNDARY_SOLUTION',
            True,
        ))

        rejection_reasons = []
        if reject_boundary and (
            proposed_roll_at_boundary or proposed_pitch_at_boundary
        ):
            rejection_reasons.append('search_boundary_reached')
        if relative_improvement < minimum_improvement:
            rejection_reasons.append('insufficient_cost_improvement')

        if rejection_reasons:
            selected = quality_baseline
            selection_source = 'best_stillness_quality_fallback'
        else:
            selected = proposed
            selection_source = 'accepted_bounded_physics_optimization'

    roll_at_boundary = bool(
        np.isclose(
            selected['roll_correction_deg'],
            float(config.INITIAL_ROLL_SEARCH_MIN_DEG),
        )
        or np.isclose(
            selected['roll_correction_deg'],
            float(config.INITIAL_ROLL_SEARCH_MAX_DEG),
        )
    )
    pitch_at_boundary = bool(
        np.isclose(
            selected['pitch_correction_deg'],
            float(config.INITIAL_PITCH_SEARCH_MIN_DEG),
        )
        or np.isclose(
            selected['pitch_correction_deg'],
            float(config.INITIAL_PITCH_SEARCH_MAX_DEG),
        )
    )
    proposed_result = locals().get('proposed', selected)
    proposed_roll_at_boundary = bool(locals().get(
        'proposed_roll_at_boundary',
        roll_at_boundary,
    ))
    proposed_pitch_at_boundary = bool(locals().get(
        'proposed_pitch_at_boundary',
        pitch_at_boundary,
    ))
    rejection_reasons = list(locals().get('rejection_reasons', []))
    relative_improvement = float(locals().get(
        'relative_improvement',
        0.0,
    ))
    if fixed_solution is not None:
        quality_baseline_result = selected
    else:
        quality_baseline_result = next(
            result
            for result in baseline_evaluations
            if result['candidate_id'] == min(
                candidate_alignments,
                key=lambda item: (
                    item['quality_score'],
                    -item['duration_s'],
                ),
            )['candidate_id']
        )
    report = {
        'enabled': True,
        'selection_source': selection_source,
        'accepted': not rejection_reasons,
        'rejection_reasons': rejection_reasons,
        'candidate_id': selected['candidate_id'],
        'candidate_roles': list(selected['roles']),
        'roll_correction_deg': selected['roll_correction_deg'],
        'pitch_correction_deg': selected['pitch_correction_deg'],
        'total_cost': selected['total_cost'],
        'cost_components': dict(selected['cost_components']),
        'adaptive_baseline_candidate_id': adaptive_baseline['candidate_id'],
        'adaptive_baseline_cost': adaptive_baseline['total_cost'],
        'selected_cost_improvement': float(
            adaptive_baseline['total_cost'] - selected['total_cost']
        ),
        'quality_baseline_candidate_id': quality_baseline_result[
            'candidate_id'
        ],
        'quality_baseline_cost': quality_baseline_result['total_cost'],
        'proposed_candidate_id': proposed_result['candidate_id'],
        'proposed_roll_correction_deg': proposed_result[
            'roll_correction_deg'
        ],
        'proposed_pitch_correction_deg': proposed_result[
            'pitch_correction_deg'
        ],
        'proposed_cost': proposed_result['total_cost'],
        'proposed_relative_improvement': relative_improvement,
        'proposed_roll_at_search_boundary': proposed_roll_at_boundary,
        'proposed_pitch_at_search_boundary': proposed_pitch_at_boundary,
        'roll_at_search_boundary': roll_at_boundary,
        'pitch_at_search_boundary': pitch_at_boundary,
        'evaluation_count': int(len(evaluation_cache)),
        'uses_video_reference': False,
        'candidate_selection_method': 'minimum_stillness_quality_score',
        'candidate_baselines': [
            {
                'candidate_id': result['candidate_id'],
                'roles': list(result['roles']),
                'quality_score': float(
                    candidate_by_id[result['candidate_id']]['quality_score']
                ),
                'trajectory_cost': float(result['total_cost']),
            }
            for result in baseline_evaluations
        ],
    }
    solution = {
        'candidate_id': selected['candidate_id'],
        'roll_correction_deg': selected['roll_correction_deg'],
        'pitch_correction_deg': selected['pitch_correction_deg'],
    }
    return selected, adaptive_baseline, report, solution


def main(
        prepare_plots=True,
        pipeline_variant=None,
        fixed_video_offset=None,
        fixed_initial_attitude_solution=None,
):
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

    sensor_start_wall = np.asarray(
        getattr(config, 'SENSOR_START_POSITION_WALL_M', [0.0, 0.0, 0.0]),
        dtype=float
    )
    if sensor_start_wall.shape != (3,) or not np.all(np.isfinite(sensor_start_wall)):
        raise ValueError(
            "SENSOR_START_POSITION_WALL_M muss drei endliche Werte [X, Y, Z] enthalten."
        )

    # Die Ausrichtung erfolgt vor SYNC_REFERENCE_PASS, ESKF und RTS. Dadurch liegen auch
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

    variant_name = (
        pipeline_variant
        if pipeline_variant is not None
        else getattr(config, 'ACTIVE_PIPELINE_VARIANT', 'V5_CURRENT')
    )
    active_options = get_pipeline_options(variant_name)

    # Bei einem normalen Programmlauf gilt der Schalter aus config.py.
    # Explizite Validierungsläufe verwenden Video dagegen ausschließlich als
    # unabhängige Referenz und speisen es niemals in den Filter ein.
    if pipeline_variant is None:
        active_options = replace(
            active_options,
            use_video_in_filter=bool(getattr(config, 'USE_VIDEO_IN_FILTER', False))
        )
    sync_reference_options = get_pipeline_options('V2_BARO')

    print(
        f" -> Aktive Pipeline: {active_options.name} | "
        f"{active_options.label}"
    )
    
    # Neue Spalten initialisieren
    df_imu['Video_Y'] = np.nan
    df_imu['Video_Z'] = np.nan
    
    # Absolute Basis-Offsets initialisieren (werden später berechnet)
    true_video_start_y = 0.0
    true_video_start_z = 0.0
    video_positions = None
    video_t = None
    optimal_offset = None
    attitude_optimization_report = None
    initial_attitude_solution = None
    adaptive_sync_reference = None
    selected_initial_accel_bias = np.zeros(3)
    selected_initial_gyro_bias = np.zeros(3)
    selected_initial_covariance = None
    
    # ==============================================================
    # 3. SYNC_REFERENCE_PASS (immer lokal in [0,0,0])
    # Unabhängiger IMU-/Barometer-Durchlauf für die Videozeitsynchronisation.
    # ==============================================================
    print("\n" + "="*55)
    print("🧭 DURCHLAUF 1: SYNC_REFERENCE_PASS")
    print("="*55)
    
    use_attitude_optimization = bool(getattr(
        config,
        'USE_INITIAL_ATTITUDE_OPTIMIZATION',
        False,
    ))
    if use_attitude_optimization:
        candidate_alignments = preprocessor.initialization_alignments
        if not candidate_alignments:
            candidate_alignments = preprocessor.build_initialization_alignments(
                df_imu,
                calib,
                true_start_idx,
            )
        selected_attitude, adaptive_attitude, \
            attitude_optimization_report, \
            initial_attitude_solution = optimize_initial_attitude(
                df_imu=df_imu,
                candidate_alignments=candidate_alignments,
                true_start_idx=true_start_idx,
                calib=calib,
                fs_dynamisch=fs_dynamisch,
                sync_reference_options=sync_reference_options,
                base_yaw_deg=base_yaw_deg,
                start_pose_yaw_deg=start_pose_yaw_deg,
                wall_start_y=sensor_start_wall[1],
                fixed_solution=fixed_initial_attitude_solution,
            )
        q_init_wall = selected_attitude['q_initial_wall']
        process_start_idx = selected_attitude['process_start_idx']
        selected_initial_accel_bias = selected_attitude[
            'initial_accel_bias_mps2'
        ].copy()
        selected_initial_gyro_bias = selected_attitude[
            'initial_gyro_bias_radps'
        ].copy()
        selected_initial_covariance = selected_attitude.get(
            'initial_covariance'
        )
        if selected_initial_covariance is not None:
            selected_initial_covariance = (
                selected_initial_covariance.copy()
            )
        positions_sync_reference = selected_attitude['positions']
        velocities_sync_reference = selected_attitude['velocities']
        times_sync_reference = selected_attitude['times']
        adaptive_sync_reference = {
            'positions': adaptive_attitude['positions'].copy(),
            'velocities': adaptive_attitude['velocities'].copy(),
            'times': adaptive_attitude['times'].copy(),
        }
        preprocessor.initialization_report[
            'attitude_optimization'
        ] = attitude_optimization_report

        print(
            f" -> Startorientierung optimiert: "
            f"Kandidat={attitude_optimization_report['candidate_id']} | "
            f"Roll={attitude_optimization_report['roll_correction_deg']:+.2f}° | "
            f"Pitch={attitude_optimization_report['pitch_correction_deg']:+.2f}° | "
            f"Kosten={attitude_optimization_report['total_cost']:.3f}"
        )
        if not attitude_optimization_report['accepted']:
            proposed_roll = attitude_optimization_report[
                'proposed_roll_correction_deg'
            ]
            proposed_pitch = attitude_optimization_report[
                'proposed_pitch_correction_deg'
            ]
            reasons = ', '.join(
                attitude_optimization_report['rejection_reasons']
            )
            print(
                " -> Vorschlag verworfen: "
                f"Roll={proposed_roll:+.2f}°, "
                f"Pitch={proposed_pitch:+.2f}° | Grund={reasons}. "
                "Verwende unveränderten Qualitätskandidaten."
            )
        if (
            attitude_optimization_report[
                'proposed_roll_at_search_boundary'
            ]
            or attitude_optimization_report[
                'proposed_pitch_at_search_boundary'
            ]
        ):
            print(
                " -> Hinweis: Der Optimierungsvorschlag liegt an einer "
                "Suchgrenze und gilt daher nicht als eindeutig bestimmt."
            )
    else:
        # Auch ohne nachgeschaltete Startlagenoptimierung wird der vom
        # Preprocessor gewählte Warm-up-Zustand vollständig übernommen.
        selected_candidate_id = None
        if preprocessor.initialization_report is not None:
            selected_candidate_id = preprocessor.initialization_report.get(
                'prestart_selected_candidate_id'
            )
        selected_alignment = next(
            (
                candidate
                for candidate in preprocessor.initialization_alignments
                if candidate['candidate_id'] == selected_candidate_id
            ),
            None,
        )
        if selected_alignment is not None:
            selected_initial_accel_bias = np.asarray(
                selected_alignment.get(
                    'initial_accel_bias_mps2',
                    np.zeros(3),
                ),
                dtype=float,
            ).copy()
            selected_initial_gyro_bias = np.asarray(
                selected_alignment.get(
                    'initial_gyro_bias_radps',
                    np.zeros(3),
                ),
                dtype=float,
            ).copy()
            wall_rotation = R.from_euler(
                'z',
                total_wall_yaw_deg,
                degrees=True,
            )
            selected_initial_covariance = rotate_initial_covariance(
                selected_alignment.get('initial_covariance'),
                wall_rotation,
            )

        positions_sync_reference, velocities_sync_reference, _, \
            times_sync_reference, _ = run_eskf_pipeline(
                df_imu,
                q_init_wall,
                process_start_idx,
                true_start_idx,
                calib,
                fs_dynamisch,
                options=sync_reference_options,
                wall_start_y=sensor_start_wall[1],
                initial_accel_bias_mps2=selected_initial_accel_bias,
                initial_gyro_bias_radps=selected_initial_gyro_bias,
                initial_covariance=selected_initial_covariance,
            )
    
    q_init_corrected = q_init_wall
    if getattr(config, 'USE_YAW_CORRECTION', True):
        target_x = getattr(config, 'TARGET_X_M', -1.2)
        target_y = getattr(config, 'TARGET_Y_M', -0.2)
        theta_opt = compute_optimal_yaw_correction(
            positions_sync_reference,
            velocities_sync_reference,
            times_sync_reference,
            target_x,
            target_y,
            ignore_start_sec=0.5, alpha=1.0, beta=2.0
        )
        print(f" -> 🎯 Yaw-Optimierung abgeschlossen! Fehlstellung: {np.degrees(theta_opt):.2f}°")
        yaw_correction = R.from_rotvec([0, 0, theta_opt])
        q_init_corrected = yaw_correction * q_init_wall
        selected_initial_covariance = rotate_initial_covariance(
            selected_initial_covariance,
            yaw_correction,
        )

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
            
            imu_t = times_sync_reference
            imu_vz = velocities_sync_reference[:, 2]
            
            # Der erste Validierungslauf bestimmt die Synchronisation. Alle
            # weiteren Varianten erhalten exakt denselben Offset, damit der
            # Variantenvergleich nicht durch unterschiedliche Sync-Ergebnisse
            # beeinflusst wird.
            if fixed_video_offset is None:
                optimal_offset = compute_optimal_time_shift(
                    imu_t, imu_vz, video_t, video_vz
                )
            else:
                optimal_offset = float(fixed_video_offset)
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
    
    positions, velocities, orientations, times, eskf = run_eskf_pipeline(
        df_imu,
        q_init_corrected,
        process_start_idx,
        true_start_idx,
        calib,
        fs_dynamisch,
        options=active_options,
        wall_start_y=sensor_start_wall[1],
        initial_accel_bias_mps2=selected_initial_accel_bias,
        initial_gyro_bias_radps=selected_initial_gyro_bias,
        initial_covariance=selected_initial_covariance,
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

    if prepare_plots:
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

    return {
        'variant': active_options.name,
        'options': active_options,
        'times': times.copy(),
        'positions_local': positions.copy(),
        'positions_wall': positions_plot.copy(),
        'velocities': velocities.copy(),
        'orientations': orientations,
        'video_times': None if video_t is None else video_t.copy(),
        'video_positions': (
            None if video_positions is None else video_positions.copy()
        ),
        'video_time_offset': optimal_offset,
        'sensor_start_wall': sensor_start_wall.copy(),
        'initial_attitude_solution': initial_attitude_solution,
        'adaptive_sync_reference': adaptive_sync_reference,
        'initialization': (
            None
            if preprocessor.initialization_report is None
            else dict(preprocessor.initialization_report)
        ),
    }

if __name__ == "__main__":
    main()
