import numpy as np
import pandas as pd
import ahrs
from scipy.signal import butter, filtfilt
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
from visualization import TrajectoryVisualizer
from filters import FilterpyESKF

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
        self.initialization_report = None
        self.initialization_candidates = []
        self.initialization_alignments = []

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

    @staticmethod
    def _robust_scale(values, axis=0):
        """Robuste Standardabweichung auf Basis der Medianabweichung."""
        values = np.asarray(values, dtype=float)
        median = np.median(values, axis=axis, keepdims=True)
        return 1.4826 * np.median(np.abs(values - median), axis=axis)

    def _initialization_window_metrics(self, acc_window_g, gyro_window_dps):
        """Bewertet ein mögliches Initialisierungsfenster robust."""
        acc_norm = np.linalg.norm(acc_window_g, axis=1)
        median_acc = np.median(acc_window_g, axis=0)
        median_acc_norm = np.linalg.norm(median_acc)
        if median_acc_norm <= 1e-9:
            return None

        gravity_direction = median_acc / median_acc_norm
        sample_norms = np.linalg.norm(acc_window_g, axis=1, keepdims=True)
        if np.any(sample_norms <= 1e-9):
            return None

        sample_directions = acc_window_g / sample_norms
        direction_angles_deg = np.degrees(np.arccos(np.clip(
            sample_directions @ gravity_direction,
            -1.0,
            1.0,
        )))

        gyro_axis_scale = self._robust_scale(gyro_window_dps, axis=0)
        gyro_magnitude = np.linalg.norm(gyro_window_dps, axis=1)
        return {
            'acc_norm_error_g': float(abs(np.median(acc_norm) - 1.0)),
            'acc_std_g': float(self._robust_scale(acc_norm)),
            'acc_direction_p90_deg': float(np.percentile(direction_angles_deg, 90.0)),
            'gyro_std_dps': float(np.linalg.norm(gyro_axis_scale)),
            'gyro_mean_dps': float(np.mean(gyro_magnitude)),
            'gyro_p95_dps': float(np.percentile(gyro_magnitude, 95.0)),
        }

    def _adaptive_initialization_limits(self, base_metrics):
        """Leitet laufabhängige Grenzen ab und beschränkt sie physikalisch."""
        percentile = float(getattr(self.config, 'INIT_NOISE_PERCENTILE', 10.0))
        factor = float(getattr(self.config, 'INIT_ADAPTIVE_THRESHOLD_FACTOR', 2.5))

        def adaptive_limit(metric_name, minimum_name, maximum_name):
            noise_floor = np.percentile(
                [metrics[metric_name] for metrics in base_metrics],
                percentile,
            )
            minimum = float(getattr(self.config, minimum_name))
            maximum = float(getattr(self.config, maximum_name))
            return float(np.clip(noise_floor * factor, minimum, maximum))

        return {
            'acc_norm_error_g': adaptive_limit(
                'acc_norm_error_g',
                'INIT_ACC_NORM_TOL_MIN_G',
                'INIT_ACC_NORM_TOL_MAX_G',
            ),
            'acc_std_g': adaptive_limit(
                'acc_std_g',
                'INIT_ACC_STD_MIN_G',
                'INIT_ACC_STD_MAX_G',
            ),
            'acc_direction_p90_deg': adaptive_limit(
                'acc_direction_p90_deg',
                'INIT_ACC_DIR_P90_MIN_DEG',
                'INIT_ACC_DIR_P90_MAX_DEG',
            ),
            'gyro_std_dps': adaptive_limit(
                'gyro_std_dps',
                'INIT_GYRO_STD_MIN_DPS',
                'INIT_GYRO_STD_MAX_DPS',
            ),
        }

    def _window_is_valid(self, metrics, limits):
        return bool(
            all(metrics[name] <= limit for name, limit in limits.items())
            and metrics['gyro_mean_dps']
            <= float(self.config.INIT_GYRO_MEAN_MAX_DPS)
            and metrics['gyro_p95_dps']
            <= float(self.config.INIT_GYRO_P95_MAX_DPS)
        )

    @staticmethod
    def _window_quality_score(metrics, limits):
        return float(sum(
            metrics[name] / max(limit, 1e-12)
            for name, limit in limits.items()
        ) + metrics['gyro_mean_dps'] / 3.0
          + metrics['gyro_p95_dps'] / 4.0)

    def _select_independent_phases(self, candidates):
        """Reduziert überlappende Fenster auf getrennte Ruhephasen."""
        if not candidates:
            return []

        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate['start_idx'],
                candidate['end_idx'],
            ),
        )
        clusters = []
        current_cluster = [ordered[0]]
        current_end = ordered[0]['end_idx']

        for candidate in ordered[1:]:
            if candidate['start_idx'] < current_end:
                current_cluster.append(candidate)
                current_end = max(current_end, candidate['end_idx'])
            else:
                clusters.append(current_cluster)
                current_cluster = [candidate]
                current_end = candidate['end_idx']
        clusters.append(current_cluster)

        preferred_duration = float(getattr(
            self.config,
            'INIT_PREFERRED_WINDOW_SECONDS',
            0.6,
        ))
        selected_phases = []
        for cluster in clusters:
            preferred = [
                candidate
                for candidate in cluster
                if candidate['duration_s'] >= preferred_duration
            ]
            pool = preferred if preferred else cluster
            selected_phases.append(min(
                pool,
                key=lambda candidate: (
                    candidate['quality_score'],
                    candidate['gyro_p95_dps'],
                    -candidate['duration_s'],
                ),
            ))

        selected_phases.sort(key=lambda candidate: (
            candidate['quality_score'],
            candidate['gyro_p95_dps'],
            -candidate['duration_s'],
        ))
        maximum = int(getattr(
            self.config,
            'INIT_MAX_PHASE_CANDIDATES',
            12,
        ))
        return selected_phases[:max(1, maximum)]

    def find_initial_stillness(
            self,
            df_imu,
            fs_dynamisch,
            calib,
            search_end_idx,
    ):
        """Findet adaptiv die längste hochwertige Phase vor dem Bewegungsstart."""
        print("Suche adaptiv nach der besten initialen Ruhephase...")

        min_samples = max(
            2,
            int(round(
                float(getattr(self.config, 'INIT_MIN_WINDOW_SECONDS', 0.2))
                * fs_dynamisch
            )),
        )
        max_samples = max(
            min_samples,
            int(round(
                float(getattr(self.config, 'INIT_MAX_WINDOW_SECONDS', 2.0))
                * fs_dynamisch
            )),
        )
        window_step_samples = max(
            1,
            int(round(
                float(getattr(self.config, 'INIT_WINDOW_STEP_SECONDS', 0.05))
                * fs_dynamisch
            )),
        )
        duration_step_samples = max(
            1,
            int(round(
                float(getattr(self.config, 'INIT_DURATION_STEP_SECONDS', 0.05))
                * fs_dynamisch
            )),
        )
        buffer_samples = max(
            0,
            int(round(
                float(getattr(self.config, 'INIT_START_BUFFER_SECONDS', 0.2))
                * fs_dynamisch
            )),
        )
        search_end = min(len(df_imu), int(search_end_idx) - buffer_samples)
        lookback_seconds = getattr(
            self.config,
            'INIT_SEARCH_LOOKBACK_SECONDS',
            None,
        )
        if lookback_seconds is None:
            search_start = 0
        else:
            lookback_samples = max(
                min_samples,
                int(round(float(lookback_seconds) * fs_dynamisch)),
            )
            search_start = max(0, search_end - lookback_samples)
        available_samples = search_end - search_start
        if available_samples < min_samples:
            raise RuntimeError(
                "Vor dem erkannten Bewegungsstart liegen nicht genügend "
                "IMU-Daten für die adaptive Initialisierung vor."
            )

        raw_acc = df_imu[
            ['A_x [g]', 'A_y [g]', 'A_z [g]']
        ].to_numpy(dtype=float)
        acc_calib_g = np.array([
            calib.calibrate_acc(sample) / 9.81
            for sample in raw_acc
        ])

        raw_gyro = df_imu[
            ['G_x [dps]', 'G_y [dps]', 'G_z [dps]']
        ].to_numpy(dtype=float)
        reference_gyro_bias = calib.gyro_bias.copy()
        gyro_corrected = raw_gyro - reference_gyro_bias

        base_metrics = []
        for start_idx in range(
                search_start,
                search_end - min_samples + 1,
                window_step_samples,
        ):
            end_idx = start_idx + min_samples
            metrics = self._initialization_window_metrics(
                acc_calib_g[start_idx:end_idx],
                gyro_corrected[start_idx:end_idx],
            )
            if metrics is not None:
                base_metrics.append(metrics)

        if not base_metrics:
            raise RuntimeError(
                "Die IMU-Daten enthalten kein numerisch gültiges "
                "Initialisierungsfenster."
            )

        adaptive_limits = self._adaptive_initialization_limits(base_metrics)
        absolute_limits = {
            'acc_norm_error_g': float(self.config.INIT_ACC_NORM_TOL_MAX_G),
            'acc_std_g': float(self.config.INIT_ACC_STD_MAX_G),
            'acc_direction_p90_deg': float(self.config.INIT_ACC_DIR_P90_MAX_DEG),
            'gyro_std_dps': float(self.config.INIT_GYRO_STD_MAX_DPS),
        }

        duration_samples = list(range(
            min(max_samples, available_samples),
            min_samples - 1,
            -duration_step_samples,
        ))
        if duration_samples[-1] != min_samples:
            duration_samples.append(min_samples)

        adaptive_candidates = []
        absolute_candidates = []
        low_confidence_candidates = []

        for sample_count in duration_samples:
            for start_idx in range(
                    search_start,
                    search_end - sample_count + 1,
                    window_step_samples,
            ):
                end_idx = start_idx + sample_count
                metrics = self._initialization_window_metrics(
                    acc_calib_g[start_idx:end_idx],
                    gyro_corrected[start_idx:end_idx],
                )
                if metrics is None:
                    continue

                candidate = {
                    **metrics,
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'duration_s': float(sample_count / fs_dynamisch),
                    'quality_score': self._window_quality_score(
                        metrics,
                        adaptive_limits,
                    ),
                }
                if all(
                    metrics[name] <= limit
                    for name, limit in absolute_limits.items()
                ):
                    low_confidence_candidates.append(candidate)
                if self._window_is_valid(metrics, absolute_limits):
                    absolute_candidates.append(candidate)
                if self._window_is_valid(metrics, adaptive_limits):
                    adaptive_candidates.append(candidate)

        if adaptive_candidates:
            candidates = adaptive_candidates
            quality_class = 'adaptiv gültig'
        elif absolute_candidates:
            candidates = absolute_candidates
            quality_class = 'nur absolute Grenzwerte erfüllt'
            print(
                " -> WARNUNG: Kein Fenster erfüllte alle laufadaptiven Grenzen. "
                "Verwende das beste physikalisch plausible Fenster."
            )
        elif low_confidence_candidates:
            candidates = low_confidence_candidates
            quality_class = 'keine echte Ruhephase – Initialisierung unsicher'
            print(
                " -> WARNUNG: Keine Phase erfüllt die harten Grenzen für "
                "Gyro-Mittelwert und Gyro-P95. Verwende den besten "
                "quasi-statischen Kandidaten mit niedriger Konfidenz; der "
                "laufbezogene Gyro-Bias wird nicht übernommen."
            )
        else:
            raise RuntimeError(
                "Keine physikalisch plausible quasi-statische Vorstartphase "
                "gefunden. Die Startorientierung wäre nicht vertrauenswürdig."
            )

        independent_phases = self._select_independent_phases(candidates)
        if not independent_phases:
            raise RuntimeError(
                "Keine voneinander unabhängigen Ruhephasen gefunden."
            )
        selected = independent_phases[0]
        init_start_idx = selected['start_idx']
        init_end_exclusive = selected['end_idx']
        df_init = df_imu.iloc[init_start_idx:init_end_exclusive].copy()

        self.initialization_candidates = []
        for candidate_number, candidate in enumerate(
                independent_phases,
                start=1,
        ):
            self.initialization_candidates.append({
                **candidate,
                'candidate_id': f'INIT_CANDIDATE_{candidate_number}',
                'roles': ['independent_stillness_phase'],
            })

        bias_min_duration = float(getattr(
            self.config,
            'INIT_GYRO_BIAS_MIN_WINDOW_SECONDS',
            0.5,
        ))
        bias_candidates = [
            candidate
            for candidate in absolute_candidates
            if candidate['duration_s'] >= bias_min_duration
            and candidate['gyro_mean_dps']
            <= float(self.config.INIT_GYRO_BIAS_MAX_MEAN_DPS)
            and candidate['gyro_std_dps']
            <= float(self.config.INIT_GYRO_BIAS_MAX_STD_DPS)
        ]

        bias_window = None
        if bias_candidates:
            bias_window = min(
                bias_candidates,
                key=lambda candidate: (
                    candidate['gyro_mean_dps']
                    + candidate['gyro_std_dps'],
                    -candidate['duration_s'],
                    search_end - candidate['end_idx'],
                ),
            )
            calib.gyro_bias = np.median(
                raw_gyro[bias_window['start_idx']:bias_window['end_idx']],
                axis=0,
            )
            bias_source = 'hochwertige Lauf-Ruhephase'
        else:
            calib.gyro_bias = reference_gyro_bias
            bias_source = 'sensor_params.json'

        start_time = float(df_init['Time'].iloc[0])
        end_time = float(df_init['Time'].iloc[-1])
        self.initialization_report = {
            'search_start_time_s': float(df_imu['Time'].iloc[search_start]),
            'search_end_time_s': float(df_imu['Time'].iloc[search_end - 1]),
            'candidate_count': int(len(candidates)),
            'independent_phase_count': int(len(independent_phases)),
            'quality_class': quality_class,
            'strict_stillness_found': bool(
                quality_class
                != 'keine echte Ruhephase – Initialisierung unsicher'
            ),
            'selected_start_time_s': start_time,
            'selected_end_time_s': end_time,
            'selected_duration_s': selected['duration_s'],
            'acc_norm_error_g': selected['acc_norm_error_g'],
            'acc_std_g': selected['acc_std_g'],
            'acc_direction_p90_deg': selected['acc_direction_p90_deg'],
            'gyro_std_dps': selected['gyro_std_dps'],
            'gyro_mean_dps': selected['gyro_mean_dps'],
            'gyro_p95_dps': selected['gyro_p95_dps'],
            'adaptive_limits': adaptive_limits,
            'gyro_bias_source': bias_source,
            'gyro_bias_dps': calib.gyro_bias.tolist(),
            'candidate_count_for_attitude_optimization': int(
                len(self.initialization_candidates)
            ),
            'attitude_candidates': [
                {
                    'candidate_id': candidate['candidate_id'],
                    'roles': list(candidate['roles']),
                    'start_time_s': float(
                        df_imu['Time'].iloc[candidate['start_idx']]
                    ),
                    'end_time_s': float(
                        df_imu['Time'].iloc[candidate['end_idx'] - 1]
                    ),
                    'duration_s': candidate['duration_s'],
                    'quality_score': candidate['quality_score'],
                    'gyro_mean_dps': candidate['gyro_mean_dps'],
                    'gyro_p95_dps': candidate['gyro_p95_dps'],
                    'gyro_std_dps': candidate['gyro_std_dps'],
                    'acc_direction_p90_deg': candidate[
                        'acc_direction_p90_deg'
                    ],
                }
                for candidate in self.initialization_candidates
            ],
            'gyro_bias_window_start_time_s': (
                None
                if bias_window is None
                else float(df_imu['Time'].iloc[bias_window['start_idx']])
            ),
            'gyro_bias_window_end_time_s': (
                None
                if bias_window is None
                else float(df_imu['Time'].iloc[bias_window['end_idx'] - 1])
            ),
        }

        print(
            f" -> Suchbereich: {self.initialization_report['search_start_time_s']:.3f} "
            f"bis {self.initialization_report['search_end_time_s']:.3f} s"
        )
        print(
            f" -> Gewähltes Fenster: {start_time:.3f} bis {end_time:.3f} s | "
            f"Dauer={selected['duration_s']:.3f} s | {quality_class}"
        )
        print(
            f" -> Qualität: Acc-Std={selected['acc_std_g']:.4f} g | "
            f"Richtungs-P90={selected['acc_direction_p90_deg']:.2f}° | "
            f"Gyro-Std={selected['gyro_std_dps']:.2f} dps | "
            f"Gyro-Mittel={selected['gyro_mean_dps']:.2f} dps | "
            f"Gyro-P95={selected['gyro_p95_dps']:.2f} dps"
        )
        print(f" -> Gyro-Bias-Quelle: {bias_source}")

        if getattr(self.config, 'SHOW_INIT_PLOT', True):
            gyro_magnitude = pd.Series(
                np.linalg.norm(gyro_corrected, axis=1),
                index=df_imu.index,
            )
            plot_end_idx = min(
                len(df_imu),
                int(search_end_idx + 2 * fs_dynamisch),
            )
            visualizer_init = TrajectoryVisualizer()
            visualizer_init.plot_auto_init(
                df_plot=df_imu.iloc[search_start:plot_end_idx],
                gyro_mag_plot=gyro_magnitude.iloc[search_start:plot_end_idx],
                df_init=df_init,
                gyro_mag_init=gyro_magnitude.loc[df_init.index],
                threshold=None,
            )

        return df_init, init_end_exclusive - 1

    def propagate_attitude_to_start(
            self,
            df_imu,
            calib,
            q_initial,
            propagation_start_idx,
            true_start_idx,
    ):
        """Führt eine Startlage gyro-basiert und adaptiv gestützt fort."""
        q_current = q_initial
        times = df_imu['Time'].to_numpy(dtype=float)
        raw_acc = df_imu[
            ['A_x [g]', 'A_y [g]', 'A_z [g]']
        ].to_numpy(dtype=float)
        raw_gyro = df_imu[
            ['G_x [dps]', 'G_y [dps]', 'G_z [dps]']
        ].to_numpy(dtype=float)

        acc_gate = max(float(getattr(
            self.config,
            'INIT_WARMUP_ACC_NORM_GATE_G',
            0.10,
        )), 1e-9)
        gyro_gate = max(float(getattr(
            self.config,
            'INIT_WARMUP_GYRO_GATE_DPS',
            15.0,
        )), 1e-9)
        gravity_gain = max(float(getattr(
            self.config,
            'INIT_WARMUP_GRAVITY_GAIN',
            0.015,
        )), 0.0)
        gravity_residuals_deg = []
        gravity_weights = []
        gravity_update_count = 0
        strict_stationary_update_count = 0
        target_gravity_world = np.array([0.0, 0.0, 1.0])

        attitude_eskf = FilterpyESKF(
            initial_pos=[0.0, 0.0, 0.0],
            initial_q=q_initial,
            gyro_noise_density=np.asarray(
                self.config.GYRO_NOISE_DENSITY,
                dtype=float,
            ),
            accel_noise_density=self.config.ACCEL_NOISE_DENSITY,
            bg_rw_density=np.asarray(
                self.config.GYRO_BIAS_RW_DENSITY,
                dtype=float,
            ),
            ba_rw_density=self.config.ACCEL_BIAS_RW_DENSITY,
            grav_unc=self.config.GRAVITY_UNCERTAINTY,
            zupt_unc=self.config.ZUPT_UNCERTAINTY,
            baro_unc=self.config.BARO_UNCERTAINTY,
            zaru_unc=self.config.ZARU_UNCERTAINTY,
            use_18_state=False,
        )
        base_gravity_covariance = attitude_eskf.R_grav.copy()

        start_idx = max(0, int(propagation_start_idx))
        end_idx = min(len(df_imu), int(true_start_idx))
        for index in range(start_idx + 1, end_idx):
            dt = float(times[index] - times[index - 1])
            if not np.isfinite(dt) or dt <= 0.0:
                continue

            gyro_dps = raw_gyro[index] - calib.gyro_bias
            gyro_rad = np.radians(gyro_dps)

            acc_mps2 = calib.calibrate_acc(raw_acc[index])
            attitude_eskf.predict(acc_mps2, gyro_rad, dt)
            q_current = attitude_eskf.q
            acc_norm = float(np.linalg.norm(acc_mps2))
            if not np.isfinite(acc_norm) or acc_norm <= 1e-9:
                continue

            acc_norm_error_g = abs(acc_norm / 9.81 - 1.0)
            gyro_magnitude_dps = float(np.linalg.norm(gyro_dps))
            if (
                acc_norm_error_g >= acc_gate
                or gyro_magnitude_dps >= gyro_gate
            ):
                continue

            acc_weight = (1.0 - acc_norm_error_g / acc_gate) ** 2
            gyro_weight = (1.0 - gyro_magnitude_dps / gyro_gate) ** 2
            update_weight = float(acc_weight * gyro_weight)

            measured_gravity_body = acc_mps2 / acc_norm
            measured_gravity_world = attitude_eskf.q.apply(
                measured_gravity_body
            )
            correction_axis = np.cross(
                measured_gravity_world,
                target_gravity_world,
            )
            axis_norm = float(np.linalg.norm(correction_axis))
            correction_angle = float(np.arctan2(
                axis_norm,
                np.clip(
                    np.dot(
                        measured_gravity_world,
                        target_gravity_world,
                    ),
                    -1.0,
                    1.0,
                ),
            ))
            gravity_residuals_deg.append(np.degrees(correction_angle))
            gravity_weights.append(update_weight)

            if axis_norm <= 1e-12 or update_weight <= 0.0:
                continue

            # Gewichtetes Gravity-Update: Bei Bewegung wird die Messkovarianz
            # vergrößert, statt die Beschleunigung hart als Schwerkraft zu
            # interpretieren. INIT_WARMUP_GRAVITY_GAIN=0.015 entspricht dem
            # nominalen Gewicht 1.0.
            gain_scale = gravity_gain / 0.015 if gravity_gain > 0.0 else 0.0
            effective_weight = float(np.clip(
                update_weight * gain_scale,
                0.02,
                1.0,
            ))
            attitude_eskf.R_grav = (
                base_gravity_covariance / effective_weight
            )
            attitude_eskf.update_gravity(acc_mps2)
            attitude_eskf.R_grav = base_gravity_covariance.copy()
            q_current = attitude_eskf.q
            gravity_update_count += 1

            strict_acc_gate = min(acc_gate, 0.05)
            strict_gyro_gate = float(getattr(
                self.config,
                'INIT_GYRO_P95_MAX_DPS',
                4.0,
            ))
            if (
                acc_norm_error_g < strict_acc_gate
                and gyro_magnitude_dps < strict_gyro_gate
            ):
                attitude_eskf.update_zupt()
                attitude_eskf.update_zaru(gyro_rad)
                q_current = attitude_eskf.q
                strict_stationary_update_count += 1

        if gravity_residuals_deg:
            residual_median = float(np.median(gravity_residuals_deg))
            residual_p90 = float(np.percentile(
                gravity_residuals_deg,
                90.0,
            ))
            mean_weight = float(np.mean(gravity_weights))
        else:
            residual_median = np.inf
            residual_p90 = np.inf
            mean_weight = 0.0

        # Bewertung verwendet ausschließlich Vorstartdaten. Eine Phase mit
        # wenigen nutzbaren Gravity-Beobachtungen erhält einen Unsicherheits-
        # Aufschlag, selbst wenn ihre verbleibenden Residuen klein sind.
        observation_penalty = 1.0 / np.sqrt(
            max(gravity_update_count, 1)
        )
        prestart_score = float(
            residual_median
            + 0.25 * residual_p90
            + observation_penalty
        )
        report = {
            'propagation_start_idx': int(start_idx),
            'propagation_end_idx': int(end_idx),
            'propagation_duration_s': float(
                times[end_idx] - times[start_idx]
                if end_idx < len(times) and end_idx > start_idx
                else 0.0
            ),
            'gravity_update_count': int(gravity_update_count),
            'strict_stationary_update_count': int(
                strict_stationary_update_count
            ),
            'gravity_candidate_sample_count': int(
                len(gravity_residuals_deg)
            ),
            'gravity_mean_weight': mean_weight,
            'gravity_residual_median_deg': residual_median,
            'gravity_residual_p90_deg': residual_p90,
            'prestart_score': prestart_score,
            'estimated_accel_bias_mps2': attitude_eskf.ba.tolist(),
            'estimated_gyro_bias_radps': attitude_eskf.bg.tolist(),
        }
        return q_current, attitude_eskf.kf.P.copy(), report

    def build_initialization_alignments(
            self,
            df_imu,
            calib,
            true_start_idx,
    ):
        """Führt jede unabhängige Ruhephase bis zum Start fort."""
        alignments = []
        for candidate in self.initialization_candidates:
            start_idx = int(candidate['start_idx'])
            end_idx = int(candidate['end_idx'])
            df_candidate = df_imu.iloc[start_idx:end_idx]
            q_candidate, _, _ = self._static_leveling_alignment(
                df_candidate,
                calib,
            )
            q_at_start, covariance_at_start, warmup_report = (
                self.propagate_attitude_to_start(
                    df_imu=df_imu,
                    calib=calib,
                    q_initial=q_candidate,
                    propagation_start_idx=end_idx - 1,
                    true_start_idx=true_start_idx,
                )
            )
            preferred_duration = max(float(getattr(
                self.config,
                'INIT_PREFERRED_WINDOW_SECONDS',
                0.6,
            )), 1e-9)
            short_window_penalty = 2.0 * max(
                0.0,
                (preferred_duration - float(candidate['duration_s']))
                / preferred_duration,
            )
            selection_score = float(
                warmup_report['prestart_score']
                + 0.5 * float(candidate['quality_score'])
                + short_window_penalty
            )
            alignments.append({
                'candidate_id': candidate['candidate_id'],
                'roles': list(candidate['roles']),
                'selection_priority': 1,
                'q_initial': q_at_start,
                'process_start_idx': int(true_start_idx),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration_s': float(candidate['duration_s']),
                'quality_score': float(candidate['quality_score']),
                'gyro_mean_dps': float(candidate['gyro_mean_dps']),
                'gyro_p95_dps': float(candidate['gyro_p95_dps']),
                'prestart_score': float(
                    warmup_report['prestart_score']
                ),
                'short_window_penalty': short_window_penalty,
                'selection_score': selection_score,
                'initial_accel_bias_mps2': np.asarray(
                    warmup_report['estimated_accel_bias_mps2'],
                    dtype=float,
                ),
                'initial_gyro_bias_radps': np.asarray(
                    warmup_report['estimated_gyro_bias_radps'],
                    dtype=float,
                ),
                'initial_covariance': covariance_at_start.copy(),
                'warmup_report': warmup_report,
            })

        recent_alignment = self._build_recent_adaptive_warmup(
            df_imu,
            calib,
            true_start_idx,
        )
        if recent_alignment is not None:
            alignments.append(recent_alignment)

        if bool(getattr(
            self.config,
            'INCLUDE_LEGACY_WARMUP_REFERENCE',
            False,
        )):
            legacy_alignment = self._build_legacy_warmup_reference(
                df_imu,
                calib,
                true_start_idx,
            )
            if legacy_alignment is not None:
                alignments.append(legacy_alignment)
        self.initialization_alignments = alignments
        return alignments

    def _build_recent_adaptive_warmup(
            self,
            df_imu,
            calib,
            true_start_idx,
    ):
        """Bestimmt die Startlage aus einem kurzen aktuellen Warm-up."""
        times = df_imu['Time'].to_numpy(dtype=float)
        if len(times) < 2:
            return None
        fs = 1.0 / float(np.median(np.diff(times)))
        buffer_samples = int(round(float(getattr(
            self.config,
            'LEGACY_WARMUP_BUFFER_SECONDS',
            0.2,
        )) * fs))
        warmup_samples = int(round(float(getattr(
            self.config,
            'LEGACY_WARMUP_WINDOW_SECONDS',
            4.0,
        )) * fs))
        leveling_samples = int(round(float(getattr(
            self.config,
            'LEGACY_WARMUP_LEVELING_SECONDS',
            0.5,
        )) * fs))
        warmup_end_idx = int(true_start_idx) - buffer_samples
        warmup_start_idx = warmup_end_idx - warmup_samples
        leveling_end_idx = warmup_start_idx + leveling_samples
        if warmup_start_idx < 0 or leveling_end_idx > warmup_end_idx:
            return None

        df_leveling = df_imu.iloc[warmup_start_idx:leveling_end_idx]
        q_base, _, _ = self._static_leveling_alignment(
            df_leveling,
            calib,
        )
        q_at_start, covariance_at_start, warmup_report = (
            self.propagate_attitude_to_start(
                df_imu=df_imu,
                calib=calib,
                q_initial=q_base,
                propagation_start_idx=warmup_start_idx,
                true_start_idx=true_start_idx,
            )
        )
        raw_acc = df_imu[
            ['A_x [g]', 'A_y [g]', 'A_z [g]']
        ].to_numpy(dtype=float)
        raw_gyro = df_imu[
            ['G_x [dps]', 'G_y [dps]', 'G_z [dps]']
        ].to_numpy(dtype=float)
        metrics = self._initialization_window_metrics(
            np.array([
                calib.calibrate_acc(sample) / 9.81
                for sample in raw_acc[
                    warmup_start_idx:leveling_end_idx
                ]
            ]),
            raw_gyro[warmup_start_idx:leveling_end_idx]
            - calib.gyro_bias,
        )
        quality_score = float(
            metrics['acc_direction_p90_deg']
            + 0.25 * metrics['gyro_mean_dps']
            + 0.10 * metrics['gyro_p95_dps']
        )
        selection_score = float(
            warmup_report['prestart_score'] + 0.5 * quality_score
        )
        return {
            'candidate_id': 'RECENT_ADAPTIVE_WARMUP',
            'roles': ['primary_recent_attitude_warmup'],
            'selection_priority': 0,
            'q_initial': q_at_start,
            'process_start_idx': int(true_start_idx),
            'start_idx': int(warmup_start_idx),
            'end_idx': int(leveling_end_idx),
            'duration_s': float(
                times[leveling_end_idx - 1] - times[warmup_start_idx]
            ),
            'quality_score': quality_score,
            'gyro_mean_dps': float(metrics['gyro_mean_dps']),
            'gyro_p95_dps': float(metrics['gyro_p95_dps']),
            'prestart_score': float(warmup_report['prestart_score']),
            'short_window_penalty': 0.0,
            'selection_score': selection_score,
            'initial_accel_bias_mps2': np.asarray(
                warmup_report['estimated_accel_bias_mps2'],
                dtype=float,
            ),
            'initial_gyro_bias_radps': np.asarray(
                warmup_report['estimated_gyro_bias_radps'],
                dtype=float,
            ),
            'initial_covariance': covariance_at_start.copy(),
            'warmup_report': {
                **warmup_report,
                'mode': 'recent_adaptive_eskf_warmup',
                'warmup_start_time_s': float(times[warmup_start_idx]),
                'warmup_end_time_s': float(times[int(true_start_idx) - 1]),
            },
        }

    def _build_legacy_warmup_reference(
            self,
            df_imu,
            calib,
            true_start_idx,
    ):
        """Bildet das frühere erzwungene Vier-Sekunden-Warm-up nach."""
        times = df_imu['Time'].to_numpy(dtype=float)
        if len(times) < 2:
            return None
        fs = 1.0 / float(np.median(np.diff(times)))
        buffer_samples = int(round(float(getattr(
            self.config,
            'LEGACY_WARMUP_BUFFER_SECONDS',
            0.2,
        )) * fs))
        warmup_samples = int(round(float(getattr(
            self.config,
            'LEGACY_WARMUP_WINDOW_SECONDS',
            4.0,
        )) * fs))
        leveling_samples = int(round(float(getattr(
            self.config,
            'LEGACY_WARMUP_LEVELING_SECONDS',
            0.5,
        )) * fs))
        warmup_end_idx = int(true_start_idx) - buffer_samples
        warmup_start_idx = warmup_end_idx - warmup_samples
        leveling_end_idx = warmup_start_idx + leveling_samples
        if warmup_start_idx < 0 or leveling_end_idx > warmup_end_idx:
            return None

        df_leveling = df_imu.iloc[warmup_start_idx:leveling_end_idx]
        q_base, _, _ = self._static_leveling_alignment(
            df_leveling,
            calib,
        )
        raw_gyro = df_imu[
            ['G_x [dps]', 'G_y [dps]', 'G_z [dps]']
        ].to_numpy(dtype=float)
        raw_acc = df_imu[
            ['A_x [g]', 'A_y [g]', 'A_z [g]']
        ].to_numpy(dtype=float)
        legacy_gyro_bias_dps = np.mean(
            raw_gyro[warmup_start_idx:leveling_end_idx],
            axis=0,
        )

        eskf = FilterpyESKF(
            initial_pos=[0.0, 0.0, 0.0],
            initial_q=q_base,
            gyro_noise_density=np.asarray(
                self.config.GYRO_NOISE_DENSITY,
                dtype=float,
            ),
            accel_noise_density=self.config.ACCEL_NOISE_DENSITY,
            bg_rw_density=np.asarray(
                self.config.GYRO_BIAS_RW_DENSITY,
                dtype=float,
            ),
            ba_rw_density=self.config.ACCEL_BIAS_RW_DENSITY,
            grav_unc=self.config.GRAVITY_UNCERTAINTY,
            zupt_unc=self.config.ZUPT_UNCERTAINTY,
            baro_unc=self.config.BARO_UNCERTAINTY,
            zaru_unc=self.config.ZARU_UNCERTAINTY,
            use_18_state=False,
        )
        update_count = 0
        for index in range(warmup_start_idx + 1, int(true_start_idx)):
            dt = float(times[index] - times[index - 1])
            if not np.isfinite(dt) or dt <= 0.0:
                continue
            acc_mps2 = calib.calibrate_acc(raw_acc[index])
            gyro_rad = np.radians(
                raw_gyro[index] - legacy_gyro_bias_dps
            )
            eskf.predict(acc_mps2, gyro_rad, dt)
            eskf.update_zupt()
            eskf.update_zaru(gyro_rad)
            eskf.update_gravity(acc_mps2)
            update_count += 1

        # Der aktuelle Filter kalibriert weiterhin mit sensor_params.json.
        # Die Differenz zur historischen Rohbias-Korrektur wird deshalb in den
        # nominalen ESKF-Biaszustand überführt.
        equivalent_bg = (
            eskf.bg
            + np.radians(legacy_gyro_bias_dps - calib.gyro_bias)
        )
        metrics = self._initialization_window_metrics(
            np.array([
                calib.calibrate_acc(sample) / 9.81
                for sample in raw_acc[
                    warmup_start_idx:leveling_end_idx
                ]
            ]),
            raw_gyro[warmup_start_idx:leveling_end_idx]
            - calib.gyro_bias,
        )
        diagnostic_score = float(
            metrics['acc_direction_p90_deg']
            + 0.25 * metrics['gyro_mean_dps']
            + 0.10 * metrics['gyro_p95_dps']
        )
        return {
            'candidate_id': 'STABILIZED_RECENT_WARMUP',
            'roles': [
                'selected_stabilized_warmup',
                'historical_ablation_reference',
            ],
            'selection_priority': 2,
            'q_initial': eskf.q,
            'process_start_idx': int(true_start_idx),
            'start_idx': int(warmup_start_idx),
            'end_idx': int(leveling_end_idx),
            'duration_s': float(
                times[leveling_end_idx - 1] - times[warmup_start_idx]
            ),
            'quality_score': diagnostic_score,
            'gyro_mean_dps': float(metrics['gyro_mean_dps']),
            'gyro_p95_dps': float(metrics['gyro_p95_dps']),
            'prestart_score': diagnostic_score,
            'short_window_penalty': 0.0,
            'selection_score': diagnostic_score,
            'initial_accel_bias_mps2': eskf.ba.copy(),
            'initial_gyro_bias_radps': equivalent_bg,
            'initial_covariance': eskf.kf.P.copy(),
            'warmup_report': {
                'mode': 'legacy_forced_zupt_zaru_gravity',
                'warmup_start_time_s': float(times[warmup_start_idx]),
                'warmup_end_time_s': float(times[int(true_start_idx) - 1]),
                'legacy_leveling_end_time_s': float(
                    times[leveling_end_idx - 1]
                ),
                'forced_update_count': int(update_count),
                'legacy_gyro_bias_dps': legacy_gyro_bias_dps.tolist(),
                'estimated_accel_bias_mps2': eskf.ba.tolist(),
                'estimated_gyro_bias_radps': equivalent_bg.tolist(),
                'prestart_score': diagnostic_score,
            },
        }

    def get_initial_alignment(self, df_imu, calib, fs_dynamisch):
        fs = fs_dynamisch
        method = getattr(self.config, 'ALIGNMENT_METHOD', 'STATIC')

        if method in ['MADGWICK', 'ESKF']:
            print(f"Nutze Alignment-Methode: {method}...")

            # Der erkannte Bewegungsstart begrenzt den Suchbereich. Dadurch
            # kann niemals eine Phase während oder nach dem Lauf als
            # Startinitialisierung ausgewählt werden.
            acc_mag = np.linalg.norm(
                df_imu[['A_x [g]', 'A_y [g]', 'A_z [g]']].to_numpy(),
                axis=1,
            )
            threshold = getattr(self.config, 'START_PEAK_THRESHOLD_G', 1.2)
            peaks = np.where(acc_mag > threshold)[0]

            if len(peaks) == 0:
                print(" -> WARNUNG: Kein Start-Peak gefunden! Falle auf STATIC zurück.")
                method = 'STATIC'
            else:
                true_start_idx = int(peaks[0])
                df_still, init_end_idx = self.find_initial_stillness(
                    df_imu,
                    fs,
                    calib,
                    search_end_idx=true_start_idx,
                )
                q_base, P0, _ = self._static_leveling_alignment(df_still, calib)
                process_start_idx = init_end_idx + 1

                if process_start_idx >= true_start_idx:
                    raise RuntimeError(
                        "Das Initialisierungsfenster endet nicht vor dem "
                        "erkannten Bewegungsstart."
                    )

                if method == 'MADGWICK':
                    df_warmup = df_imu.iloc[process_start_idx:true_start_idx]
                    q_base_scipy = q_base.as_quat()
                    q_base_ahrs = np.array([q_base_scipy[3], q_base_scipy[0], q_base_scipy[1], q_base_scipy[2]])

                    gyro_data = df_warmup[['G_x [dps]', 'G_y [dps]', 'G_z [dps]']].values
                    acc_data = df_warmup[['A_x [g]', 'A_y [g]', 'A_z [g]']].values
                    gyro_rad = np.deg2rad(gyro_data - calib.gyro_bias)
                    acc_calib = np.array([calib.calibrate_acc(a) for a in acc_data])

                    if len(gyro_rad) < 2:
                        return q_base, P0, true_start_idx, true_start_idx

                    madgwick = ahrs.filters.Madgwick(gain=0.05, frequency=fs)
                    Q = np.zeros((len(gyro_rad), 4))
                    Q[0] = q_base_ahrs

                    for i in range(1, len(gyro_rad)):
                        Q[i] = madgwick.updateIMU(Q[i-1], gyr=gyro_rad[i], acc=acc_calib[i])

                    final_q_madgwick = Q[-1]
                    q_init = R.from_quat([
                        final_q_madgwick[1],
                        final_q_madgwick[2],
                        final_q_madgwick[3],
                        final_q_madgwick[0],
                    ])

                    print(f" -> Madgwick Alignment beendet. Trajektorie startet bei {true_start_idx/fs:.2f}s")
                    return q_init, P0, true_start_idx, true_start_idx

                elif method == 'ESKF':
                    alignments = self.build_initialization_alignments(
                        df_imu,
                        calib,
                        true_start_idx,
                    )
                    selected_alignment = min(
                        alignments,
                        key=lambda candidate: (
                            candidate['candidate_id'] != str(getattr(
                                self.config,
                                'INITIAL_ATTITUDE_SOURCE',
                                'STABILIZED_RECENT_WARMUP',
                            )),
                            candidate.get('selection_priority', 1),
                            candidate['selection_score'],
                            candidate['quality_score'],
                            candidate['gyro_p95_dps'],
                            -candidate['duration_s'],
                        ),
                    )
                    selected_candidate_id = selected_alignment[
                        'candidate_id'
                    ]
                    self.initialization_report[
                        'prestart_selection_method'
                    ] = 'configured_source_then_prestart_consistency'
                    self.initialization_report[
                        'prestart_selected_candidate_id'
                    ] = selected_candidate_id
                    self.initialization_report[
                        'prestart_candidates'
                    ] = [
                        {
                            'candidate_id': candidate['candidate_id'],
                            'quality_score': candidate['quality_score'],
                            'gyro_mean_dps': candidate['gyro_mean_dps'],
                            'gyro_p95_dps': candidate['gyro_p95_dps'],
                            'prestart_score': candidate['prestart_score'],
                            'short_window_penalty': candidate[
                                'short_window_penalty'
                            ],
                            'selection_score': candidate['selection_score'],
                            'warmup_report': dict(
                                candidate['warmup_report']
                            ),
                            'selected': (
                                candidate['candidate_id']
                                == selected_candidate_id
                            ),
                        }
                        for candidate in alignments
                    ]
                    print(
                        " -> Startlagenvergleich: "
                        f"{len(alignments)} Kandidat(en) | "
                        f"gewählt={selected_candidate_id} | "
                        f"Auswahl-Score="
                        f"{selected_alignment['selection_score']:.3f}"
                    )
                    return (
                        selected_alignment['q_initial'],
                        P0,
                        true_start_idx,
                        true_start_idx,
                    )

        if method == 'STATIC':
            print("⚓ Nutze Static Leveling Alignment (Wasserwaage)...")
            df_init, init_idx = self.find_initial_stillness(
                df_imu,
                fs,
                calib,
                search_end_idx=len(df_imu),
            )
            q_init, P0, _ = self._static_leveling_alignment(df_init, calib)
            return q_init, P0, init_idx, init_idx

        raise ValueError(
            "ALIGNMENT_METHOD muss 'STATIC', 'MADGWICK' oder 'ESKF' sein."
        )

    def _static_leveling_alignment(self, df_init, calib):
        """Bestimmt Roll/Pitch robust aus der ausgewählten Gravity-Phase."""
        if df_init.empty:
            raise ValueError("Das Initialisierungsfenster darf nicht leer sein.")

        raw_accs_init = df_init[['A_x [g]', 'A_y [g]', 'A_z [g]']].values
        accs_init_calib = np.array([calib.calibrate_acc(a) for a in raw_accs_init])
        median_acc = np.median(accs_init_calib, axis=0)
        median_acc_norm = np.linalg.norm(median_acc)
        if median_acc_norm <= 1e-9:
            raise ValueError(
                "Der Gravitationsvektor im Initialisierungsfenster ist ungültig."
            )
        v_acc_body = median_acc / median_acc_norm

        v2 = np.array([0.0, 0.0, 1.0])
        axis = np.cross(v_acc_body, v2)
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(v_acc_body, v2), -1.0, 1.0))
            q_level = R.from_rotvec(axis * angle)
        else:
            q_level = R.identity()

        # Der Gyro-Bias wird hier absichtlich nicht erneut überschrieben. Die
        # adaptive Suche hat bereits unabhängig entschieden, ob eine Phase für
        # eine laufbezogene Bias-Aktualisierung hochwertig genug ist.
        P0 = float(df_init['P [hPa]'].median())
        return q_level, P0, int(df_init.index[-1])
    
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
