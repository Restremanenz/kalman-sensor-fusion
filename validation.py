import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import config
from Kalman import main as run_kalman


def calculate_error_metrics(error_y, error_z):
    """Berechnet die in der Methodik definierten Fehlerkennwerte."""
    error_y = np.asarray(error_y, dtype=float)
    error_z = np.asarray(error_z, dtype=float)
    error_yz = np.hypot(error_y, error_z)

    valid = np.isfinite(error_y) & np.isfinite(error_z)
    if not np.any(valid):
        raise ValueError("Keine gültigen IMU-Video-Vergleichspunkte vorhanden.")

    return {
        'rmse_y_m': float(np.sqrt(np.mean(error_y[valid] ** 2))),
        'rmse_z_m': float(np.sqrt(np.mean(error_z[valid] ** 2))),
        'rmse_yz_m': float(np.sqrt(np.mean(error_yz[valid] ** 2))),
        'max_error_yz_m': float(np.max(error_yz[valid])),
        'sample_count': int(np.sum(valid)),
    }


def interpolate_video_reference(result):
    """Interpoliert die unabhängige Videoreferenz auf die IMU-Zeitpunkte.

    Der Vergleich erfolgt relativ zum gemeinsamen Bewegungsstart. Absolute
    Unterschiede der Sensor- und Video-Startposition beeinflussen dadurch die
    Bewegungsfehler nicht.
    """
    imu_times = np.asarray(result['times'], dtype=float)
    video_times = result['video_times']
    video_positions = result['video_positions']
    offset = result['video_time_offset']

    if video_times is None or video_positions is None or offset is None:
        raise ValueError(
            "Videodaten oder Zeitversatz fehlen. USE_VIDEO_DATA muss True sein."
        )

    video_times = np.asarray(video_times, dtype=float)
    video_positions = np.asarray(video_positions, dtype=float)
    video_evaluation_times = imu_times - float(offset)
    valid = (
        (video_evaluation_times >= video_times[0])
        & (video_evaluation_times <= video_times[-1])
    )

    if not np.any(valid):
        raise ValueError("IMU und Video besitzen nach dem Sync keinen Überlapp.")

    video_y = np.full(imu_times.shape, np.nan, dtype=float)
    video_z = np.full(imu_times.shape, np.nan, dtype=float)
    video_y[valid] = np.interp(
        video_evaluation_times[valid], video_times, video_positions[:, 1]
    )
    video_z[valid] = np.interp(
        video_evaluation_times[valid], video_times, video_positions[:, 2]
    )

    # Referenzbewegung relativ zu dem Videozeitpunkt, der dem ersten
    # ausgegebenen IMU-Zustand entspricht.
    video_start_time = imu_times[0] - float(offset)
    if video_start_time < video_times[0] or video_start_time > video_times[-1]:
        raise ValueError("Der IMU-Start liegt außerhalb der Videoreferenz.")
    video_start_y = np.interp(
        video_start_time, video_times, video_positions[:, 1]
    )
    video_start_z = np.interp(
        video_start_time, video_times, video_positions[:, 2]
    )

    video_y[valid] -= video_start_y
    video_z[valid] -= video_start_z
    return video_y, video_z, valid


def create_variant_dataframe(result):
    """Erzeugt Zeitreihen und IMU-Video-Fehler für eine Pipelinevariante."""
    times = np.asarray(result['times'], dtype=float)
    positions = np.asarray(result['positions_local'], dtype=float)
    velocities = np.asarray(result['velocities'], dtype=float)

    video_y, video_z, valid = interpolate_video_reference(result)
    error_y = np.full(times.shape, np.nan, dtype=float)
    error_z = np.full(times.shape, np.nan, dtype=float)
    error_y[valid] = positions[valid, 1] - video_y[valid]
    error_z[valid] = positions[valid, 2] - video_z[valid]
    error_yz = np.hypot(error_y, error_z)

    dataframe = pd.DataFrame({
        'time_s': times - times[0],
        'imu_x_m': positions[:, 0],
        'imu_y_m': positions[:, 1],
        'imu_z_m': positions[:, 2],
        'imu_vx_mps': velocities[:, 0],
        'imu_vy_mps': velocities[:, 1],
        'imu_vz_mps': velocities[:, 2],
        'video_y_m': video_y,
        'video_z_m': video_z,
        'error_y_m': error_y,
        'error_z_m': error_z,
        'error_yz_m': error_yz,
    })
    metrics = calculate_error_metrics(error_y, error_z)
    return dataframe, metrics


def run_validation():
    """Führt alle konfigurierten Varianten mit derselben Videoreferenz aus."""
    if not getattr(config, 'USE_VIDEO_DATA', False):
        raise ValueError(
            "Für die Validierung muss USE_VIDEO_DATA in config.py True sein."
        )

    variant_names = list(getattr(config, 'VALIDATION_PIPELINE_VARIANTS', []))
    if not variant_names:
        raise ValueError("VALIDATION_PIPELINE_VARIANTS darf nicht leer sein.")

    run_name = Path(config.LOG_FOLDER).resolve().name
    output_dir = Path(config.VALIDATION_OUTPUT_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_video_offset = None
    imu_start_time = None
    imu_duration = None
    summary_rows = []
    variant_options = {}
    initialization_report = None
    shared_initial_attitude_solution = None

    for variant_name in variant_names:
        print("\n" + "#" * 64)
        print(f"VALIDIERUNG: {variant_name}")
        print("#" * 64)

        result = run_kalman(
            prepare_plots=False,
            pipeline_variant=variant_name,
            fixed_video_offset=shared_video_offset,
            fixed_initial_attitude_solution=shared_initial_attitude_solution,
        )
        if shared_video_offset is None:
            shared_video_offset = float(result['video_time_offset'])
            imu_start_time = float(result['times'][0])
            imu_duration = float(result['times'][-1] - result['times'][0])
            initialization_report = result.get('initialization')
            shared_initial_attitude_solution = result.get(
                'initial_attitude_solution'
            )

        dataframe, metrics = create_variant_dataframe(result)
        dataframe.to_csv(output_dir / f"{variant_name}.csv", index=False)

        summary_rows.append({
            'variant': variant_name,
            **metrics,
        })
        variant_options[variant_name] = asdict(result['options'])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    attitude_report = (
        initialization_report or {}
    ).get('attitude_optimization', {})
    candidate_rows = []
    for candidate in attitude_report.get('candidate_baselines', []):
        endpoint = candidate.get('endpoint_local_m', [np.nan] * 3)
        components = candidate.get('cost_components', {})
        warmup = candidate.get('warmup_report', {})
        candidate_rows.append({
            'candidate_id': candidate.get('candidate_id'),
            'roles': '|'.join(candidate.get('roles', [])),
            'selected': bool(candidate.get('selected', False)),
            'duration_s': candidate.get('duration_s'),
            'quality_score': candidate.get('quality_score'),
            'gyro_mean_dps': candidate.get('gyro_mean_dps'),
            'gyro_p95_dps': candidate.get('gyro_p95_dps'),
            'prestart_score': candidate.get('prestart_score'),
            'selection_score': candidate.get('selection_score'),
            'warmup_mode': warmup.get('mode'),
            'gravity_update_count': warmup.get(
                'gravity_update_count'
            ),
            'strict_stationary_update_count': warmup.get(
                'strict_stationary_update_count'
            ),
            'trajectory_cost': candidate.get('trajectory_cost'),
            'endpoint_x_local_m': endpoint[0],
            'endpoint_y_local_m': endpoint[1],
            'endpoint_z_local_m': endpoint[2],
            'maximum_abs_lateral_m': candidate.get(
                'maximum_abs_lateral_m'
            ),
            'barometer_cost': components.get('barometer'),
            'wall_cost': components.get('wall'),
            'corridor_cost': components.get('corridor'),
            'endpoint_cost': components.get('endpoint'),
            'lateral_shape_cost': components.get('lateral_shape'),
        })
    if candidate_rows:
        pd.DataFrame(candidate_rows).to_csv(
            output_dir / 'initialization_candidates.csv',
            index=False,
        )

    with open(config.VIDEO_DATA_FILE, "r", encoding="utf-8") as file:
        video_metadata = json.load(file)

    video_time_data = video_metadata["Time"]
    contact_data = video_metadata.get("Contacts", {})

    json_start_time = float(video_time_data["Start"])
    json_run_duration = float(video_time_data["Overall"])
    json_end_time = json_start_time + json_run_duration

    kinematic_video_start = imu_start_time - shared_video_offset
    mapped_imu_end = kinematic_video_start + imu_duration

    run_timing = getattr(config, "VALIDATION_RUN_TIMING", {}).get(
        run_name,
        {},
    )

    reaction_time = run_timing.get("reaction_time_s")
    official_finish_time = run_timing.get("finish_time_s")

    official_signal_video_time = None
    official_motion_video_time = None
    official_motion_duration = None

    if reaction_time is not None and official_finish_time is not None:
        reaction_time = float(reaction_time)
        official_finish_time = float(official_finish_time)

        # Annahme: Das im JSON gespeicherte Ende entspricht dem Buzzer.
        official_signal_video_time = json_end_time - official_finish_time
        official_motion_video_time = (
            official_signal_video_time + reaction_time
        )
        official_motion_duration = official_finish_time - reaction_time

    metadata = {
        'log_folder': str(Path(config.LOG_FOLDER).resolve()),
        'video_data_file': str(Path(config.VIDEO_DATA_FILE).resolve()),
        'video_time_offset_s': shared_video_offset,
        'imu_start_time_s': imu_start_time,
        'video_time_at_imu_start_s': imu_start_time - shared_video_offset,
        'synchronization': {
            'method': 'SYNC_REFERENCE_PASS_VERTICAL_VELOCITY',
            'video_time_offset_s': shared_video_offset,
            'kinematic_video_start_s': kinematic_video_start,
            'mapped_imu_end_video_s': mapped_imu_end,
            'imu_duration_s': imu_duration,
        },
        'video_timing': {
            'json_start_s': json_start_time,
            'json_run_duration_s': json_run_duration,
            'json_end_s': json_end_time,
            'json_end_assumed_as_buzzer': True,
            'official_reaction_time_s': reaction_time,
            'official_signal_to_buzzer_s': official_finish_time,
            'official_motion_to_buzzer_s': official_motion_duration,
            'derived_signal_video_time_s': official_signal_video_time,
            'derived_motion_start_video_time_s': official_motion_video_time,
            'kinematic_minus_motion_start_s': (
                None
                if official_motion_video_time is None
                else kinematic_video_start - official_motion_video_time
            ),
            'json_start_minus_motion_start_s': (
                None
                if official_motion_video_time is None
                else json_start_time - official_motion_video_time
            ),
            'mapped_imu_end_minus_json_end_s': (
                mapped_imu_end - json_end_time
            ),
            'imu_duration_minus_official_motion_duration_s': (
                None
                if official_motion_duration is None
                else imu_duration - official_motion_duration
            ),
        },
        'video_contact_splits': {
            'split_times_s': contact_data.get("SplitTimes", []),
            'split_times_relative_s': contact_data.get(
                "SplitTimes_Rel",
                [],
            ),
            'frame_numbers': contact_data.get("FrameNumbers", []),
        },
        'wall_frame_base_yaw_deg': float(config.WALL_FRAME_BASE_YAW_DEG),
        'start_pose_yaw_correction_deg': float(
            config.START_POSE_YAW_CORRECTION_DEG
        ),
        'sensor_start_position_wall_m': list(
            map(float, config.SENSOR_START_POSITION_WALL_M)
        ),
        'target_position_local_xy_m': [
            float(config.TARGET_X_M),
            float(config.TARGET_Y_M),
        ],
        'initialization': initialization_report,
        'variants': variant_options,
    }
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\nVALIDIERUNG ABGESCHLOSSEN")
    print(summary.to_string(index=False))
    print(f"\nAusgabeordner: {output_dir.resolve()}")
    return summary, output_dir


if __name__ == "__main__":
    run_validation()
