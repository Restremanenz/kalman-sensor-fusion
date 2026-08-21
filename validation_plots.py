import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


VARIANT_COLORS = {
    "V1_IMU": "#0072B2",
    "V2_BARO": "#E69F00",
    "V3_RTS": "#009E73",
    "V4_WALL": "#CC79A7",
    "V5_CURRENT": "#D55E00",
}

VIDEO_COLOR = "#202020"


def configure_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.5,
        "figure.constrained_layout.use": True,
    })


def save_figure(fig, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_validation_data():
    run_name = Path(config.LOG_FOLDER).resolve().name
    validation_dir = Path(config.VALIDATION_OUTPUT_DIR) / run_name

    metadata_file = validation_dir / "metadata.json"
    metrics_file = validation_dir / "metrics_summary.csv"

    if not metadata_file.exists() or not metrics_file.exists():
        raise FileNotFoundError(
            "Zuerst validation.py ausführen, damit CSV- und Metadaten vorliegen."
        )

    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    metrics = pd.read_csv(metrics_file)
    variants = {}

    for variant_name in metadata["variants"]:
        csv_file = validation_dir / f"{variant_name}.csv"
        variants[variant_name] = pd.read_csv(csv_file)

    return validation_dir, metadata, metrics, variants


def load_native_video_velocity(metadata):
    with open(metadata["video_data_file"], "r", encoding="utf-8") as file:
        video_data = json.load(file)

    fps = float(video_data.get("FPS", 60.0))

    position_z = np.asarray(
        video_data["COG"]["COG_PositionY"]["Filt"],
        dtype=float,
    )
    speed = np.asarray(
        video_data["COG"]["COG_Velocity"]["Filt"],
        dtype=float,
    )

    sample_count = min(len(position_z), len(speed))
    position_z = position_z[:sample_count]
    speed = speed[:sample_count]

    video_time = np.arange(sample_count, dtype=float) / fps
    velocity_z = np.gradient(position_z, video_time)

    video_time -= float(metadata["video_time_at_imu_start_s"])
    return video_time, velocity_z, speed


def plot_sync_velocity(output_dir, metadata, variants):
    reference_name = (
        "V2_BARO" if "V2_BARO" in variants else next(iter(variants))
    )
    reference = variants[reference_name]

    video_time, video_vz, video_speed = load_native_video_velocity(metadata)

    imu_time = reference["time_s"].to_numpy()
    imu_vz = reference["imu_vz_mps"].to_numpy()
    # Vergleich mit dem Video ausschließlich in der beobachtbaren Wandebene.
    imu_velocity_yz = reference[
        ["imu_vy_mps", "imu_vz_mps"]
    ].to_numpy()

    imu_speed_yz = np.linalg.norm(
        imu_velocity_yz,
        axis=1,
    )

    synchronization = metadata["synchronization"]
    video_timing = metadata["video_timing"]

    kinematic_start = float(
        synchronization["kinematic_video_start_s"]
    )

    signal_relative = (
        float(video_timing["derived_signal_video_time_s"])
        - kinematic_start
    )
    motion_relative = (
        float(video_timing["derived_motion_start_video_time_s"])
        - kinematic_start
    )
    json_start_relative = (
        float(video_timing["json_start_s"])
        - kinematic_start
    )
    buzzer_relative = (
        float(video_timing["json_end_s"])
        - kinematic_start
    )

    events = [
        (
            signal_relative,
            "Startsignal",
            "#666666",
        ),
        (
            motion_relative,
            "Bewegungsbeginn\n(+0,162 s)",
            "#009E73",
        ),
        (
            0.0,
            "IMU-Start\n(>2 g)",
            "#0072B2",
        ),
        (
            json_start_relative,
            "JSON-Start",
            "#CC79A7",
        ),
        (
            buzzer_relative,
            "Buzzer",
            "#D55E00",
        ),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [0.55, 2.0, 2.0],
        },
    )

    event_axis, velocity_axis, speed_axis = axes

    event_axis.hlines(
        0.0,
        signal_relative,
        buzzer_relative,
        color="0.55",
        linewidth=1.0,
    )

    for index, (event_time, label, color) in enumerate(events):
        event_axis.scatter(
            event_time,
            0.0,
            color=color,
            s=28,
            zorder=3,
        )

        vertical_offset = 10 if index % 2 == 0 else -12
        vertical_alignment = "bottom" if vertical_offset > 0 else "top"

        event_axis.annotate(
            label,
            xy=(event_time, 0.0),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=7,
            color=color,
        )

    event_axis.set_ylim(-0.35, 0.35)
    event_axis.set_yticks([])
    event_axis.grid(False)
    event_axis.set_title(
        "Zeitereignisse relativ zum kinematischen IMU-Start"
    )

    velocity_axis.plot(
        imu_time,
        imu_vz,
        label="IMU: vertikale Geschwindigkeit",
        color=VARIANT_COLORS.get(reference_name),
    )
    velocity_axis.plot(
        video_time,
        video_vz,
        label="Video: vertikale Geschwindigkeit",
        color=VIDEO_COLOR,
        linestyle="--",
    )
    velocity_axis.set_ylabel("Geschwindigkeit Z [m/s]")
    velocity_axis.set_title(
        "Kontrolle der Video-IMU-Synchronisation"
    )
    velocity_axis.legend()

    speed_axis.plot(
        imu_time,
        imu_speed_yz,
        label="IMU: YZ-Geschwindigkeitsbetrag",
        color=VARIANT_COLORS.get(reference_name),
    )
    speed_axis.plot(
        video_time,
        video_speed,
        label="Video: COG-Geschwindigkeitsbetrag",
        color=VIDEO_COLOR,
        linestyle="--",
    )
    speed_axis.set_xlabel("Zeit relativ zum IMU-Start [s]")
    speed_axis.set_ylabel("YZ-Geschwindigkeitsbetrag [m/s]")
    speed_axis.legend()

    for axis in (velocity_axis, speed_axis):
        axis.axvline(
            0.0,
            color="#0072B2",
            linewidth=0.8,
            linestyle=":",
        )
        axis.axvline(
            buzzer_relative,
            color="#D55E00",
            linewidth=0.8,
            linestyle=":",
        )

    x_min = min(signal_relative - 0.05, float(video_time.min()))
    x_max = max(
        buzzer_relative + 0.05,
        float(reference["time_s"].max()),
    )
    speed_axis.set_xlim(x_min, x_max)

    save_figure(fig, output_dir, "01_sync_velocity")


def plot_yz_trajectories(output_dir, variants):
    first_frame = next(iter(variants.values()))

    fig, ax = plt.subplots(figsize=(6.5, 7.0))

    ax.plot(
        first_frame["video_y_m"],
        first_frame["video_z_m"],
        color=VIDEO_COLOR,
        linestyle="--",
        linewidth=2.2,
        label="Video-Referenz",
        zorder=10,
    )

    for variant_name, dataframe in variants.items():
        ax.plot(
            dataframe["imu_y_m"],
            dataframe["imu_z_m"],
            color=VARIANT_COLORS.get(variant_name),
            label=variant_name,
        )

    ax.scatter(0.0, 0.0, color=VIDEO_COLOR, marker="o", s=30, zorder=11)
    ax.set_xlabel("Seitliche Bewegung Y [m]")
    ax.set_ylabel("Vertikale Bewegung Z [m]")
    ax.set_title("Rekonstruierte Trajektorie in der Wandebene")
    ax.axis("equal")
    ax.legend()

    save_figure(fig, output_dir, "02_trajectory_yz")


def plot_position_over_time(output_dir, variants):
    first_frame = next(iter(variants.values()))

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)

    axes[0].plot(
        first_frame["time_s"],
        first_frame["video_y_m"],
        color=VIDEO_COLOR,
        linestyle="--",
        linewidth=2.2,
        label="Video",
    )
    axes[1].plot(
        first_frame["time_s"],
        first_frame["video_z_m"],
        color=VIDEO_COLOR,
        linestyle="--",
        linewidth=2.2,
        label="Video",
    )

    for variant_name, dataframe in variants.items():
        color = VARIANT_COLORS.get(variant_name)

        axes[0].plot(
            dataframe["time_s"],
            dataframe["imu_y_m"],
            color=color,
            label=variant_name,
        )
        axes[1].plot(
            dataframe["time_s"],
            dataframe["imu_z_m"],
            color=color,
            label=variant_name,
        )

    axes[0].set_ylabel("Y [m]")
    axes[0].set_title("Seitliche Position")
    axes[0].legend(ncol=3)

    axes[1].set_xlabel("Zeit seit Bewegungsstart [s]")
    axes[1].set_ylabel("Z [m]")
    axes[1].set_title("Vertikale Position")

    save_figure(fig, output_dir, "03_position_time")


def plot_error_over_time(output_dir, variants):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)

    for variant_name, dataframe in variants.items():
        color = VARIANT_COLORS.get(variant_name)

        axes[0].plot(
            dataframe["time_s"],
            dataframe["error_y_m"],
            color=color,
            label=variant_name,
        )
        axes[1].plot(
            dataframe["time_s"],
            dataframe["error_z_m"],
            color=color,
            label=variant_name,
        )

    for axis in axes:
        axis.axhline(0.0, color=VIDEO_COLOR, linewidth=0.8)

    axes[0].set_ylabel("Fehler Y [m]")
    axes[0].set_title("Seitlicher Fehler gegenüber Video")
    axes[0].legend(ncol=3)

    axes[1].set_xlabel("Zeit seit Bewegungsstart [s]")
    axes[1].set_ylabel("Fehler Z [m]")
    axes[1].set_title("Vertikaler Fehler gegenüber Video")

    save_figure(fig, output_dir, "04_error_time")


def plot_error_metrics(output_dir, metrics):
    variants = metrics["variant"].tolist()
    x_positions = np.arange(len(variants))
    bar_width = 0.24

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.5),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    metric_columns = [
        ("rmse_y_m", "RMSE Y"),
        ("rmse_z_m", "RMSE Z"),
        ("rmse_yz_m", "RMSE YZ"),
    ]

    metric_colors = ["#0072B2", "#E69F00", "#009E73"]

    for index, ((column, label), color) in enumerate(
        zip(metric_columns, metric_colors)
    ):
        axes[0].bar(
            x_positions + (index - 1) * bar_width,
            metrics[column],
            width=bar_width,
            color=color,
            label=label,
        )

    axes[0].set_ylabel("RMSE [m]")
    axes[0].set_title("Fehlerkennwerte der Pipelinevarianten")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(variants)
    axes[0].legend(ncol=3)

    axes[1].bar(
        x_positions,
        metrics["max_error_yz_m"],
        width=0.55,
        color=[VARIANT_COLORS.get(name) for name in variants],
    )
    axes[1].set_xlabel("Pipelinevariante")
    axes[1].set_ylabel("Max. Fehler YZ [m]")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(variants)

    save_figure(fig, output_dir, "05_error_metrics")


def plot_xz_side_view(output_dir, variants):
    fig, ax = plt.subplots(figsize=(6.3, 7.0))

    for variant_name, dataframe in variants.items():
        ax.plot(
            dataframe["imu_x_m"],
            dataframe["imu_z_m"],
            color=VARIANT_COLORS.get(variant_name),
            label=variant_name,
        )

    ax.axvline(0.0, color=VIDEO_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Wandnormale Bewegung X [m]")
    ax.set_ylabel("Vertikale Bewegung Z [m]")
    ax.set_title("Plausibilitätsprüfung der wandnormalen Bewegung")
    ax.legend()

    save_figure(fig, output_dir, "06_side_view_xz")


def create_validation_plots():
    configure_plot_style()

    validation_dir, metadata, metrics, variants = load_validation_data()
    output_dir = validation_dir / "plots"

    plot_sync_velocity(output_dir, metadata, variants)
    plot_yz_trajectories(output_dir, variants)
    plot_position_over_time(output_dir, variants)
    plot_error_over_time(output_dir, variants)
    plot_error_metrics(output_dir, metrics)
    plot_xz_side_view(output_dir, variants)

    print(f"Validierungsplots gespeichert unter: {output_dir.resolve()}")
    return output_dir


if __name__ == "__main__":
    create_validation_plots()
