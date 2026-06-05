import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class TrajectoryVisualizer:
    """Kapselt alle Matplotlib-Plots und 3D-Animationen"""
    
    def __init__(self, animation_fps=30):
        self.fps = animation_fps
        self.animations = [] 

    def plot_auto_init(self, df_plot, gyro_mag_plot, df_init, gyro_mag_init, threshold):
        """Zeigt den Plot zur Überprüfung der gefundenen Ruhephase an."""
        plt.figure(figsize=(12, 5))
        plt.plot(df_plot['Time'], gyro_mag_plot, color='lightgray', label='Gyroskop Magnitude (Roh)')
        plt.plot(df_init['Time'], gyro_mag_init, color='blue', linewidth=2, label='Extrahierte Ruhephase')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold} dps)')
        plt.title('Erkennung der initialen Ruhephase')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Magnitude (dps)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show() # Dieser Plot pausiert das Skript absichtlich!

    def plot_static_trajectory(self, positions):
        """Erstellt den statischen 3D-Plot der Route."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='KF Trajektorie', color='b')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='Ende')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Trajektorie')
        ax.legend()

    def plot_animated_trajectory(self, positions, orientations, fs):
        """Erstellt die 3D-Animation mit dem rotierenden Sensor-Würfel."""
        step = max(1, int(fs / self.fps))
        num_samples = len(positions)
        frames = range(0, num_samples, step)

        # Dimensionen des Sensor-Würfels
        hx, hy, hz = 0.15, 0.10, 0.03 
        base_vertices = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]
        ])
        face_indices = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]
        face_colors = ['#888888', '#4287f5', '#f54242', '#42f569', '#f5e042', '#f58d42']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        initial_faces = [base_vertices[idx] for idx in face_indices]
        poly3d = Poly3DCollection(initial_faces, facecolors=face_colors, edgecolors='black', linewidths=1.0, alpha=0.9)
        ax.add_collection3d(poly3d)

        traj_line_anim, = ax.plot([], [], [], color='blue', linestyle='-', linewidth=2, label='ESKF Trajektorie')
        ax.legend(loc='upper left')

        max_range = np.max(np.abs(positions))
        limit = max(0.5, max_range * 1.5) 
        
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Animierte Kalman Trajektorie')

        def update(frame):
            rot = orientations[frame]
            rotated_verts = rot.apply(base_vertices)
            translated_verts = rotated_verts + positions[frame]
            
            new_faces = [translated_verts[idx] for idx in face_indices]
            poly3d.set_verts(new_faces)
            
            hist_idx = list(range(0, frame, max(1, int(step/2))))
            if len(hist_idx) > 0:
                traj_line_anim.set_data_3d(positions[hist_idx, 0], positions[hist_idx, 1], positions[hist_idx, 2])
                
            return [poly3d, traj_line_anim]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/self.fps, blit=False)
        self.animations.append(ani) 
    def plot_raw_sensor_data(self, df):
        """Plottet die rohen IMU-Daten in zwei verknüpften Subplots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 1. Accelerometer Plot
        ax1.plot(df['Time'], df['A_x [g]'], label='A_x', color='#d62728', alpha=0.8) # Rot
        ax1.plot(df['Time'], df['A_y [g]'], label='A_y', color='#2ca02c', alpha=0.8) # Grün
        ax1.plot(df['Time'], df['A_z [g]'], label='A_z', color='#1f77b4', alpha=0.8) # Blau
        ax1.set_ylabel('Beschleunigung [g]')
        ax1.set_title('Accelerometer Rohdaten')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 2. Gyroskop Plot
        ax2.plot(df['Time'], df['G_x [dps]'], label='G_x', color='#d62728', alpha=0.8)
        ax2.plot(df['Time'], df['G_y [dps]'], label='G_y', color='#2ca02c', alpha=0.8)
        ax2.plot(df['Time'], df['G_z [dps]'], label='G_z', color='#1f77b4', alpha=0.8)
        ax2.set_xlabel('Zeit [s]')
        ax2.set_ylabel('Drehgeschwindigkeit [dps]')
        ax2.set_title('Gyroskop Rohdaten')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        
    def plot_velocity(self, times, velocities):
        """Plottet die vom ESKF berechnete Geschwindigkeit in x, y, z."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Geschwindigkeiten plotten
        ax.plot(times, velocities[:, 0], label='V_x', color='#d62728', alpha=0.8) # Rot
        ax.plot(times, velocities[:, 1], label='V_y', color='#2ca02c', alpha=0.8) # Grün
        ax.plot(times, velocities[:, 2], label='V_z', color='#1f77b4', alpha=0.8) # Blau

        ax.set_xlabel('Zeit [s]')
        ax.set_ylabel('Geschwindigkeit [m/s]')
        ax.set_title('Geschwindigkeitsverlauf (ESKF)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
    def plot_altitude(self, df):
        """Plottet die Barometer-Höhenkurve (Roh vs. Zero-Phase gefiltert)."""
        fig, ax = plt.subplots(figsize=(12, 5))

        # 1. Rohe berechnete Höhe (aus P [hPa])
        if 'Altitude [m]' in df.columns:
            ax.plot(df['Time'], df['Altitude [m]'], color='lightgray', label='Höhe (Roh / Rauschen)', alpha=0.9)
        
        # 2. Gefilterte Höhe (Zero-Phase Butterworth)
        if 'Altitude_filt [m]' in df.columns:
            ax.plot(df['Time'], df['Altitude_filt [m]'], color='#1f77b4', linewidth=2, label='Höhe (Gefiltert - 2Hz)')

        ax.set_xlabel('Zeit [s]')
        ax.set_ylabel('Relative Höhe [m]')
        ax.set_title('Barometer: Berechnetes Höhenprofil')
        
        # Schöne y-Achsen-Skalierung (Start ist 0)
        ax.set_ylim([-1.0, 16.0]) # Speed Klettern geht bis ~15m
        
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
    def show_all(self):
        """Öffnet alle vorbereiteten Fenster gleichzeitig."""
        plt.show()