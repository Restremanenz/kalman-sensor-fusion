import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection

class TrajectoryVisualizer:
    """Kapselt alle Matplotlib-Plots und 3D-Animationen"""
    
    def __init__(self, config):
        # Konfiguration als Instanzvariable speichern
        self.config = config
        self.fps = getattr(config, 'ANIMATION_FPS', 30)
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
        """Plottet die rohen IMU- und Magnetometer-Daten in drei verknüpften Subplots."""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

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
        ax2.set_ylabel('Drehgeschwindigkeit [dps]')
        ax2.set_title('Gyroskop Rohdaten')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # 3. Magnetometer Plot 
        if 'M_x [G]' in df.columns:
            ax3.plot(df['Time'], df['M_x [G]'], label='M_x', color='#d62728', alpha=0.8)
            ax3.plot(df['Time'], df['M_y [G]'], label='M_y', color='#2ca02c', alpha=0.8)
            ax3.plot(df['Time'], df['M_z [G]'], label='M_z', color='#1f77b4', alpha=0.8)
            ax3.set_ylabel('Magnetische Induktion [G]')
            ax3.set_title('Magnetometer Rohdaten')
            ax3.legend(loc='upper right')
            ax3.grid(True, linestyle='--', alpha=0.6)
        else:
            # Falls alte Logs ohne Mag geladen werden
            ax3.text(0.5, 0.5, 'Keine Magnetometerdaten im DataFrame vorhanden', 
                     horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_ylabel('Magnetfeld [G]')

        # Das X-Label gehört jetzt an die Achse des untersten Plots
        ax3.set_xlabel('Zeit [s]')
        
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

    def plot_2d_wall_with_trajectory(self, positions, velocities):
        """
        Erstellt einen 2D-Plot (Frontalansicht: Y-Achse = Breite, Z-Achse = Höhe).
        Nutzt Parameter aus der globalen config.py.
        """
        fig, ax = plt.subplots(figsize=(5, 12)) 
        
        image_path = getattr(self.config, 'VIS_WALL_BG_IMAGE', "speedwall_2D.png")
        wall_length = getattr(self.config, 'VIS_WALL_LENGTH', 15.0)

        try:
            img = plt.imread(image_path)
            extent = [-1.5, 1.5, 0.0, wall_length]
            ax.imshow(img, extent=extent, origin='upper', alpha=0.6)
        except Exception as e:
            print(f"[WARNUNG] Konnte Hintergrundbild ({image_path}) nicht laden: {e}")

        # Sensordaten anpassen (Kopie!)
        plot_pos = positions.copy()

        if getattr(self.config, 'VIS_MIRROR_Y', True):
            plot_pos[:, 1] = -plot_pos[:, 1]

        plot_pos[:, 1] += getattr(self.config, 'VIS_SENSOR_OFFSET_Y', 0.3)
        plot_pos[:, 2] += getattr(self.config, 'VIS_SENSOR_START_Z', 1.1)

        # Geschwindigkeit berechnen
        v_abs = np.linalg.norm(velocities, axis=1)
        points = np.array([plot_pos[:, 1], plot_pos[:, 2]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        v_abs_segments = (v_abs[:-1] + v_abs[1:]) / 2.0 

        # Farbige Linie
        colormap = getattr(self.config, 'VIS_COLORMAP', 'turbo')
        norm = plt.Normalize(v_abs_segments.min(), v_abs_segments.max())
        lc = LineCollection(segments, cmap=colormap, norm=norm)
        lc.set_array(v_abs_segments)
        lc.set_linewidth(3.5)

        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Geschwindigkeit [m/s]', rotation=270, labelpad=15)
        
        ax.scatter(plot_pos[0, 1], plot_pos[0, 2], color='green', s=120, zorder=5, label='Start', edgecolors='white')
        ax.scatter(plot_pos[-1, 1], plot_pos[-1, 2], color='red', s=120, zorder=5, label='Ende', edgecolors='white')

        ax.set_aspect('equal')
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-0.5, wall_length + 0.5])
        
        ax.set_xlabel('Y-Achse (Breite) [m]')
        ax.set_ylabel('Z-Achse (Höhe) [m]')
        ax.set_title('2D Speedwand-Analyse (Frontalansicht)')
        
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

    def plot_2d_side_view_with_trajectory(self, positions, velocities):
        """
        Erstellt einen 2D-Plot der Seitenansicht (X-Achse = Tiefe, Z-Achse = Höhe).
        Nutzt Parameter aus der globalen config.py und spiegelt die Ansicht bei Bedarf.
        """
        fig, ax = plt.subplots(figsize=(5, 12)) 
        
        plot_pos = positions.copy()
        
        # Spiegelung und Wand-Richtung 
        if getattr(self.config, 'VIS_MIRROR_X', True):
            # Kletterer in den positiven Bereich (+X) spiegeln
            plot_pos[:, 0] = -plot_pos[:, 0]
            # Wand muss nach rechts (+X) überhängen, um über dem Kletterer zu sein
            wall_direction = 1.0  
        else:
            # Originaldaten belassen (-X)
            # Wand muss nach links (-X) überhängen
            wall_direction = -1.0 
            
        # Offsets anwenden
        plot_pos[:, 2] += getattr(self.config, 'VIS_SENSOR_START_Z', 1.1)
        plot_pos[:, 0] += getattr(self.config, 'VIS_SENSOR_OFFSET_X', 0.2)

        wall_length = getattr(self.config, 'VIS_WALL_LENGTH', 15.0)
        wall_thickness = getattr(self.config, 'VIS_WALL_THICKNESS', 0.10)
        angle_rad = np.radians(getattr(self.config, 'WALL_INCLINATION_DEG', 5.0))
        
        # Wand berechnen (nutzt wall_direction für den korrekten Überhang)
        x_wall_front = np.array([0, wall_direction * wall_length * np.sin(angle_rad)])
        z_wall = np.array([0, wall_length * np.cos(angle_rad)])
        
        # Wenn der Überhang nach rechts (+X) kippt, muss die Dicke nach links (-X) gehen.
        x_wall_back = x_wall_front - (wall_direction * wall_thickness)
        
        # Wand zeichnen (von der Rückseite zur Vorderseite füllen)
        ax.fill_betweenx(z_wall, x_wall_back, x_wall_front, color='#555555', label=f'Wand ({np.degrees(angle_rad):.0f}°)', zorder=1)
        
        # Trajektorie und Geschwindigkeit
        v_abs = np.linalg.norm(velocities, axis=1)
        points = np.array([plot_pos[:, 0], plot_pos[:, 2]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        v_abs_segments = (v_abs[:-1] + v_abs[1:]) / 2.0 

        colormap = getattr(self.config, 'VIS_COLORMAP', 'turbo')
        norm = plt.Normalize(v_abs_segments.min(), v_abs_segments.max())
        lc = LineCollection(segments, cmap=colormap, norm=norm)
        lc.set_array(v_abs_segments)
        lc.set_linewidth(3.5)

        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Geschwindigkeit [m/s]', rotation=270, labelpad=15)
        
        ax.scatter(plot_pos[0, 0], plot_pos[0, 2], color='green', s=120, zorder=5, label='Start', edgecolors='white')
        ax.scatter(plot_pos[-1, 0], plot_pos[-1, 2], color='red', s=120, zorder=5, label='Ende', edgecolors='white')

        ax.set_aspect('equal')
        
        # Symmetrische Limits, damit beide Varianten (gespiegelt oder nicht) gut aussehen
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-0.5, wall_length + 0.5])
        
        ax.set_xlabel('X-Achse (Tiefe / Wandabstand) [m]')
        ax.set_ylabel('Z-Achse (Höhe) [m]')
        ax.set_title('2D Speedwand-Analyse (Seitenansicht)')
        
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

    def plot_hip_rotation(self, times, orientations):
        """Plottet die Hüftrotation (Euler-Winkel) über die Zeit."""
        
        # Konvertiere die Liste der Scipy-Rotationsobjekte in Euler-Winkel (in Grad)
        # 'ZYX' entspricht Yaw, Pitch, Roll
        euler_angles = np.array([q.as_euler('ZYX', degrees=True) for q in orientations])
        
        yaw = euler_angles[:, 0]
        pitch = euler_angles[:, 1]
        roll = euler_angles[:, 2]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # 1. Yaw (Z-Achse) - Die wichtigste Metrik für Speedklettern!
        ax1.plot(times, yaw, color='#9467bd', linewidth=2, label='Yaw (Z-Rotation)')
        ax1.set_title('Hüft-Eindrehung (Yaw) - Linke vs. Rechte Hüfte zur Wand')
        ax1.set_ylabel('Winkel [°]')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='black', linewidth=1, alpha=0.5, linestyle='-') # 0° Referenz
        ax1.legend(loc='upper right')

        # 2. Pitch (Y-Achse)
        ax2.plot(times, pitch, color='#8c564b', linewidth=2, label='Pitch (Y-Rotation)')
        ax2.set_title('Beugung (Pitch) - Hüfte an die Wand drücken vs. Durchhängen')
        ax2.set_ylabel('Winkel [°]')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')

        # 3. Roll (X-Achse)
        ax3.plot(times, roll, color='#e377c2', linewidth=2, label='Roll (X-Rotation)')
        ax3.set_title('Seitliches Kippen (Roll) - Becken-Tilt')
        ax3.set_ylabel('Winkel [°]')
        ax3.set_xlabel('Zeit [s]')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper right')

        plt.tight_layout()

    def plot_2d_wall_with_yaw(self, positions, orientations):
        """
        Erstellt die 2D-Frontalansicht, färbt die Trajektorie aber nach der 
        Hüft-Eindrehung (Yaw-Winkel) statt nach Geschwindigkeit ein.
        """
        fig, ax = plt.subplots(figsize=(5, 12)) 
        
        # Hintergrundbild laden (exakt wie in deiner anderen Methode)
        image_path = getattr(self.config, 'VIS_WALL_BG_IMAGE', "speedwall_2D.png")
        wall_length = getattr(self.config, 'VIS_WALL_LENGTH', 15.0)

        try:
            img = plt.imread(image_path)
            extent = [-1.5, 1.5, 0.0, wall_length]
            # origin='upper' falls das Bild vorher auf dem Kopf stand!
            ax.imshow(img, extent=extent, origin='upper', alpha=0.6) 
        except Exception as e:
            pass

        # Positionen kopieren und Offsets anwenden
        plot_pos = positions.copy()
        if getattr(self.config, 'VIS_MIRROR_Y', True):
            plot_pos[:, 1] = -plot_pos[:, 1]
        
        plot_pos[:, 1] += getattr(self.config, 'VIS_SENSOR_OFFSET_Y', 0.3)
        plot_pos[:, 2] += getattr(self.config, 'VIS_SENSOR_START_Z', 1.1)

        # --- NEU: Yaw-Winkel aus Orientierungen berechnen ---
        euler_angles = np.array([q.as_euler('ZYX', degrees=True) for q in orientations])
        yaw = euler_angles[:, 0]  # Z-Rotation
        
        # Segmente für den Linien-Plot vorbereiten
        points = np.array([plot_pos[:, 1], plot_pos[:, 2]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        yaw_segments = (yaw[:-1] + yaw[1:]) / 2.0 

        # --- NEU: Diverging Colormap (Blau - Weiß - Rot) ---
        # Wir fixieren die Skala symmetrisch, z.B. von -45° bis +45°
        max_angle = 60.0  # Erhöhe diesen Wert, falls der Kletterer sich weiter als 60° dreht
        norm = plt.Normalize(vmin=-max_angle, vmax=max_angle)
        
        # 'coolwarm' oder 'bwr' (Blue-White-Red) eignen sich perfekt
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(yaw_segments)
        lc.set_linewidth(4.0)

        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hüft-Eindrehung (Yaw) [°]\nBlau = Linke Hüfte | Rot = Rechte Hüfte', rotation=270, labelpad=25)
        
        # Start und Ende markieren
        ax.scatter(plot_pos[0, 1], plot_pos[0, 2], color='green', s=120, zorder=5, label='Start', edgecolors='white')
        ax.scatter(plot_pos[-1, 1], plot_pos[-1, 2], color='red', s=120, zorder=5, label='Ende', edgecolors='white')

        ax.set_aspect('equal')
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-0.5, wall_length + 0.5])
        
        ax.set_xlabel('Y-Achse (Breite) [m]')
        ax.set_ylabel('Z-Achse (Höhe) [m]')
        ax.set_title('Rotation-Heatmap: Hüft-Eindrehung an der Wand')
        
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()