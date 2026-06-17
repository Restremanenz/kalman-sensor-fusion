import numpy as np

class PostProcessor:
    """Kapselt alle Post-Processing-Schritte nach dem Kalman-Filter (z.B. den Smoother)."""
    
    def __init__(self, config):
        self.config = config

    def apply_smoother(self, positions, velocities, times_plot, df_imu):
        """
        Wendet Randbedingungen (Boundary Conditions) am Ende des Laufs an und 
        verteilt den Rest-Drift linear rückwärts über die gesamte Trajektorie.
        """
        print("Wende physikalischen Rückwärts-Smoother an...")
        num_samples = len(velocities)
        
        # Lineare Faktoren von 0.0 (Start) bis 1.0 (Ende) für die Drift-Verteilung
        drift_factors = np.linspace(0.0, 1.0, num_samples)[:, np.newaxis]

        smoothed_velocities = velocities.copy()
        smoothed_positions = positions.copy()

        # --- A. GESCHWINDIGKEIT (Nur für komplette Runs!) ---
        # Sicherheitsabfrage: Niemals V=0 erzwingen, wenn wir mitten im Lauf abschneiden!
        force_v_zero = getattr(self.config, 'FORCE_V_END_ZERO', False)
        if getattr(self.config, 'MAX_PROCESS_TIME', None) is not None and force_v_zero:
            print(" -> WARNUNG: Geschwindigkeits-Smoother deaktiviert, da MAX_PROCESS_TIME aktiv ist!")
            force_v_zero = False

        if force_v_zero:
            print(" -> Korrigiere End-Geschwindigkeit auf 0.0 m/s")
            v_end_error = velocities[-1] - np.array([0.0, 0.0, 0.0])
            smoothed_velocities = velocities - (drift_factors * v_end_error)
            
            # Da sich die Geschwindigkeit geändert hat, müssen wir die Route neu integrieren
            smoothed_positions[0] = positions[0]
            dt_array = np.diff(times_plot)
            for i in range(1, num_samples):
                smoothed_positions[i] = smoothed_positions[i-1] + smoothed_velocities[i] * dt_array[i-1]

        # --- B. POSITION (Für alle Runs, zieht Track physikalisch korrekt an die Wand) ---
        target_pos = smoothed_positions[-1].copy() # Default: Nichts ändern

        if getattr(self.config, 'SMOOTH_XY_TO_ZERO', False):
            target_pos[0] = getattr(self.config, 'TARGET_X_M', 0.0)
            target_pos[1] = getattr(self.config, 'TARGET_Y_M', 0.0)
            print(f" -> Korrigiere End-Position X/Y auf [{target_pos[0]:.2f}, {target_pos[1]:.2f}]")
        
        if getattr(self.config, 'SMOOTH_TO_BARO_Z', False):
            # Holt sich dynamisch die absolut letzte Barometer-Höhe aus dem Datensatz
            baro_end_z = df_imu['Altitude_filt [m]'].iloc[-1]
            target_pos[2] = baro_end_z
            print(f" -> Korrigiere End-Position Z auf Barometer-Höhe: {baro_end_z:.2f}m")

        # Differenz (Fehler) zwischen ESKF-Ende und unserem physikalischen Ziel berechnen
        pos_end_error = smoothed_positions[-1] - target_pos
        
        # Fehler quadratisch über den gesamten Datensatz rückwärts abziehen
        drift_factors_squared = drift_factors ** 2
        smoothed_positions -= (drift_factors_squared * pos_end_error)

        return smoothed_positions, smoothed_velocities