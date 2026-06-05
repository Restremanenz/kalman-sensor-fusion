import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter

class FilterpyESKF15:
    """
    15-State Error-State Kalman Filter.
    Nominal: Position(3), Velocity(3), Quaternion(4), Accel-Bias(3), Gyro-Bias(3)
    Error:   d_Pos(3), d_Vel(3), d_Theta(3), d_ba(3), d_bg(3) = 15 Dimensionen
    """

    def __init__(self, initial_pos, initial_q, gyro_noise_std, accel_noise, bg_rw, ba_rw, grav_unc, zupt_unc, baro_unc, zaru_unc):
        self.p = np.array(initial_pos, dtype=float)
        self.v = np.zeros(3)
        self.q = initial_q
        self.ba = np.zeros(3)  # In-Run Accelerometer Bias
        self.bg = np.zeros(3)  # In-Run Gyroskop Bias
        self.g = np.array([0, 0, -9.81]) 

        self.kf = KalmanFilter(dim_x=15, dim_z=3) 
        self.kf.x = np.zeros(15) 
        
        # Initiale Unsicherheit (P-Matrix)
        self.kf.P = np.eye(15) * 0.01 
        self.kf.P[9:12, 9:12] *= 0.1   # Unsicherheit des Start-Accel-Bias
        self.kf.P[12:15, 12:15] *= 0.1 # Unsicherheit des Start-Gyro-Bias
        
        # Base Process Noise (Q-Matrix Kern)
        self.base_Q = np.zeros((12, 12))
        self.base_Q[0:3, 0:3] = np.eye(3) * (accel_noise)**2  
        self.base_Q[3:6, 3:6] = np.diag(gyro_noise_std**2)
        self.base_Q[6:9, 6:9] = np.eye(3) * (ba_rw)**2
        self.base_Q[9:12, 9:12] = np.eye(3) * (bg_rw)**2

        self.R_grav = np.eye(3) * (grav_unc)**2  
        self.R_zupt = np.eye(3) * (zupt_unc)**2
        self.R_baro = np.eye(3) * 1e8          # Unendlich hohes Rauschen für X und Y
        self.R_baro[2, 2] = baro_unc**2        # Echtes (niedriges) Rauschen für Z
        self.R_zaru = np.eye(3) * (zaru_unc)**2

    def update_barometer(self, baro_z):
        """Update der Z-Achse durch das Barometer (3D Padding für FilterPy)."""
        # H-Matrix muss nun 3 Zeilen (für 3D-Messung) und 15 Spalten haben
        H = np.zeros((3, 15))
        H[2, 2] = 1.0  # Wir mappen nur die Z-Position (Index 2) auf die Z-Messung (Reihe 2)
        
        # Innovation (Messung - Vorhersage)
        # X und Y setzen wir Nullen (werden vom Filter wegen R=1e8 komplett ignoriert)
        innovation = np.array([0.0, 0.0, baro_z - self.p[2]])
        
        self.kf.update(z=innovation, R=self.R_baro, H=H)
        self._inject_error_and_reset()

    def predict(self, acc_meas, gyro_meas, dt):
        """Koppelnavigation und Fehlerpropagierung mit Bias-Kompensation."""
        # Wahre Sensorwerte durch Abzug des geschätzten In-Run Bias
        acc_true = acc_meas - self.ba
        gyro_true = gyro_meas - self.bg

        acc_world = self.q.apply(acc_true)
        R_mat = self.q.as_matrix()

        # Nominal State Update
        self.p += self.v * dt + 0.5 * (acc_world + self.g) * dt**2
        self.v += (acc_world + self.g) * dt
        self.q = self.q * R.from_rotvec(gyro_true * dt)

        # Jacobian: Fehler-Zustands-Übergangsmatrix (F_x)
        F_x = np.eye(15)
        F_x[0:3, 3:6] = np.eye(3) * dt
        
        acc_skew = np.array([
            [0, -acc_world[2], acc_world[1]],
            [acc_world[2], 0, -acc_world[0]],
            [-acc_world[1], acc_world[0], 0]
        ])
        F_x[3:6, 6:9] = -acc_skew * dt            # Einfluss Orientierungsfehler auf Speed
        F_x[3:6, 9:12] = -R_mat * dt              # Einfluss Acc-Bias auf Speed
        F_x[6:9, 12:15] = -R_mat * dt             # Einfluss Gyro-Bias auf Orientierung (Globaler Fehler)

        # Jacobian: Noise-Mapping Matrix (F_i)
        F_i = np.zeros((15, 12))
        F_i[3:6, 0:3] = R_mat * dt                # Acc Noise -> Velocity
        F_i[6:9, 3:6] = R_mat * dt                # Gyro Noise -> Orientation
        F_i[9:12, 6:9] = np.eye(3) * dt           # Acc RW -> Acc Bias
        F_i[12:15, 9:12] = np.eye(3) * dt         # Gyro RW -> Gyro Bias

        self.kf.F = F_x
        self.kf.Q = F_i @ self.base_Q @ F_i.T
        self.kf.predict()

    def _inject_error_and_reset(self):
        """Injiziert den Error-State und setzt Fehler zurück."""
        self.p += self.kf.x[0:3]
        self.v += self.kf.x[3:6]
        
        error_rot = R.from_rotvec(self.kf.x[6:9])
        self.q = error_rot * self.q
        
        self.ba += self.kf.x[9:12]
        self.bg += self.kf.x[12:15]
        
        self.kf.x = np.zeros(15)

    def update_zupt(self):
        """Zero Velocity Update (Geschwindigkeit = 0)."""
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3) 
        
        innovation = np.array([0.0, 0.0, 0.0]) - self.v
        self.kf.update(z=innovation, R=self.R_zupt, H=H)
        self._inject_error_and_reset()

    def update_zaru(self, gyro_meas):
        """
        Zero Angular Rate Update (ZARU).
        Zwingt die Drehrate im Stillstand auf 0 und eliminiert dadurch den Yaw-Drift (Gieren).
        """
        H = np.zeros((3, 15))
        # Die Beobachtung mappt direkt negativ auf den Gyroskop-Bias (Index 12 bis 15 im Error-State)
        H[0:3, 12:15] = -np.eye(3) 
        
        # Innovation: Wahre Drehrate (0.0) minus aktuell geschätzte Drehrate
        gyro_true = gyro_meas - self.bg
        innovation = np.array([0.0, 0.0, 0.0]) - gyro_true
        
        self.kf.update(z=innovation, R=self.R_zaru, H=H)
        self._inject_error_and_reset()

    def update_gravity(self, acc_meas):
        """Nutzt Beschleunigung im Stillstand als Wasserwaage."""
        acc_true = acc_meas - self.ba
        acc_norm = np.linalg.norm(acc_true)
        
        if abs(acc_norm - 9.81) > 0.5:
            return 

        a_dir = acc_true / acc_norm
        R_mat_inv = self.q.inv().as_matrix()
        g_pred = R_mat_inv @ np.array([0, 0, 1]) # g_global ist [0,0,1]

        H = np.zeros((3, 15))
        
        # d(Z) / d(Theta_global) = R^T * [g_global]_x
        g_glob_skew = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        H[0:3, 6:9] = R_mat_inv @ g_glob_skew
        
        # d(Z) / d(b_a) = - (1 / ||a||) * I
        H[0:3, 9:12] = -np.eye(3) / acc_norm

        innovation = a_dir - g_pred
        self.kf.update(z=innovation, R=self.R_grav, H=H)
        self._inject_error_and_reset()