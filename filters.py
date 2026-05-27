import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter

class FilterpyESKF:
    """
    Error-State Kalman Filter basierend auf filterpy.
    Nominaler Zustand: Position (3), Velocity (3), Quaternion (4)
    Error Zustand: d_Pos (3), d_Vel (3), d_Theta (3) = 9 Dimensionen
    """

    def __init__(self, initial_pos, gyro_noise_std, accel_noise, grav_unc, zupt_unc):
        self.p = np.array(initial_pos, dtype=float)
        self.v = np.zeros(3)
        self.q = R.from_quat([0, 0, 0, 1])
        self.g = np.array([0, 0, -9.81]) 

        self.kf = KalmanFilter(dim_x=9, dim_z=3) 
        self.kf.x = np.zeros(9) 
        self.kf.P = np.eye(9) * 0.01 
        
        self.base_Q = np.eye(6)

        self.base_Q[0:3, 0:3] *= (accel_noise)**2  
        self.base_Q[3:6, 3:6] = np.diag(gyro_noise_std**2)

        self.R_grav = np.eye(3) * (grav_unc)**2  
        self.R_zupt = np.eye(3) * (zupt_unc)**2

    def predict(self, acc, gyro, dt):
        """Koppelnavigation und Fehlerpropagierung."""
        acc_world = self.q.apply(acc)
        self.p += self.v * dt + 0.5 * (acc_world + self.g) * dt**2
        self.v += (acc_world + self.g) * dt
        self.q = self.q * R.from_rotvec(gyro * dt)

        F_x = np.eye(9)
        F_x[0:3, 3:6] = np.eye(3) * dt
        
        acc_skew = np.array([
            [0, -acc_world[2], acc_world[1]],
            [acc_world[2], 0, -acc_world[0]],
            [-acc_world[1], acc_world[0], 0]
        ])
        F_x[3:6, 6:9] = -acc_skew * dt

        F_i = np.zeros((9, 6))
        F_i[3:6, 0:3] = np.eye(3) * dt
        F_i[6:9, 3:6] = np.eye(3) * dt

        self.kf.F = F_x
        self.kf.Q = F_i @ self.base_Q @ F_i.T
        self.kf.predict()

    def _inject_error_and_reset(self):
        """Injiziert den Error-State in den Nominal-State und setzt Fehler zurück."""
        self.p += self.kf.x[0:3]
        self.v += self.kf.x[3:6]
        
        error_rot = R.from_rotvec(self.kf.x[6:9])
        self.q = error_rot * self.q
        
        self.kf.x = np.zeros(9)

    def update_zupt(self):
        """Zero Velocity Update: Zwingt Geschwindigkeit auf 0."""
        H = np.zeros((3, 9))
        H[0:3, 3:6] = np.eye(3) 
        
        innovation = np.array([0.0, 0.0, 0.0]) - self.v
        self.kf.update(z=innovation, R=self.R_zupt, H=H)
        self._inject_error_and_reset()

    def update_gravity(self, acc):
        """Nutzt den Beschleunigungssensor im Stillstand als Wasserwaage gegen Pitch/Roll Drift."""
        acc_norm = np.linalg.norm(acc)
        if abs(acc_norm - 9.81) > 0.5:
            return 

        a_dir = acc / acc_norm
        g_pred = self.q.inv().apply([0, 0, 1])

        H = np.zeros((3, 9))
        H[0:3, 6:9] = np.array([
            [0, -g_pred[2], g_pred[1]],
            [g_pred[2], 0, -g_pred[0]],
            [-g_pred[1], g_pred[0], 0]
        ])

        innovation = a_dir - g_pred
        self.kf.update(z=innovation, R=self.R_grav, H=H)
        self._inject_error_and_reset()