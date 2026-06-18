import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter

import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
import config

class FilterpyESKF:
    def __init__(self, initial_pos, initial_q, gyro_noise_std, accel_noise, bg_rw, ba_rw, 
                 grav_unc, zupt_unc, baro_unc, zaru_unc, use_18_state=False, mag_rw=1e-5, mag_unc=0.5):
        
        self.use_18_state = use_18_state
        self.dim_x = 18 if use_18_state else 15  # Dynamische Matrixgröße
        self.dim_noise = 15 if use_18_state else 12

        self.p = np.array(initial_pos, dtype=float)
        self.v = np.zeros(3)
        self.q = initial_q
        self.ba = np.zeros(3)  
        self.bg = np.zeros(3)  
        self.bm = np.zeros(3)  # In-Run Magnetometer Bias (Nur für 18-State)
        self.g = np.array([0, 0, -9.81]) 

        # Normierte globale Magnetfeld-Referenz
        self.mag_ref = np.array(config.GLOBAL_MAG_REF) / np.linalg.norm(config.GLOBAL_MAG_REF)

        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=3) 
        self.kf.x = np.zeros(self.dim_x) 
        
        # Initiale Unsicherheit (P-Matrix) dynamisch anpassen
        self.kf.P = np.eye(self.dim_x) * 0.01 
        self.kf.P[9:12, 9:12] *= 0.1   # Accel-Bias
        self.kf.P[12:15, 12:15] *= 0.1 # Gyro-Bias
        if self.use_18_state:
            self.kf.P[15:18, 15:18] *= 0.5 # Mag-Bias Startunsicherheit
        
        # Base Process Noise dynamisch anpassen
        self.base_Q = np.zeros((self.dim_noise, self.dim_noise))
        self.base_Q[0:3, 0:3] = np.eye(3) * (accel_noise)**2  
        self.base_Q[3:6, 3:6] = np.diag(gyro_noise_std**2)
        self.base_Q[6:9, 6:9] = np.eye(3) * (ba_rw)**2
        self.base_Q[9:12, 9:12] = np.eye(3) * (bg_rw)**2
        if self.use_18_state:
            self.base_Q[12:15, 12:15] = np.eye(3) * (mag_rw)**2

        # Messrauschen
        self.R_grav = np.eye(3) * (grav_unc)**2  
        self.R_zupt = np.eye(3) * (zupt_unc)**2
        self.R_baro = np.eye(3) * 1e8         
        self.R_baro[2, 2] = baro_unc**2        
        self.R_zaru = np.eye(3) * (zaru_unc)**2
        self.R_mag = np.eye(3) * (mag_unc)**2  # Magnetometer Update Rauschen

    def predict(self, acc_meas, gyro_meas, dt):
        # Wahre Sensorwerte
        acc_true = acc_meas - self.ba
        gyro_true = gyro_meas - self.bg

        acc_world = self.q.apply(acc_true)
        R_mat = self.q.as_matrix()

        # Nominal State Update
        self.p += self.v * dt + 0.5 * (acc_world + self.g) * dt**2
        self.v += (acc_world + self.g) * dt
        self.q = self.q * R.from_rotvec(gyro_true * dt)

        # Jacobian F_x dynamisch bauen
        F_x = np.eye(self.dim_x)
        F_x[0:3, 3:6] = np.eye(3) * dt
        
        acc_skew = np.array([
            [0, -acc_world[2], acc_world[1]],
            [acc_world[2], 0, -acc_world[0]],
            [-acc_world[1], acc_world[0], 0]
        ])
        F_x[3:6, 6:9] = -acc_skew * dt            
        F_x[3:6, 9:12] = -R_mat * dt              
        F_x[6:9, 12:15] = -R_mat * dt             

        # Jacobian F_i dynamisch bauen
        F_i = np.zeros((self.dim_x, self.dim_noise))
        F_i[3:6, 0:3] = R_mat * dt                
        F_i[6:9, 3:6] = R_mat * dt                
        F_i[9:12, 6:9] = np.eye(3) * dt           
        F_i[12:15, 9:12] = np.eye(3) * dt         
        
        if self.use_18_state:
            # Mag RW mappt 1:1 auf den Mag-Bias Error-State
            F_i[15:18, 12:15] = np.eye(3) * dt

        self.kf.F = F_x
        self.kf.Q = F_i @ self.base_Q @ F_i.T
        self.kf.predict()

    def _inject_error_and_reset(self):
        self.p += self.kf.x[0:3]
        self.v += self.kf.x[3:6]
        
        error_rot = R.from_rotvec(self.kf.x[6:9])
        self.q = error_rot * self.q
        
        self.ba += self.kf.x[9:12]
        self.bg += self.kf.x[12:15]
        
        if self.use_18_state:
            self.bm += self.kf.x[15:18]
            
        self.kf.x = np.zeros(self.dim_x)

    def update_mag(self, mag_meas):
        """Korrekturschritt durch das Magnetometer (Heading & Mag-Bias)."""
        # Wenn im 15-State-Modus aufgerufen (sollte eigentlich nicht passieren), abbrechen
        if not self.use_18_state:
            return

        mag_true = mag_meas - self.bm
        
        # Erwartetes Magnetfeld im Body-Frame: R^T * m_global
        R_mat_inv = self.q.inv().as_matrix()
        mag_pred = R_mat_inv @ self.mag_ref
        
        H = np.zeros((3, self.dim_x))
        
        # d(Z) / d(Theta_global) = R^T * [m_global]_x
        mag_glob_skew = np.array([
            [0, -self.mag_ref[2], self.mag_ref[1]],
            [self.mag_ref[2], 0, -self.mag_ref[0]],
            [-self.mag_ref[1], self.mag_ref[0], 0]
        ])
        H[0:3, 6:9] = R_mat_inv @ mag_glob_skew
        
        # d(Z) / d(b_m) = I
        H[0:3, 15:18] = np.eye(3)

        innovation = mag_true - mag_pred
        self._robust_update(z=innovation, H=H, R=self.R_mag)

    def _robust_update(self, z, H, R):
        """
        Führt ein numerisch stabiles Kalman-Update durch (Joseph-Form + Symmetrisierung).
        Verhindert, dass die Kovarianzmatrix (P) durch Rundungsfehler nach 10.000 Iterationen implodiert.
        """
        # 1. Alte Kovarianzmatrix VOR dem Update sichern
        P_prior = self.kf.P.copy()
        
    # 2. Dimensionen via NumPy flexibel absichern (erlaubt 1D und 3D Updates!)
        y = np.atleast_1d(z)  # Innovation
        H = np.atleast_2d(H)  # Messmatrix
        R = np.atleast_2d(R)  # Messrauschen
        
        # 3. Innovation-Kovarianz: S = H @ P @ H.T + R
        S = H @ P_prior @ H.T + R
        
        # 4. Kalman-Gain berechnen: K = P @ H.T @ inv(S)
        # Für 1D-Baro ist S eine 1x1 Matrix -> Die Invertierung ist absolut stabil!
        K = P_prior @ H.T @ np.linalg.inv(S)
        
        # 5. Fehler-Zustand berechnen und im FilterPy-Objekt hinterlegen
        self.kf.x = K @ y
        
        # 6. Fehler in den nominalen Zustand injizieren und x wieder auf 0 setzen
        self._inject_error_and_reset()
        
        # 7. Joseph-Form für die Kovarianzmatrix P anwenden (Symmetrie-Garantie)
        I = np.eye(self.dim_x)
        I_KH = I - (K @ H)
        P_joseph = (I_KH @ P_prior @ I_KH.T) + (K @ R @ K.T)
        
        # 8. Symmetrie erzwingen und zurückschreiben
        self.kf.P = 0.5 * (P_joseph + P_joseph.T)

    def update_zupt(self):
        """Zero Velocity Update (Geschwindigkeit = 0)."""
        H = np.zeros((3, self.dim_x))
        H[0:3, 3:6] = np.eye(3) 
        
        innovation = np.array([0.0, 0.0, 0.0]) - self.v
        self._robust_update(z=innovation, H=H, R=self.R_zupt)

    def update_zaru(self, gyro_meas):
        """
        Zero Angular Rate Update (ZARU).
        Zwingt die Drehrate im Stillstand auf 0 und eliminiert dadurch den Yaw-Drift (Gieren).
        """
        H = np.zeros((3, self.dim_x))
        # Die Beobachtung mappt direkt negativ auf den Gyroskop-Bias (Index 12 bis 15 im Error-State)
        H[0:3, 12:15] = -np.eye(3) 
        
        # Innovation: Wahre Drehrate (0.0) minus aktuell geschätzte Drehrate
        gyro_true = gyro_meas - self.bg
        innovation = np.array([0.0, 0.0, 0.0]) - gyro_true
        
        self._robust_update(z=innovation, H=H, R=self.R_zaru)

    def update_barometer(self, baro_z):
        """Update der Z-Achse durch das Barometer"""
        # H-Matrix hat jetzt genau 1 Zeile und dim_x Spalten
        H = np.zeros((1, self.dim_x))
        H[0, 2] = 1.0  # Wir mappen die Z-Position (Index 2) direkt auf die Messung
        
        # Die 1D-Innovation ist ein einfacher Skalar
        innovation = baro_z - self.p[2]
        
        # Reines Skalar-Rauschen (z.B. 0.3^2 = 0.09) -> Keine Riesenmatrizen mehr!
        R_baro_1d = self.R_baro[2, 2]  
        
        # Das Update läuft nun absolut sauber ohne numerische Verzerrungen
        self._robust_update(z=innovation, H=H, R=R_baro_1d)

    def update_gravity(self, acc_meas):
        """Nutzt Beschleunigung im Stillstand als Wasserwaage."""
        acc_true = acc_meas - self.ba
        acc_norm = np.linalg.norm(acc_true)
        
        if abs(acc_norm - 9.81) > 0.5:
            return 

        a_dir = acc_true / acc_norm
        R_mat_inv = self.q.inv().as_matrix()
        g_pred = R_mat_inv @ np.array([0, 0, 1]) # g_global ist [0,0,1]

        H = np.zeros((3, self.dim_x))
        
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
        self._robust_update(z=innovation, H=H, R=self.R_grav)

    def update_position_xy(self, target_xy, uncertainty=0.01):
        """Boundary Update: Zwingt die X/Y Position auf einen Zielwert (z.B. Startwand)."""
        H = np.zeros((2, self.dim_x))
        H[0, 0] = 1.0  # Mapping auf State X
        H[1, 1] = 1.0  # Mapping auf State Y
        
        innovation = target_xy - self.p[0:2]
        self._robust_update(z=innovation, H=H, R=np.eye(2) * (uncertainty**2))

    def update_position_z(self, target_z, uncertainty=0.01):
        """Boundary Update: Zwingt die Z Position auf einen Zielwert (z.B. Baro-Endhöhe)."""
        H = np.zeros((1, self.dim_x))
        H[0, 2] = 1.0  # Mapping auf State Z
        
        innovation = target_z - self.p[2]
        self._robust_update(z=innovation, H=H, R=np.array([[uncertainty**2]]))

    def update_wall_constraint(self, normal_xy, inclination_deg, uncertainty=0.3):
        """
        Soft Constraint: Zwingt den Kletterer an die überhängende Wand (z.B. 5°).
        normal_xy: Ein 2D-Vektor [nx, ny], der angibt, welche Richtung in der XY-Ebene 
                   vom Startpunkt aus von der Wand WEG zeigt.
        """
        # 1. Normalenvektor normieren (falls man z.B. [1, 1] übergibt)
        n = np.array(normal_xy, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            return  # Ungültiger Vektor, abbruch
        nx, ny = n / n_norm

        # 2. Tangens des Überhangs berechnen (5° = 0.0874)
        tan_theta = np.tan(np.radians(inclination_deg))

        # 3. Messmatrix H aufbauen (Gleichung: nx*X + ny*Y - Z*tan(theta) = 0)
        H = np.zeros((1, self.dim_x))
        H[0, 0] = nx          # Ableitung nach Position X
        H[0, 1] = ny          # Ableitung nach Position Y
        H[0, 2] = -tan_theta  # Ableitung nach Position Z

        # 4. Innovation berechnen
        # Wie weit ist unsere aktuelle Position von der idealen 5°-Ebene entfernt?
        current_val = nx * self.p[0] + ny * self.p[1] - self.p[2] * tan_theta
        innovation = 0.0 - current_val # Zielwert ist exakt 0

        # 5. Robustes Update ausführen
        # Das Rauschen (uncertainty) wirkt wie ein Gummiband. 
        # z.B. 0.3m bedeutet: Der Filter toleriert es, wenn der Kletterer sich leicht von der Wand wegdrückt.
        self._robust_update(z=innovation, H=H, R=np.array([[uncertainty**2]]))