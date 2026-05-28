# ==========================================
# DATEI- UND PFADEINSTELLUNGEN
# ==========================================
LOG_FOLDER = "./Data/lena4"
ACCEL_CALIB_FILE = "acc_param.json"
GYRO_CALIB_FILE = "gyro_bias.json"

# ==========================================
# ABLAUF-STEUERUNG & INITIALISIERUNG
# ==========================================
USE_AUTO_INIT = True           # Start Ruhephase suchen
STILLNESS_THRESHOLD = 2.0      # Drehgeschwindigkeit
MIN_STILL_SECONDS = 0.1        # Ruhesekunden

# ==========================================
# KALMAN FILTER TUNING 
# ==========================================
# Prozessrauschen (Q): IMU-Integration
ACCEL_NOISE_DENSITY = 0.01      # (m/s^2) Je höher, desto mehr vertraut der Filter auf externe Updates

# Messrauschen (R)
GRAVITY_UNCERTAINTY = 0.5      # (m/s^2) 
ZUPT_UNCERTAINTY = 0.05        # (m/s) 

# Bias Instability (Random Walk) 
# Erlaubt dem Filter, den Gyro- und Accel-Bias während des Laufs anzupassen
ACCEL_BIAS_RW = 1e-4           # Wie schnell darf sich der Accel-Bias ändern?
GYRO_BIAS_RW = 1e-5            # Wie schnell darf sich der Gyro-Bias ändern?

# Heuristik-Schwellenwerte
ZUPT_THRESHOLD_MS2 = 0.09       # Stehend

# ==========================================
# VISUALISIERUNG
# ==========================================
ANIMATION_FPS = 30
SHOW_RAW_SENSOR_DATA = True
SHOW_VELOCITY = True           