# ==========================================
# DATEI- UND PFADEINSTELLUNGEN
# ==========================================
LOG_FOLDER = "./Data/1m_LR/20260608_00_17_00"
ACCEL_CALIB_FILE = "acc_param.json"
GYRO_CALIB_FILE = "gyro_bias.json"

# ==========================================
# ABLAUF-STEUERUNG & INITIALISIERUNG
# ==========================================
USE_AUTO_INIT = True           # Start Ruhephase suchen
STILLNESS_THRESHOLD = 3.0      # Drehgeschwindigkeit
MIN_STILL_SECONDS = 0.1        # Ruhesekunden

MAX_PROCESS_TIME = None

# ==========================================
# POST-PROCESSING (Smoother)
# ==========================================
USE_SMOOTHER = True            # Smoother ein-/ausschalten

# Positions-Ziele am Ende des Laufs
SMOOTH_TO_BARO_Z = True        # Z-Achse an die exakt letzte Barometer-Höhe angleichen
SMOOTH_XY_TO_ZERO = False       # X/Y-Achsen exakt über den Startpunkt zwingen
TARGET_X_M = 0.0               # (m) Ziel X (0.0 = exakt über Start)
TARGET_Y_M = 0.0               # (m) Ziel Y (0.0 = exakt über Start)

# Geschwindigkeits-Ziele
# VORSICHT: Nur auf True setzen, wenn der Run komplett bis zum Stillstand/Buzzer läuft!
# Bei gekürzten Läufen (MAX_PROCESS_TIME) zwingend auf False lassen!
FORCE_V_END_ZERO = True

# ==========================================
# KALMAN FILTER TUNING 
# ==========================================
USE_ZUPT = True

# Barometer Tuning
BARO_UNCERTAINTY = 0.3         # (m) Messrauschen des Barometers
USE_BARO_PRE_FILTER = True     # Zero-Phase Filter für Barometer
BARO_CUTOFF_HZ = 1.5           

# Prozessrauschen (Q): IMU-Integration
ACCEL_NOISE_DENSITY = 0.05      # (m/s^2) Je höher, desto mehr vertraut der Filter auf externe Updates

# Messrauschen (R)
GRAVITY_UNCERTAINTY = 0.5      # (m/s^2) 
ZUPT_UNCERTAINTY = 0.05        # (m/s) 
ZARU_UNCERTAINTY = 0.01

# Bias Instability (Random Walk) 
# Erlaubt dem Filter, den Gyro- und Accel-Bias während des Laufs anzupassen
ACCEL_BIAS_RW = 1e-4           # Wie schnell darf sich der Accel-Bias ändern?
GYRO_BIAS_RW = 1e-5            # Wie schnell darf sich der Gyro-Bias ändern?

# Heuristik-Schwellenwerte
ZUPT_THRESHOLD_MS2 = 0.2       # Stehend

# ==========================================
# VISUALISIERUNG
# ==========================================
ANIMATION_FPS = 30
SHOW_RAW_SENSOR_DATA = False
SHOW_VELOCITY = False   
SHOW_ALTITUDE = False        
SHOW_INIT_PLOT = False  