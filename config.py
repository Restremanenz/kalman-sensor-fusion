# ==========================================
# DATEI- UND PFADEINSTELLUNGEN
# ==========================================
LOG_FOLDER = "./Data/7sek"
IMU_CALIB_FILE = "sensor_params.json"

# ==========================================
# ABLAUF-STEUERUNG & DEBUGGING
# ==========================================
USE_AUTO_INIT = True           # Master-Schalter: True = Start-Ausrichtung berechnen, False = Ignorieren
MAX_PROCESS_TIME = None        # (s) Zum Debuggen: Bricht den Datensatz nach X Sekunden ab

# ==========================================
# ALIGNMENT & INITIALISIERUNG (START-PHASE)
# ==========================================
# Wähle die Methode: 'STATIC', 'MADGWICK' oder 'ESKF'
ALIGNMENT_METHOD = 'ESKF'    

# Basis-Parameter (Werden von allen Methoden genutzt, um den Gyro-Bias am Start zu finden)
STILLNESS_THRESHOLD = 4      # (dps) Unterhalb dieser Drehrate gilt der Sensor als ruhend
MIN_STILL_SECONDS = 0.6        # (s) So lange muss der Sensor für den initialen Gyro-Bias ruhen

# Spezifische Parameter für dynamische Starts (MADGWICK und ESKF)
START_PEAK_THRESHOLD_G = 2   # (g) Ab dieser Beschleunigung gilt der Athlet als gestartet
WARMUP_WINDOW_SEC = 4.0        # (s) Dauer der Einschwingphase VOR dem Start
WARMUP_BUFFER_SEC = 0.2        # (s) Sicherheitsabstand vom Start-Peak zurück

# ==============================================================
# END-DETECTION (Ziel-Erkennung)
# ==============================================================
USE_END_DETECTION = True             
FREEFALL_THRESHOLD_G = 0.2           # Unter 0.xg gewertet als Losgelassen / Freier Fall

# ==========================================
# POST-PROCESSING (RTS Smoother)
# ==========================================
USE_SMOOTHER = True            # Smoother ein-/ausschalten

# Positions-Ziele am Ende des Laufs
SMOOTH_TO_BARO_Z = False        # Z-Achse an die exakt letzte Barometer-Höhe angleichen
SMOOTH_XY_TO_ZERO = True       # X/Y-Achsen exakt über den Startpunkt zwingen
TARGET_X_M = -1.1               # (m) Ziel X (0.0 = exakt über Start)
TARGET_Y_M = 0.4 - 0.9              # (m) Ziel Y (0.0 = exakt über Start)

TARGET_XY_UNCERTAINTY = 0.1     # Mögliche Abweichung

# Geschwindigkeits-Ziele
# VORSICHT: Nur auf True setzen, wenn der Run komplett bis zum Stillstand läuft!
FORCE_V_END_ZERO = False

# ==========================================
# DOMAIN KNOWLEDGE: VIRTUELLEN KORRIDOR & WAND VORGEBEN
# ==========================================
USE_YAW_CORRECTION = True   # True = 2-Pass PCA Korrektur, False = Original
USE_WALL_CONSTRAINT = False
WALL_INCLINATION_DEG = 5.0    # (Grad) Überhang der genormten Speed-Wand

# In welche Richtung (im initialen Sensor-Koordinatensystem) zeigt der Kletterer-Rücken von der Wand WEG?
# Beispiele: 
# [0.0, 1.0] -> Sensor-Y-Achse zeigt exakt vom Rücken weg.
# [1.0, 0.0] -> Sensor-X-Achse zeigt exakt vom Rücken weg.
# [1.0, 1.0] -> Sensor wurde um 45 Grad schief aufgeklebt.
WALL_NORMAL_XY = [-1.0, 0.0]   

WALL_UNCERTAINTY = 0.8             # (m) Toleranz/Gummiband-Effekt (z.B. 30 cm)

# ==========================================
# KALMAN FILTER TUNING 
# ==========================================

# Barometer Tuning
BARO_UNCERTAINTY = 0.3         # (m) Messrauschen des Barometers
USE_BARO_PRE_FILTER = True     # Zero-Phase Filter für Barometer
BARO_CUTOFF_HZ = 1.5           

# Prozessrauschen (Q): IMU-Integration
ACCEL_NOISE_DENSITY = 0.05      # (m/s^2) Je höher, desto mehr vertraut der Filter auf externe Updates

# Messrauschen (R)
GRAVITY_UNCERTAINTY = 0.5      # (m/s^2) 

# Allan Variance Parameter
# Weißes Rauschen (Angle Random Walk) in rad/s
GYRO_NOISE_STD = [4.752365e-05, 5.086460e-05, 3.940751e-05]

# Bias Instability (Random Walk) 
# Erlaubt dem Filter, den Gyro- und Accel-Bias während des Laufs anzupassen
ACCEL_BIAS_RW = 1e-4           # Wie schnell darf sich der Accel-Bias ändern?
GYRO_BIAS_RW = [3.937686e-06, 6.420872e-06, 9.472721e-06]

# ==========================================
# ROBUSTE STILLSTANDSERKENNUNG (ZUPT)
# ==========================================
USE_ZUPT = True                # ZUPT im Filter ein-/ausschalten
OFFLINE_ZUPT_THRESHOLD = 0.05   # (g) Schwellenwert für gefilterte Beschleunigung
OFFLINE_ZUPT_HP_CUTOFF = 0.01  # (Hz) Hochpass-Filter (entfernt 1g Schwerkraft)
OFFLINE_ZUPT_LP_CUTOFF = 5.0   # (Hz) Tiefpass-Filter (glättet Sensorrauschen)

ZUPT_UNCERTAINTY = 0.05        # (m/s) 
ZARU_UNCERTAINTY = 0.02
# ==========================================
# MAGNETOMETER & 18-STATE ESKF
# ==========================================
USE_MAGNETOMETER = False      # HAUPTSCHALTER
USE_18_STATE_ESKF = False     # TOGGLE: True = In-Run Mag-Updates (18-State) | False = Nur Start-Heading (15-State)
MAG_CALIB_FILE = "mag_param.json"

# Referenz-Magnetfeld laut NOAA-Kalkulator https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm normiert auf die Magnitude 1.0 im [Nord, Ost, Up] System
GLOBAL_MAG_REF = [0.4649, -0.0195, -0.8852] 

MAG_UNCERTAINTY = 0.05         # (Gauss) 
MAG_BIAS_RW = 1e-3             # Random Walk für Mag-Bias 

# ==========================================
# VIDEO ANALYSE INTEGRATION
# ==========================================
USE_VIDEO_DATA = False
VIDEO_DATA_FILE = "./Data/data_Right.json"

VIDEO_UNCERTAINTY = 0.3        # (m) Messrauschen des Video-Systems
SYNC_VELOCITY_THRESHOLD = 0.5  # (m/s) Schwellenwert für den kinematischen Start

# Genormte Speedwand-Konstanten
STARTGRIFF_Y_M = 0.75          # (m) Der 1. Griff ist 75cm rechts vom Wand-Nullpunkt
STARTGRIFF_Z_M = 1.6875          # (m) Montagehöhe des Griffs (Boden bis Griff)

# ==========================================
# VISUALISIERUNG
# ==========================================
ANIMATION_FPS = 30
SHOW_RAW_SENSOR_DATA = False
SHOW_VELOCITY = False   
SHOW_ALTITUDE = False        
SHOW_INIT_PLOT = False  
SHOW_2D_FRONT = True
SHOW_2D_FRONT_VIDEO = True              
SHOW_2D_SIDE = True
SHOW_3D_TRAJECTORY = True         
SHOW_ANIMATED_TRAJECTORY = False     
SHOW_HIP_ROTATION = False      
SHOW_2D_FRONT_YAW = False   

# Sensor-Offsets für die exakte Ausrichtung in den Plots
VIS_SENSOR_START_Z = 1.1          # (m) Starthöhe des Sensors (Offset Z)
VIS_SENSOR_OFFSET_Y = 0.8         # (m) Verschiebung auf der Breite (Offset Y)
VIS_SENSOR_OFFSET_X = 0.5         # (m) Verschiebung zur Wand in der Seitenansicht (Offset X)
VIS_MIRROR_Y = False               # Soll die Y-Achse gespiegelt werden? --> rechts entspricht dann +y
VIS_MIRROR_X = True               # Soll die X-Achse gespiegelt werden? --> weck von der Wand entspricht dann +x

# Wand-Eigenschaften für die Visualisierung
VIS_WALL_LENGTH = 15.0            # (m) Länge der Kletterwand
VIS_WALL_THICKNESS = 0.10         # (m) Dicke der gezeichneten Wand im Plot
VIS_COLORMAP = "turbo"            # Farbschema für die Geschwindigkeits-Trajektorie
VIS_WALL_BG_IMAGE = "./Images/speedwall_2D.png" # Dateipfad zum Hintergrundbild