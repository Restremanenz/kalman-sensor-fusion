# ==========================================
# DATEI- UND PFADEINSTELLUNGEN
# ==========================================
LOG_FOLDER = "./Data/7sek"
IMU_CALIB_FILE = "sensor_params.json"

# ==========================================
# ABLAUF-STEUERUNG & INITIALISIERUNG
# ==========================================
USE_AUTO_INIT = True           # Start Ruhephase suchen   Hauptschalter
STILLNESS_THRESHOLD = 2.0      # Drehgeschwindigkeit
MIN_STILL_SECONDS = 0.1        # Ruhesekunden

MAX_PROCESS_TIME = None

# ==========================================
# ALIGNMENT & INITIALISIERUNG (START-PHASE)
# ==========================================
# Toggle: True = Retrograder Madgwick (Dynamisch), False = Static Leveling (Wasserwaage)
USE_DYNAMIC_ALIGNMENT = True    

# Parameter für das dynamische Alignment (wenn True)
START_PEAK_THRESHOLD_G = 2    # (g) Ab dieser Beschleunigung gilt der Athlet als gestartet
WARMUP_WINDOW_SEC = 4.0         # (s) Dauer der Madgwick-Einschwingphase vor dem Start
WARMUP_BUFFER_SEC = 0.4         # (s) Sicherheitsabstand vom Start-Peak zurück (Vermeidet Fliehkräfte am Ende)

# ==============================================================
# END-DETECTION (Ziel-Erkennung)
# ==============================================================
USE_END_DETECTION = True             
FREEFALL_THRESHOLD_G = 0.3           # Unter 0.6g werten wir als "Losgelassen / Freier Fall"

# ==========================================
# POST-PROCESSING (RTS Smoother)
# ==========================================
USE_SMOOTHER = True            # Smoother ein-/ausschalten

# Positions-Ziele am Ende des Laufs
SMOOTH_TO_BARO_Z = False        # Z-Achse an die exakt letzte Barometer-Höhe angleichen
SMOOTH_XY_TO_ZERO = False       # X/Y-Achsen exakt über den Startpunkt zwingen
TARGET_X_M = 0.0               # (m) Ziel X (0.0 = exakt über Start)
TARGET_Y_M = 0.0               # (m) Ziel Y (0.0 = exakt über Start)

# Geschwindigkeits-Ziele
# VORSICHT: Nur auf True setzen, wenn der Run komplett bis zum Stillstand/Buzzer läuft!
# Bei gekürzten Läufen (MAX_PROCESS_TIME) zwingend auf False lassen!
FORCE_V_END_ZERO = False

# ==========================================
# DOMAIN KNOWLEDGE: VIRTUELLEN KORRIDOR & WAND VORGEBEN
# ==========================================
USE_WALL_CONSTRAINT = False
WALL_INCLINATION_DEG = 5.0    # (Grad) Überhang der genormten Speed-Wand

# In welche Richtung (im initialen Sensor-Koordinatensystem) zeigt der Kletterer-Rücken von der Wand WEG?
# Beispiele: 
# [0.0, 1.0] -> Sensor-Y-Achse zeigt exakt vom Rücken weg.
# [1.0, 0.0] -> Sensor-X-Achse zeigt exakt vom Rücken weg.
# [1.0, 1.0] -> Sensor wurde um 45 Grad schief aufgeklebt.
WALL_NORMAL_XY = [0.0, 1.0]   

WALL_UNCERTAINTY = 0.3              # (m) Toleranz/Gummiband-Effekt (z.B. 30 cm)

WALL_DISABLE_ON_DESCENT = True      # Automatische Abschaltung beim Abseilen
DESCENT_DETECTION_THRESHOLD = 0.3   # (m) Ab welchem Höhenverlust gilt der Lauf als "beendet"?

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

# Bias Instability (Random Walk) 
# Erlaubt dem Filter, den Gyro- und Accel-Bias während des Laufs anzupassen
ACCEL_BIAS_RW = 1e-4           # Wie schnell darf sich der Accel-Bias ändern?
GYRO_BIAS_RW = 1e-5            # Wie schnell darf sich der Gyro-Bias ändern?

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
# VISUALISIERUNG
# ==========================================
ANIMATION_FPS = 30
SHOW_RAW_SENSOR_DATA = True
SHOW_VELOCITY = True   
SHOW_ALTITUDE = True        
SHOW_INIT_PLOT = True  