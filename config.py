# ==========================================
# DATEI- UND PFADEINSTELLUNGEN
# ==========================================
LOG_FOLDER = "./Data/20260702_09_22_58"
IMU_CALIB_FILE = "sensor_params.json"

# ==========================================
# ABLAUF-STEUERUNG & DEBUGGING
# ==========================================
USE_AUTO_INIT = True           # Master-Schalter: True = Start-Ausrichtung berechnen, False = Ignorieren
MAX_PROCESS_TIME = None        # (s) Zum Debuggen: Bricht den Datensatz nach X Sekunden ab

# Reproduzierbare Ausbaustufe für den normalen Programmlauf.
ACTIVE_PIPELINE_VARIANT = "V5_CURRENT"

# Varianten für die spätere automatische Validierung.
VALIDATION_PIPELINE_VARIANTS = [
    "V1_IMU",
    "V2_BARO",
    "V3_RTS",
    "V4_WALL",
    "V5_CURRENT",
]

# Zielordner für reproduzierbare CSV- und Metadaten-Exporte.
VALIDATION_OUTPUT_DIR = "./Validation"

# Offizielle Zeitinformationen pro Validierungslauf.
# finish_time_s: Zeit vom Startsignal bis zum Buzzer.
VALIDATION_RUN_TIMING = {
    "7sek": {
        "reaction_time_s": 0.162,
        "finish_time_s": 7.197,
    },
}

# ==========================================
# ALIGNMENT & INITIALISIERUNG (START-PHASE)
# ==========================================
# Wähle die Methode: 'STATIC', 'MADGWICK' oder 'ESKF'
ALIGNMENT_METHOD = 'ESKF'    

# Adaptive Mehrfenster-Erkennung der besten quasi-statischen Vorstartphase.
# Es wird keine feste Ruhedauer mehr erzwungen. Stattdessen untersucht das
# Programm alle Fensterlängen zwischen Minimum und Maximum und wählt das
# längste Fenster, das die laufabhängig bestimmten Qualitätsgrenzen erfüllt.
INIT_MIN_WINDOW_SECONDS = 0.20
INIT_MAX_WINDOW_SECONDS = 2.00
INIT_WINDOW_STEP_SECONDS = 0.05
INIT_DURATION_STEP_SECONDS = 0.05
# None durchsucht die gesamte Aufzeichnung vor dem erkannten Start. Dadurch
# werden auch frühe, tatsächlich ruhige Phasen nicht mehr verworfen.
INIT_SEARCH_LOOKBACK_SECONDS = None
INIT_START_BUFFER_SECONDS = 0.20
INIT_PREFERRED_WINDOW_SECONDS = 0.60
INIT_MAX_PHASE_CANDIDATES = 12

# Adaptive Grenzen werden aus dem unteren Rausch-Perzentil jedes Laufs
# abgeleitet, bleiben aber zwischen einer Unter- und Obergrenze beschränkt.
INIT_NOISE_PERCENTILE = 10.0
INIT_ADAPTIVE_THRESHOLD_FACTOR = 2.5

INIT_ACC_NORM_TOL_MIN_G = 0.015
INIT_ACC_NORM_TOL_MAX_G = 0.08
INIT_ACC_STD_MIN_G = 0.008
INIT_ACC_STD_MAX_G = 0.03
INIT_ACC_DIR_P90_MIN_DEG = 0.50
INIT_ACC_DIR_P90_MAX_DEG = 2.00
INIT_GYRO_STD_MIN_DPS = 0.50
INIT_GYRO_STD_MAX_DPS = 4.50

# Eine kleine Standardabweichung allein genügt nicht: Auch eine gleichmäßige
# Drehung kann eine kleine Streuung besitzen. Deshalb gelten zusätzlich harte
# Grenzen für Betrag und 95-%-Quantil der korrigierten Drehrate.
INIT_GYRO_MEAN_MAX_DPS = 3.0
INIT_GYRO_P95_MAX_DPS = 4.0

# Adaptive Orientierungsfortschreibung von jeder Ruhephase bis zum Start.
# Beschleunigung wird nur weich als Gravity-Beobachtung verwendet, wenn Betrag
# und Drehrate eine quasi-statische Situation unterstützen.
INIT_WARMUP_ACC_NORM_GATE_G = 0.10
INIT_WARMUP_GYRO_GATE_DPS = 15.0
INIT_WARMUP_GRAVITY_GAIN = 0.015

# Der laufbezogene Gyro-Bias wird nur bei einer deutlich strengeren,
# ausreichend langen echten Ruhephase aktualisiert. Andernfalls bleibt die
# unabhängige Sensorkalibrierung aus sensor_params.json erhalten.
INIT_GYRO_BIAS_MIN_WINDOW_SECONDS = 0.50
INIT_GYRO_BIAS_MAX_MEAN_DPS = 1.00
INIT_GYRO_BIAS_MAX_STD_DPS = 1.50

# Beschränkte Startorientierungsoptimierung im Wandkoordinatensystem.
# Es werden ausschließlich Roll und Pitch variiert; die Yaw-Ausrichtung bleibt
# vollständig unter Kontrolle von WALL_FRAME_BASE_YAW_DEG und
# START_POSE_YAW_CORRECTION_DEG.
USE_INITIAL_ATTITUDE_OPTIMIZATION = True
USE_INITIAL_ATTITUDE_FINE_TUNING = False
EXPORT_INITIALIZATION_CANDIDATE_RUNS = True

# Historisch äquivalenter Warm-up-Kandidat zur Ablationsanalyse und als
# konfigurierbare praktische Startlagenquelle.
INCLUDE_LEGACY_WARMUP_REFERENCE = True
INITIAL_ATTITUDE_SOURCE = "STABILIZED_RECENT_WARMUP"
LEGACY_WARMUP_WINDOW_SECONDS = 4.0
LEGACY_WARMUP_BUFFER_SECONDS = 0.2
LEGACY_WARMUP_LEVELING_SECONDS = 0.5

# Beim erkannten Bewegungsstart wird ein lokales Trajektorienkoordinatensystem
# definiert. Die Lage- und Bias-Kovarianzen aus dem Warm-up bleiben erhalten;
# nur Position und Geschwindigkeit erhalten diese neuen Startunsicherheiten.
START_POSITION_STD_M = 0.001
START_VELOCITY_STD_MPS = 0.05

INITIAL_ROLL_SEARCH_MIN_DEG = -5.0
INITIAL_ROLL_SEARCH_MAX_DEG = 5.0
INITIAL_PITCH_SEARCH_MIN_DEG = -5.0
INITIAL_PITCH_SEARCH_MAX_DEG = 5.0
INITIAL_ATTITUDE_COARSE_STEP_DEG = 2.5
INITIAL_ATTITUDE_FINE_STEP_DEG = 0.5
INITIAL_ATTITUDE_FINE_RADIUS_DEG = 0.5
INITIAL_ATTITUDE_PRIOR_STD_DEG = 2.0
INITIAL_ATTITUDE_MIN_RELATIVE_IMPROVEMENT = 0.05
INITIAL_ATTITUDE_REJECT_BOUNDARY_SOLUTION = True

# Unsicherheiten und Gewichte der video-unabhängigen Kostenfunktion. Die
# laterale Form ist bewusst weich gewichtet, damit reale Seitwärtsbewegungen
# des Athleten nicht vollständig entfernt werden.
INITIAL_OPT_TARGET_X_STD_M = 0.50
INITIAL_OPT_TARGET_Y_STD_M = 0.30
INITIAL_OPT_LATERAL_SHAPE_STD_M = 0.75
INITIAL_OPT_BARO_STD_M = 0.30
INITIAL_OPT_ACCEL_BIAS_STD_MPS2 = 1.00
INITIAL_OPT_GYRO_BIAS_STD_RADPS = 0.20

INITIAL_OPT_BARO_WEIGHT = 1.0
INITIAL_OPT_WALL_WEIGHT = 0.5
INITIAL_OPT_CORRIDOR_WEIGHT = 0.5
INITIAL_OPT_ENDPOINT_WEIGHT = 0.1
INITIAL_OPT_LATERAL_SHAPE_WEIGHT = 0.5
INITIAL_OPT_ATTITUDE_PRIOR_WEIGHT = 1.0
INITIAL_OPT_BIAS_WEIGHT = 0.25

# Spezifischer Parameter zur Erkennung des eigentlichen Bewegungsstarts.
START_PEAK_THRESHOLD_G = 2.0  # (g) Ab dieser Beschleunigung gilt der Athlet als gestartet

# ==============================================================
# END-DETECTION (Ziel-Erkennung)
# ==============================================================
USE_END_DETECTION = True             
FREEFALL_THRESHOLD_G = 0.2           # Unter 0.xg gewertet als Losgelassen / Freier Fall

# Positions-Ziele am Ende des Laufs
SMOOTH_TO_BARO_Z = False        # Z-Achse an die exakt letzte Barometer-Höhe angleichen
TARGET_X_M = 1.1                # (m) +X zeigt von der Wand weg
TARGET_Y_M = -0.4               # (m) +Y zeigt nach rechts

TARGET_XY_UNCERTAINTY = 0.1     # Mögliche Abweichung

# Geschwindigkeits-Ziele
# VORSICHT: Nur auf True setzen, wenn der Run komplett bis zum Stillstand läuft!
FORCE_V_END_ZERO = False

# ==========================================
# DOMAIN KNOWLEDGE: VIRTUELLEN KORRIDOR & WAND VORGEBEN
# ==========================================
USE_YAW_CORRECTION = False   # True = 2-Pass PCA Korrektur, False = Original

# Festes rechtshändiges Wandkoordinatensystem:
# +X = von der Wand weg, -X = in die Wand
# +Y = nach rechts,       -Y = nach links
# +Z = nach oben
WALL_FRAME_BASE_YAW_DEG = 180.0     # (Grad) Basis-Drehung des Wandkoordinatensystems

# Korrektur der reproduzierbaren Hüftstellung beim Start.
# Positiv = von oben gegen den Uhrzeigersinn, negativ = im Uhrzeigersinn.
START_POSE_YAW_CORRECTION_DEG = -10.0

WALL_INCLINATION_DEG = 5.0    # (Grad) Überhang der genormten Speed-Wand

# Normalenrichtung der Wand im oben definierten Wandkoordinatensystem.
# zeigt von der Wand weg und ist deshalb exakt +X.
WALL_NORMAL_XY = [1.0, 0.0]

WALL_UNCERTAINTY = 0.8             # (m) Toleranz/Gummiband-Effekt 

# Seitliche Grenzen als absolute Koordinaten der Speedwand.
# +Y zeigt nach rechts, -Y nach links. Intern werden die Grenzen automatisch
# relativ zur Startposition des Sensors umgerechnet.
CORRIDOR_Y_MIN_M = -1.5
CORRIDOR_Y_MAX_M = 1.5
CORRIDOR_UNCERTAINTY = 0.2
CORRIDOR_UPDATE_HZ = 20.0

# ==========================================
# KALMAN FILTER TUNING 
# ==========================================

# Barometer Tuning
BARO_UNCERTAINTY = 0.3         # (m) Messrauschen des Barometers
USE_BARO_PRE_FILTER = True     # Zero-Phase Filter für Barometer
BARO_CUTOFF_HZ = 1.5           

# Kontinuierliche Prozessrauschdichten (Q)
# Aus der stationären Messung: ca. 0.0022 bis 0.0030 (m/s²)/sqrt(Hz)
ACCEL_NOISE_DENSITY = 0.003

# Messrauschen (R)
GRAVITY_UNCERTAINTY = 0.5      # (m/s^2) 

# Angle Random Walk aus der Allan-Deviation bei tau = 1 s
GYRO_NOISE_DENSITY = [4.752365e-05, 5.086460e-05, 3.940751e-05]

# Kontinuierliche Random-Walk-Anregung der Biaszustände
ACCEL_BIAS_RW_DENSITY = 1e-4
GYRO_BIAS_RW_DENSITY = [3.937686e-06, 6.420872e-06, 9.472721e-06]

# ==========================================
# ROBUSTE STILLSTANDSERKENNUNG (ZUPT)
# ==========================================
ZARU_GYRO_THRESHOLD_DPS = 4.0  # Separater Grenzwert für In-Run-ZUPT/ZARU
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
# Video einlesen, zeitlich synchronisieren und für Vergleichsplots bereitstellen.
USE_VIDEO_DATA = False

# True: Video-Y/Z als Positionsmessung in den ESKF einspeisen.
# False: Video bleibt eine unabhängige Referenz und wird nur geplottet.
USE_VIDEO_IN_FILTER = False

VIDEO_DATA_FILE = "./Data/data_Right.json"

# Darstellung im Vergleichsplot:
# "START_ALIGNED" = Videostart nur im Plot auf den IMU-Start verschieben.
# "ABSOLUTE"      = beide Trajektorien in ihren absoluten Wandkoordinaten.
VIDEO_COMPARISON_MODE = "START_ALIGNED"

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
SHOW_VELOCITY = True   
SHOW_ALTITUDE = False        
SHOW_INIT_PLOT = False  
SHOW_2D_FRONT = True
SHOW_2D_FRONT_VIDEO = True              
SHOW_2D_SIDE = True
SHOW_3D_TRAJECTORY = True         
SHOW_ANIMATED_TRAJECTORY = False     
SHOW_HIP_ROTATION = False      
SHOW_2D_FRONT_YAW = False

# Absolute Sensorstartposition im Wandkoordinatensystem [X, Y, Z].
# Diese Position ist unabhängig davon, ob Videodaten geladen werden.
SENSOR_START_POSITION_WALL_M = [0.5, 0.8, 1.1]

# Wand-Eigenschaften für die Visualisierung
VIS_WALL_LENGTH = 15.0            # (m) Länge der Kletterwand
VIS_WALL_THICKNESS = 0.10         # (m) Dicke der gezeichneten Wand im Plot
VIS_COLORMAP = "turbo"            # Farbschema für die Geschwindigkeits-Trajektorie
VIS_WALL_BG_IMAGE = "./Images/speedwall_2D.png" # Dateipfad zum Hintergrundbild
