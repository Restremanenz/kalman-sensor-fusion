import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import config
from st_log_reader import STLogReader
from preprocessing import IMUPreprocessor
from Kalman import IMUCalibration, run_eskf_pipeline

def main():
    # ==============================================================
    # 1. SETUP FÜR BATCH-VERARBEITUNG
    # ==============================================================
    
    base_folder = "./Data/Data_1m"

    # Automatisch alle Unterordner finden und als Liste speichern
    run_folders = [
        os.path.join(base_folder, folder_name) 
        for folder_name in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, folder_name))
    ]
    
    run_folders.sort()
    results = []

    print(f"Starte Batch-Verarbeitung von {len(run_folders)} Läufen...\n")

    # ==============================================================
    # 2. DIE SCHLEIFE ÜBER ALLE LÄUFE
    # ==============================================================
    for idx, folder in enumerate(run_folders):
        print(f"--- Verarbeite Lauf {idx+1}/{len(run_folders)} ---")
        
        try:
            # Überschreibe den Config-Pfad für den aktuellen Schleifendurchlauf
            config.LOG_FOLDER = folder
            
            reader = STLogReader(config.LOG_FOLDER)
            calib = IMUCalibration(config.ACCEL_CALIB_FILE, config.GYRO_CALIB_FILE)
            preprocessor = IMUPreprocessor(config)
            
            # Pipeline-Schritte aufrufen
            df_imu, fs_dynamisch = preprocessor.load_and_merge_data(reader)
            df_init, init_idx = preprocessor.find_initial_stillness(df_imu, fs_dynamisch)
            q_init, P0 = preprocessor.initialize_run(df_init, calib)
            df_imu = preprocessor.process_barometer_and_crop(df_imu, P0, init_idx, fs_dynamisch)
            
            # Kalman Filter rechnen lassen
            positions, velocities, orientations, times, eskf = run_eskf_pipeline(
                df_imu, q_init, init_idx, calib, fs_dynamisch
            )
            
            # 1-Meter Test Auswertung für diesen Lauf
            start_pos = positions[0]
            end_pos = positions[-1]
            total_distance = np.linalg.norm(end_pos - start_pos)
            error = total_distance - 1.0
            
            # Ergebnis speichern
            results.append({
                "Lauf": f"Lauf {idx+1}",
                "Distanz [m]": round(total_distance, 3),
                "Fehler [m]": round(error, 3),
                "Ba_X": round(eskf.ba[0], 4), # Optional: Bias mitloggen um zu sehen ob der Sensor driftet
            })
            
        except Exception as e:
            print(f"FEHLER bei Lauf {idx+1}: {e}")
            results.append({
                "Lauf": f"Lauf {idx+1}",
                "Distanz [m]": None,
                "Fehler [m]": None,
                "Ba_X": None
            })

# ==============================================================
    # 3. ZUSAMMENFASSUNG & STATISTIK
    # ==============================================================
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print(" 📊 GESAMTAUSWERTUNG 1-METER TEST")
    print("="*50)
    print(df_results.to_string(index=False))
    print("-" * 50)
    
    valid_df = df_results.dropna(subset=["Distanz [m]"])
    if not valid_df.empty:
        mean_dist = valid_df["Distanz [m]"].mean()
        std_dist = valid_df["Distanz [m]"].std()
        
        print(f"Durchschnittliche Distanz : {mean_dist:.3f} m")
        print(f"Standardabweichung        : {std_dist:.3f} m")
        print(f"Durchschnittlicher Fehler : {mean_dist - 1.0:+.3f} m")
    print("="*50 + "\n")

    # ==============================================================
    # 4. VISUALISIERUNG DER BATCH-ERGEBNISSE
    # ==============================================================
    if not valid_df.empty:
        print("Erstelle Auswertungs-Plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Plot 1: Balkendiagramm (Jeder Lauf einzeln) ---
        ax1.bar(valid_df["Lauf"], valid_df["Distanz [m]"], color='#4C72B0', edgecolor='black')
        ax1.axhline(y=1.0, color='#C44E52', linestyle='--', linewidth=2, label='Wahre Distanz (1.0m)')
        ax1.axhline(y=mean_dist, color='#55A868', linestyle='-', linewidth=2, label=f'Durchschnitt ({mean_dist:.2f}m)')
        
        ax1.set_title("Gemessene Distanz pro Lauf")
        ax1.set_ylabel("Berechnete Strecke [m]")
        ax1.tick_params(axis='x', rotation=45)
        
        # Y-Achse schön skalieren (z.B. von 0.5 bis 1.5 Meter für bessere Lesbarkeit)
        ax1.set_ylim([0.0, max(1.5, valid_df["Distanz [m]"].max() + 0.2)])
        ax1.legend(loc='lower right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Plot 2: Boxplot (Statistische Streuung / Präzision) ---
        box = ax2.boxplot(valid_df["Distanz [m]"], vert=True, patch_artist=True, widths=0.4)
        for patch in box['boxes']:
            patch.set_facecolor('lightgray')
            
        ax2.axhline(y=1.0, color='#C44E52', linestyle='--', linewidth=2, label='Wahre Distanz (1.0m)')
        ax2.set_title("Statistische Streuung (Filter-Präzision)")
        ax2.set_ylabel("Berechnete Strecke [m]")
        ax2.set_xticks([1])
        ax2.set_xticklabels(["Verteilung aller Läufe"])
        ax2.legend(loc='lower right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        
        # Bild automatisch im Ordner speichern für deine Bachelorarbeit!
        plt.savefig("Batch_Auswertung_1m_Test.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()