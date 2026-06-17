import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import config
from st_log_reader import STLogReader
from preprocessing import IMUPreprocessor
from Kalman import IMUCalibration, run_eskf_pipeline

def main():

    # ---------------------------------------------------------
    #  VARIABLEN 
    # ---------------------------------------------------------
    TEST_NAME = "Return-to-Origin Test"
    
    base_folder = r"./Data/1m_LR" 
    # ---------------------------------------------------------

    # Automatisch alle Unterordner finden
    run_folders = [
        os.path.join(base_folder, folder_name) 
        for folder_name in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, folder_name))
    ]
    run_folders.sort()

    results = []
    print(f"Starte Batch-Verarbeitung: '{TEST_NAME}' mit {len(run_folders)} Läufen...\n")

    # ==============================================================
    # SCHLEIFE ÜBER ALLE LÄUFE
    # ==============================================================
    for idx, folder in enumerate(run_folders):
        print(f"--- Verarbeite Lauf {idx+1}/{len(run_folders)} ---")
        
        try:
            config.LOG_FOLDER = folder
            
            reader = STLogReader(config.LOG_FOLDER)
            calib = IMUCalibration(config.IMU_CALIB_FILE)
            preprocessor = IMUPreprocessor(config)
            
            df_imu, fs_dynamisch = preprocessor.load_and_merge_data(reader)
            df_init, init_idx = preprocessor.find_initial_stillness(df_imu, fs_dynamisch)
            q_init, P0 = preprocessor.initialize_run(df_init, calib)
            df_imu = preprocessor.process_barometer_and_crop(df_imu, P0, init_idx, fs_dynamisch)
            
            positions, velocities, orientations, times, eskf = run_eskf_pipeline(
                df_imu, q_init, init_idx, calib, fs_dynamisch
            )
            
            # --- AUSWERTUNG FÜR RETURN-TO-ORIGIN ---
            start_pos = positions[0]
            end_pos = positions[-1]
            
            # 1. Wurde der Sensor wirklich ca. 1m bewegt? (Maximaler Abstand vom Start)
            distances_from_start = np.linalg.norm(positions[:, :2] - start_pos[:2], axis=1)
            max_dist_xy = np.max(distances_from_start)
            
            # 2. Wie groß ist der Fehler am ENDE? (Sollte idealerweise 0 sein)
            end_error_3d = np.linalg.norm(end_pos - start_pos)
            end_error_xy = np.linalg.norm([end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]])
            end_error_z = end_pos[2] - start_pos[2]
            
            # Koordinaten für den Scatter-Plot speichern (relativ zum Start)
            end_x_rel = end_pos[0] - start_pos[0]
            end_y_rel = end_pos[1] - start_pos[1]
            
            results.append({
                "Lauf": f"Lauf {idx+1}",
                "Max_Dist_XY": round(max_dist_xy, 3),
                "End_Fehler_3D": round(end_error_3d, 3),
                "End_Fehler_XY": round(end_error_xy, 3),
                "End_Fehler_Z": round(end_error_z, 3),
                "End_X": end_x_rel,
                "End_Y": end_y_rel
            })
            
        except Exception as e:
            print(f"FEHLER bei Lauf {idx+1}: {e}")

    # ==============================================================
    # 3. ZUSAMMENFASSUNG & STATISTIK
    # ==============================================================
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*65)
    print(f" 📊 GESAMTAUSWERTUNG: {TEST_NAME.upper()}")
    print("="*65)
    # Nur die relevanten Spalten für die Konsole formatiert ausgeben
    print(df_results[["Lauf", "Max_Dist_XY", "End_Fehler_3D", "End_Fehler_XY", "End_Fehler_Z"]].to_string(index=False))
    print("-" * 65)
    
    valid_df = df_results.dropna(subset=["End_Fehler_3D"])
    if not valid_df.empty:
        mean_err_xy = valid_df["End_Fehler_XY"].mean()
        mean_err_z = valid_df["End_Fehler_Z"].mean()
        mean_max_dist = valid_df["Max_Dist_XY"].mean()
        
        print(f"Ø Bewegte Distanz (Check) : {mean_max_dist:.3f} m")
        print(f"Ø End-Fehler X/Y (Drift)  : {mean_err_xy:.3f} m")
        print(f"Ø End-Fehler Z (Höhe)     : {mean_err_z:+.3f} m")
    print("="*65 + "\n")

    # ==============================================================
    # 4. VISUALISIERUNG: RTO-DASHBOARD
    # ==============================================================
    if not valid_df.empty:
        print("Erstelle RTO-Auswertungs-Plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Plot 1: Balkendiagramm (Absoluter Endfehler XY) ---
        ax1.bar(valid_df["Lauf"], valid_df["End_Fehler_XY"], color='#E1812C', edgecolor='black')
        ax1.axhline(y=mean_err_xy, color='#C44E52', linestyle='--', linewidth=2, label=f'Ø XY-Drift ({mean_err_xy:.3f}m)')
        ax1.axhline(y=0.0, color='black', linewidth=1)
        
        ax1.set_title("Verbleibender XY-Positionsfehler am Ende")
        ax1.set_ylabel("Abweichung zum Startpunkt [m]")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Plot 2: 2D Scatter-Plot ("Zielscheibe") ---
        # Plottet alle Endpunkte relativ zum Startpunkt (0,0)
        ax2.scatter(valid_df["End_X"], valid_df["End_Y"], color='red', marker='x', s=100, label='Endpositionen', zorder=3)
        ax2.scatter(0, 0, color='green', marker='P', s=150, label='Startpunkt (0,0)', zorder=4)
        
        # Durchschnitts-Fehlerkreis einzeichnen
        circle = plt.Circle((0, 0), mean_err_xy, color='blue', fill=False, linestyle='--', linewidth=2, label=f'Ø Fehler-Radius ({mean_err_xy:.2f}m)')
        ax2.add_patch(circle)
        
        # Achsen zentrieren und gleichmäßig skalieren
        max_val = max(abs(valid_df["End_X"]).max(), abs(valid_df["End_Y"]).max()) * 1.5
        max_val = max(max_val, 0.2) # Mindestens 20cm Skalierung
        ax2.set_xlim(-max_val, max_val)
        ax2.set_ylim(-max_val, max_val)
        ax2.set_aspect('equal', adjustable='box')
        
        ax2.axhline(0, color='black', linewidth=0.5, zorder=1)
        ax2.axvline(0, color='black', linewidth=0.5, zorder=1)
        ax2.set_title("Zielscheiben-Ansicht (Draufsicht XY)")
        ax2.set_xlabel("X-Drift [m]")
        ax2.set_ylabel("Y-Drift [m]")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig("Batch_Auswertung_RTO_Test.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()