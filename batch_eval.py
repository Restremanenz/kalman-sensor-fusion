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
    TEST_NAME = "1-Meter Test"
    
    base_folder = r"./Data/Data_1m" 
    # ---------------------------------------------------------

    # Automatisch alle Unterordner finden
    run_folders = [
        os.path.join(base_folder, folder_name) 
        for folder_name in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, folder_name))
    ]
    run_folders.sort()

    results = []
    trajectories = [] # Speichert die Pfade für die Visualisierung
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
            
            # --- AUSWERTUNG FÜR 1-METER BEWEGUNG ---
            start_pos = positions[0]
            end_pos = positions[-1]
            
            # Relative Endposition zum Start
            end_x_rel = end_pos[0] - start_pos[0]
            end_y_rel = end_pos[1] - start_pos[1]
            end_z_rel = end_pos[2] - start_pos[2]
            
            # 1. Finale Distanz in der XY-Ebene (Sollte 1.0m sein)
            final_dist_xy = np.linalg.norm([end_x_rel, end_y_rel])
            
            # 2. Fehler zur 1m-Marke
            dist_error = final_dist_xy - 1.0
            
            # Pfad speichern für die Plots (relativ zum Start)
            traj_x = positions[:, 0] - start_pos[0]
            traj_y = positions[:, 1] - start_pos[1]
            traj_z = positions[:, 2] - start_pos[2]
            t_rel = times - times[0]
            
            trajectories.append({
                "label": f"Lauf {idx+1}",
                "x": traj_x,
                "y": traj_y,
                "z": traj_z,
                "t": t_rel
            })
            
            results.append({
                "Lauf": f"Lauf {idx+1}",
                "End_Distanz_XY": round(final_dist_xy, 3),
                "Fehler_zu_1m": round(dist_error, 3),
                "End_Fehler_Z": round(end_z_rel, 3),
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
    print(df_results[["Lauf", "End_Distanz_XY", "Fehler_zu_1m", "End_Fehler_Z"]].to_string(index=False))
    print("-" * 65)
    
    valid_df = df_results.dropna(subset=["End_Distanz_XY"])
    if not valid_df.empty:
        mean_dist = valid_df["End_Distanz_XY"].mean()
        mean_err_1m = valid_df["Fehler_zu_1m"].mean()
        mean_err_z = valid_df["End_Fehler_Z"].mean()
        
        print(f"Ø Gemessene End-Distanz : {mean_dist:.3f} m")
        print(f"Ø Abweichung von 1m     : {mean_err_1m:+.3f} m")
        print(f"Ø End-Fehler Z (Höhe)   : {mean_err_z:+.3f} m")
    print("="*65 + "\n")

# ==============================================================
    # 4. VISUALISIERUNG: 1-METER-DASHBOARD
    # ==============================================================
    if not valid_df.empty and trajectories:
        print("Erstelle Auswertungs-Plots für die 1m-Bewegung...")
        
        # -------------------------------------------------------------
        # FENSTER 1: Detailliertes Dashboard (Trajektorien & Fehler)
        # -------------------------------------------------------------
        fig1 = plt.figure(figsize=(16, 10))

        # Plot 1: 2D Trajektorien
        ax1 = fig1.add_subplot(2, 2, 1)
        for traj in trajectories:
            ax1.plot(traj["x"], traj["y"], alpha=0.7, label=traj["label"])
            ax1.scatter(traj["x"][-1], traj["y"][-1], marker='x', s=60)
        ax1.scatter(0, 0, color='green', marker='P', s=150, label='Start (0,0)', zorder=5)
        circle = plt.Circle((0, 0), 1.0, color='blue', fill=False, linestyle='--', linewidth=2, label='1m Ideal-Radius')
        ax1.add_patch(circle)
        ax1.set_title("2D Pfadverlauf (Trajektorien)")
        ax1.set_xlabel("X-Position [m]")
        ax1.set_ylabel("Y-Position [m]")
        ax1.axis('equal')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='lower left', fontsize='small')

        # Plot 2: Fehlerbalken
        ax2 = fig1.add_subplot(2, 2, 2)
        ax2.bar(valid_df["Lauf"], valid_df["Fehler_zu_1m"], color='#4C72B0', edgecolor='black')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.axhline(mean_err_1m, color='#C44E52', linestyle='--', linewidth=2, label=f'Ø Fehler ({mean_err_1m:+.3f}m)')
        ax2.set_title("Fehler bei der Distanzmessung")
        ax2.set_ylabel("Abweichung zur 1m-Marke [m]")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.legend()

        # Plot 3: Distanz über die Zeit
        ax3 = fig1.add_subplot(2, 2, 3)
        for traj in trajectories:
            dist_over_time = np.sqrt(traj["x"]**2 + traj["y"]**2)
            ax3.plot(traj["t"], dist_over_time, alpha=0.7)
        ax3.axhline(1.0, color='blue', linestyle='--', linewidth=2, label='1m Ziel')
        ax3.set_title("Bewegungsprofil (XY-Distanz über Zeit)")
        ax3.set_xlabel("Zeit [s]")
        ax3.set_ylabel("Distanz vom Start [m]")
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()

        # Plot 4: Höhen-Drift
        ax4 = fig1.add_subplot(2, 2, 4)
        for traj in trajectories:
            ax4.plot(traj["t"], traj["z"], alpha=0.7)
        ax4.axhline(0, color='black', linewidth=1)
        ax4.set_title("Drift in der Höhe (Z-Achse) über Zeit")
        ax4.set_xlabel("Zeit [s]")
        ax4.set_ylabel("Z-Position [m]")
        ax4.grid(True, linestyle='--', alpha=0.6)

        fig1.tight_layout()
        fig1.savefig("Batch_Auswertung_1m_Dashboard.png", dpi=300, bbox_inches='tight')

        # -------------------------------------------------------------
        # FENSTER 2: Genauigkeit & Präzision (Bar-Chart & Boxplot)
        # -------------------------------------------------------------
        fig2, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
        
        target_dist = 1.0
        std_dist = valid_df["End_Distanz_XY"].std()
        
        # --- Plot 2.1: Balkendiagramm ---
        ax_bar.bar(valid_df["Lauf"], valid_df["End_Distanz_XY"], color='#4C72B0', edgecolor='black')
        ax_bar.axhline(target_dist, color='#C44E52', linestyle='--', linewidth=2.5, label=f'Ziel ({target_dist:.1f}m)')
        ax_bar.axhline(mean_dist, color='#55A868', linestyle='-', linewidth=2.5, label=f'Ø Distanz ({mean_dist:.2f}m)')
        
        # Y-Achse so einstellen, dass die Schwankungen um die 1m-Marke gut sichtbar sind
        ax_bar.set_ylim(0.5, 1.5) 
        
        ax_bar.set_title("Erreichte Distanz", fontsize=14)
        ax_bar.set_ylabel("XY-Distanz [m]", fontsize=12)
        ax_bar.tick_params(axis='x', rotation=45)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        ax_bar.legend(loc='lower right', fontsize=12)
        
        # --- Plot 2.2: Boxplot ---
        box = ax_box.boxplot(valid_df["End_Distanz_XY"], widths=0.5, patch_artist=True)
        
        # Boxplot hellgrau färben, um deinen Screenshot exakt zu spiegeln
        for patch in box['boxes']:
            patch.set_facecolor('#cccccc')
            
        ax_box.axhline(target_dist, color='#C44E52', linestyle='--', linewidth=2.5, label=f'Ziel ({target_dist:.1f}m)')
        
        ax_box.set_title("Streuung der Zieldistanz", fontsize=14)
        ax_box.set_ylabel("XY-Distanz [m]", fontsize=12)
        ax_box.set_xticks([1])
        ax_box.set_xticklabels([f"Alle {len(valid_df)} Läufe\n(Std: {std_dist:.3f}m)"], fontsize=12)
        ax_box.grid(axis='y', linestyle='--', alpha=0.7)
        ax_box.legend(loc='lower right', fontsize=12)
        
        fig2.tight_layout()
        fig2.savefig("Batch_Auswertung_1m_BarBox.png", dpi=300, bbox_inches='tight')

        # Zeigt beide Fenster an
        plt.show()

if __name__ == "__main__":
    main()
