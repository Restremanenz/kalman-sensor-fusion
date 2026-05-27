import os
import json
import pandas as pd
from stdatalog_core.HSD.HSDatalog import HSDatalog

class STLogReader:
    """
    Eine Helper-Klasse zum Auslesen von STMicroelectronics SensorTile.box PRO Logs.
    Liest Sensordaten (.dat) als Pandas DataFrames und extrahiert Metadaten (ODR, FS) 
    aus der device_config.json.
    """
    
    def __init__(self, acquisition_folder: str):
        self.folder = os.path.abspath(acquisition_folder)
        
        if not os.path.exists(self.folder):
            raise FileNotFoundError(f"Der Ordner '{self.folder}' existiert nicht!")
            
        self.hsd_factory = HSDatalog()
        
        # Initialisiere die ST Bibliothek
        self.hsd = self.hsd_factory.create_hsd(acquisition_folder=self.folder)
        if self.hsd is None:
            raise ValueError(f"ST-Datalog Fehler: Konnte HSDatalog-Instanz nicht erstellen. "
                             f"Prüfen Sie, ob .dat und .json Dateien in {self.folder} liegen.")

        # Lade Konfigurations-Dateien manuell für einfachen Zugriff auf Metadaten
        self.device_config = self._load_json("device_config.json")
        self.acquisition_info = self._load_json("acquisition_info.json")

    def _load_json(self, filename: str) -> dict:
        """Lädt eine JSON-Datei aus dem Log-Ordner."""
        filepath = os.path.join(self.folder, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def get_available_sensors(self) -> list:
        """Gibt eine Liste aller aktivierten Sensoren im Log zurück."""
        sensors = []
        if self.device_config and "devices" in self.device_config:
            for device in self.device_config["devices"]:
                for component in device.get("components", []):
                    for comp_name, comp_data in component.items():
                        # Wenn der Sensor enabled ist
                        if isinstance(comp_data, dict) and comp_data.get("enable", False):
                            sensors.append(comp_name)
        return sensors

    def get_sensor_info(self, sensor_name: str) -> dict:
        """
        Extrahiert die wichtigsten Metadaten (Abtastrate, Messbereich, Sensitivität) 
        für einen bestimmten Sensor aus der device_config.json.
        """
        if self.device_config and "devices" in self.device_config:
            for device in self.device_config["devices"]:
                for component in device.get("components", []):
                    if sensor_name in component:
                        info = component[sensor_name]
                        
                        # measodr (Measured ODR) ist die exakte physikalische Rate (z.B. 917.63 Hz). 
                        # Falls nicht vorhanden, Fallback auf ODR oder samples_per_ts
                        actual_odr = info.get("measodr", info.get("samples_per_ts", None))
                        
                        return {
                            "sensor_name": sensor_name,
                            "measured_odr_hz": actual_odr,
                            "full_scale": info.get("fs", None),
                            "sensitivity": info.get("sensitivity", None),
                            "data_type": info.get("data_type", None)
                        }
        return None

    def get_sensor_data(self, sensor_name: str) -> pd.DataFrame:
        """
        Liest die .dat Datei des Sensors ein und gibt sie als Pandas DataFrame zurück.
        """
        sensor = self.hsd_factory.get_sensor(self.hsd, sensor_name)
        if sensor is None:
            raise ValueError(f"Sensor '{sensor_name}' wurde im Datalog nicht gefunden!")

        # Daten auslesen 
        df_list = self.hsd_factory.get_dataframe(self.hsd, sensor)
        
        # Manchmal gibt ST eine Liste von Dataframes zurück (bei aufgeteilten Logs)
        if isinstance(df_list, list) and len(df_list) > 0:
            df = df_list[0]
        else:
            df = df_list
            
        return df