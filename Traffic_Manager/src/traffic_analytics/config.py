from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    dataset_root: Path
    dataset: str
    output_root: Path
    rolling_window: int = 12
    forecast_horizon: int = 3
    anomaly_zscore: float = 2.5
    congestion_threshold: float = 0.55
    peak_start_hour: int = 7
    peak_end_hour: int = 10
    evening_peak_start_hour: int = 16
    evening_peak_end_hour: int = 19

    @property
    def dataset_path(self) -> Path:
        return self.dataset_root / self.dataset

    @property
    def readings_path(self) -> Path:
        return self.dataset_path / "sensor-readings.csv"

    @property
    def distances_path(self) -> Path:
        return self.dataset_path / "distances.csv"

    @property
    def sensors_path(self) -> Path:
        return self.dataset_path / "sensors.txt"
