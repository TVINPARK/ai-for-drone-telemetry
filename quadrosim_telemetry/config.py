"""
Конфигурация системы ТВ-телеметрия Квадросима
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ROI:
    """Region of Interest с координатами (x, y, width, height)"""
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class HUDConfig:
    """Конфигурация зон HUD"""
    pilot_info: Optional[ROI] = None  # Верх-лево: регион и ФИО
    datetime: Optional[ROI] = None    # Дата/время
    battery_voltage: Optional[ROI] = None  # Напряжение батареи
    battery_current: Optional[ROI] = None   # Ток батареи
    flight_mode: Optional[ROI] = None  # Режим полёта
    time_limit: Optional[ROI] = None   # Верх-право: лимит времени
    speed: Optional[ROI] = None        # Центр-лево: скорость
    altitude: Optional[ROI] = None     # Центр-право: высота
    laps_info: Optional[ROI] = None    # Низ-лево: круги
    current_time: Optional[ROI] = None # Низ-лево: текущее время
    best_time: Optional[ROI] = None    # Низ-право: лучшее время
    left_stick: Optional[ROI] = None   # Низ-центр: левый стик
    right_stick: Optional[ROI] = None  # Низ-центр: правый стик


@dataclass
class CaptureConfig:
    """Настройки захвата экрана"""
    fps: int = 60
    monitor_index: int = 0
    use_roi: bool = True
    codec: str = "libx264"


@dataclass
class OCRConfig:
    """Настройки OCR"""
    whitelist: str = "0123456789.,:VAMC/"
    min_confidence: float = 0.7
    median_filter_size: int = 5


@dataclass
class StickConfig:
    """Настройки распознавания стиков"""
    dot_color_lower: list = None  # HSV нижняя граница цвета точки
    dot_color_upper: list = None  # HSV верхняя граница цвета точки
    min_area: int = 10
    max_area: int = 500
    
    def __post_init__(self):
        if self.dot_color_lower is None:
            self.dot_color_lower = [0, 100, 100]
        if self.dot_color_upper is None:
            self.dot_color_upper = [15, 255, 255]


@dataclass
class LoggerConfig:
    """Настройки логгирования"""
    db_path: str = "data/telemetry.db"
    batch_size: int = 100
    use_parquet: bool = False
    parquet_path: str = "data/telemetry.parquet"


@dataclass
class AnalysisConfig:
    """Настройки анализа"""
    sector_count: int = 3  # Количество виртуальных секторов
    delta_resample_points: int = 100  # Точек для ресемплинга дельта-кривой
    min_lap_time: float = 5.0  # Минимальное время круга (сек)
    crash_velocity_threshold: float = 0.5  # Порог обнаружения краша


@dataclass
class SystemConfig:
    """Основная конфигурация системы"""
    hud: HUDConfig
    capture: CaptureConfig
    ocr: OCRConfig
    sticks: StickConfig
    logger: LoggerConfig
    analysis: AnalysisConfig
    
    @classmethod
    def default(cls) -> 'SystemConfig':
        return cls(
            hud=HUDConfig(),
            capture=CaptureConfig(),
            ocr=OCRConfig(),
            sticks=StickConfig(),
            logger=LoggerConfig(),
            analysis=AnalysisConfig()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        return cls(
            hud=HUDConfig(**{k: ROI(**v) if v and isinstance(v, dict) else v 
                            for k, v in data.get('hud', {}).items()}),
            capture=CaptureConfig(**data.get('capture', {})),
            ocr=OCRConfig(**data.get('ocr', {})),
            sticks=StickConfig(**data.get('sticks', {})),
            logger=LoggerConfig(**data.get('logger', {})),
            analysis=AnalysisConfig(**data.get('analysis', {}))
        )


class ConfigManager:
    """Менеджер конфигурации"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config: Optional[SystemConfig] = None
    
    def load(self) -> SystemConfig:
        """Загрузить конфигурацию из файла"""
        if not self.config_path.exists():
            self.config = SystemConfig.default()
            self.save()
            return self.config
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.config = SystemConfig.from_dict(data)
        return self.config
    
    def save(self, config: Optional[SystemConfig] = None):
        """Сохранить конфигурацию в файл"""
        if config:
            self.config = config
        
        if self.config is None:
            raise ValueError("No config to save")
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
    
    def update_roi(self, field_name: str, roi: ROI):
        """Обновить ROI для конкретного поля HUD"""
        if self.config is None:
            self.load()
        
        setattr(self.config.hud, field_name, roi)
        self.save()
    
    def get_roi(self, field_name: str) -> Optional[ROI]:
        """Получить ROI для конкретного поля"""
        if self.config is None:
            self.load()
        
        return getattr(self.config.hud, field_name, None)
