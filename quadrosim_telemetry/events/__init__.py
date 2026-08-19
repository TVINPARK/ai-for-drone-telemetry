"""
Модуль детекции событий полёта
- Старт/конец круга
- Начало/завершение вылета
- Краш (резкий обрыв скорости/высоты)
- Live-дельта к лучшему кругу
"""
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class EventType(Enum):
    """Типы событий"""
    FLIGHT_START = "flight_start"
    FLIGHT_END = "flight_end"
    LAP_START = "lap_start"
    LAP_END = "lap_end"
    CRASH = "crash"
    BEST_LAP = "best_lap"
    SECTOR_BEST = "sector_best"


@dataclass
class Event:
    """Событие полёта"""
    event_type: EventType
    timestamp: float  # performance_counter
    flight_time: float  # время от начала вылета
    lap_number: Optional[int] = None
    sector_number: Optional[int] = None
    value: Optional[float] = None  # время круга/сектора, скорость и т.д.
    data: Optional[Dict[str, Any]] = None  # дополнительные данные


class LapDetector:
    """Детектор кругов по таймеру текущего времени"""
    
    def __init__(self, min_lap_time: float = 5.0):
        """
        Args:
            min_lap_time: Минимальное допустимое время круга (сек)
        """
        self.min_lap_time = min_lap_time
        
        self.current_lap_time: Optional[float] = None
        self.last_lap_time: Optional[float] = None
        self.lap_count: int = 0
        self.best_lap_time: Optional[float] = None
        
        self._last_valid_time: Optional[float] = None
        self._lap_start_time: Optional[float] = None
    
    def process_current_time(self, current_time: float, 
                             flight_time: float) -> Optional[Event]:
        """
        Обработать текущее время круга
        
        Args:
            current_time: Время текущего круга от симулятора (сек)
            flight_time: Время от начала вылета
            
        Returns:
            Событие LAP_END если круг завершён, иначе None
        """
        event = None
        
        # Детекция сброса таймера (новый круг)
        if self._last_valid_time is not None and current_time < self._last_valid_time:
            # Таймер сброшен - круг завершён
            lap_duration = flight_time - (self._lap_start_time or flight_time)
            
            if lap_duration >= self.min_lap_time:
                self.lap_count += 1
                self.last_lap_time = lap_duration
                
                # Проверяем лучший круг
                is_best = False
                if self.best_lap_time is None or lap_duration < self.best_lap_time:
                    self.best_lap_time = lap_duration
                    is_best = True
                
                event = Event(
                    event_type=EventType.LAP_END,
                    timestamp=time.perf_counter(),
                    flight_time=flight_time,
                    lap_number=self.lap_count,
                    value=lap_duration,
                    data={'is_best': is_best}
                )
                
                self._lap_start_time = flight_time
        
        elif self._last_valid_time is None:
            # Первый отсчёт
            self._lap_start_time = flight_time
        
        self._last_valid_time = current_time
        self.current_lap_time = current_time
        
        return event
    
    def reset(self):
        """Сбросить состояние для нового вылета"""
        self.current_lap_time = None
        self.last_lap_time = None
        self.lap_count = 0
        self.best_lap_time = None
        self._last_valid_time = None
        self._lap_start_time = None


class CrashDetector:
    """Детектор крашей по резкому падению скорости/высоты"""
    
    def __init__(self, 
                 velocity_threshold: float = 0.5,
                 altitude_threshold: float = 0.3,
                 min_speed_before_crash: float = 5.0):
        """
        Args:
            velocity_threshold: Порог падения скорости (доля от предыдущей)
            altitude_threshold: Порог падения высоты (м/кадр)
            min_speed_before_crash: Минимальная скорость до краша
        """
        self.velocity_threshold = velocity_threshold
        self.altitude_threshold = altitude_threshold
        self.min_speed_before_crash = min_speed_before_crash
        
        self.last_speed: Optional[float] = None
        self.last_altitude: Optional[float] = None
        self.crash_detected: bool = False
    
    def process(self, speed: float, altitude: float, 
                stick_throttle: float) -> Optional[Event]:
        """
        Проверить на краш
        
        Args:
            speed: Текущая скорость (м/с)
            altitude: Текущая высота (м)
            stick_throttle: Позиция стика газа (-1..1)
            
        Returns:
            Событие CRASH если обнаружен, иначе None
        """
        if self.crash_detected:
            return None
        
        event = None
        
        if self.last_speed is not None and self.last_altitude is not None:
            # Резкое падение скорости при ненулевом газе
            if (self.last_speed > self.min_speed_before_crash and 
                speed < self.last_speed * self.velocity_threshold and
                stick_throttle > 0.2):
                event = Event(
                    event_type=EventType.CRASH,
                    timestamp=time.perf_counter(),
                    flight_time=time.perf_counter(),
                    value=speed,
                    data={
                        'speed_before': self.last_speed,
                        'speed_after': speed,
                        'altitude': altitude
                    }
                )
                self.crash_detected = True
            
            # Резкое падение высоты
            elif (self.last_altitude - altitude) > self.altitude_threshold * 10:
                event = Event(
                    event_type=EventType.CRASH,
                    timestamp=time.perf_counter(),
                    flight_time=time.perf_counter(),
                    value=altitude,
                    data={
                        'altitude_before': self.last_altitude,
                        'altitude_after': altitude,
                        'speed': speed
                    }
                )
                self.crash_detected = True
        
        self.last_speed = speed
        self.last_altitude = altitude
        
        return event
    
    def reset(self):
        """Сбросить состояние"""
        self.last_speed = None
        self.last_altitude = None
        self.crash_detected = False


class LiveDeltaCalculator:
    """Калькулятор live-дельты к лучшему кругу в реальном времени"""
    
    def __init__(self, resample_points: int = 100):
        """
        Args:
            resample_points: Количество точек для ресемплинга
        """
        self.resample_points = resample_points
        
        # Лучший круг (референс)
        self.best_lap_data: Optional[List[Dict]] = None
        self.best_lap_duration: Optional[float] = None
        
        # Текущий круг
        self.current_lap_data: List[Dict] = []
        self.lap_start_time: Optional[float] = None
    
    def set_best_lap(self, lap_data: List[Dict], duration: float):
        """
        Установить лучший круг для сравнения
        
        Args:
            lap_data: Список кадров лучшего круга
            duration: Длительность круга (сек)
        """
        self.best_lap_data = lap_data
        self.best_lap_duration = duration
    
    def start_new_lap(self):
        """Начать новый круг"""
        self.current_lap_data = []
        self.lap_start_time = time.perf_counter()
    
    def add_frame(self, frame_data: Dict):
        """Добавить кадр текущего круга"""
        if self.lap_start_time is not None:
            frame_with_time = {
                **frame_data,
                'lap_time': time.perf_counter() - self.lap_start_time
            }
            self.current_lap_data.append(frame_with_time)
    
    def calculate_delta(self) -> Optional[float]:
        """
        Вычислить текущую дельту к лучшему кругу
        
        Returns:
            Дельта в секундах (положительная = хуже лучшего, отрицательная = лучше)
        """
        if not self.best_lap_data or not self.current_lap_data:
            return None
        
        current_time = self.current_lap_data[-1]['lap_time']
        
        # Находим соответствующую точку в лучшем круге
        best_progress = current_time / self.best_lap_duration
        best_index = int(best_progress * len(self.best_lap_data))
        best_index = min(best_index, len(self.best_lap_data) - 1)
        
        best_frame = self.best_lap_data[best_index]
        current_frame = self.current_lap_data[-1]
        
        # Сравниваем пройденное расстояние (или позицию)
        # Для простоты используем разницу во времени
        delta = current_time - best_frame.get('lap_time', current_time)
        
        return delta
    
    def get_delta_history(self) -> List[float]:
        """Получить историю дельт по всем точкам текущего круга"""
        if not self.best_lap_data:
            return []
        
        deltas = []
        for frame in self.current_lap_data:
            t = frame['lap_time']
            progress = t / self.best_lap_duration
            idx = min(int(progress * len(self.best_lap_data)), len(self.best_lap_data) - 1)
            
            ref_time = self.best_lap_data[idx].get('lap_time', t)
            deltas.append(t - ref_time)
        
        return deltas


class EventManager:
    """Центральный менеджер событий"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Конфигурация детекторов
        """
        config = config or {}
        
        self.lap_detector = LapDetector(
            min_lap_time=config.get('min_lap_time', 5.0)
        )
        
        self.crash_detector = CrashDetector(
            velocity_threshold=config.get('velocity_threshold', 0.5),
            altitude_threshold=config.get('altitude_threshold', 0.3)
        )
        
        self.delta_calculator = LiveDeltaCalculator(
            resample_points=config.get('resample_points', 100)
        )
        
        self.events: List[Event] = []
        self.flight_active = False
        self.flight_start_time: Optional[float] = None
    
    def start_flight(self):
        """Начать вылет"""
        self.flight_active = True
        self.flight_start_time = time.perf_counter()
        
        self.lap_detector.reset()
        self.crash_detector.reset()
        self.delta_calculator.start_new_lap()
        
        self.events.append(Event(
            event_type=EventType.FLIGHT_START,
            timestamp=time.perf_counter(),
            flight_time=0
        ))
    
    def end_flight(self):
        """Завершить вылет"""
        if not self.flight_active:
            return
        
        self.flight_active = False
        
        self.events.append(Event(
            event_type=EventType.FLIGHT_END,
            timestamp=time.perf_counter(),
            flight_time=time.perf_counter() - (self.flight_start_time or time.perf_counter())
        ))
    
    def process_frame(self, 
                      current_time: Optional[float] = None,
                      speed: Optional[float] = None,
                      altitude: Optional[float] = None,
                      stick_throttle: Optional[float] = None,
                      frame_data: Optional[Dict] = None) -> List[Event]:
        """
        Обработать кадр телеметрии
        
        Args:
            current_time: Время текущего круга от симулятора
            speed: Скорость
            altitude: Высота
            stick_throttle: Газ
            frame_data: Дополнительные данные
            
        Returns:
            Список произошедших событий
        """
        if not self.flight_active:
            return []
        
        events = []
        flight_time = time.perf_counter() - (self.flight_start_time or time.perf_counter())
        
        # Детекция круга
        if current_time is not None:
            lap_event = self.lap_detector.process_current_time(current_time, flight_time)
            if lap_event:
                events.append(lap_event)
                self.events.append(lap_event)
                
                # Начинаем новый круг для delta calculator
                self.delta_calculator.start_new_lap()
        
        # Детекция краша
        if speed is not None and altitude is not None and stick_throttle is not None:
            crash_event = self.crash_detector.process(speed, altitude, stick_throttle)
            if crash_event:
                events.append(crash_event)
                self.events.append(crash_event)
        
        # Добавляем кадр в delta calculator
        if frame_data:
            self.delta_calculator.add_frame(frame_data)
        
        return events
    
    def get_live_delta(self) -> Optional[float]:
        """Получить текущую live-дельту"""
        return self.delta_calculator.calculate_delta()
    
    def get_events(self) -> List[Event]:
        """Получить все события вылета"""
        return self.events.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Получить краткую сводку вылета"""
        laps = [e for e in self.events if e.event_type == EventType.LAP_END]
        crashes = [e for e in self.events if e.event_type == EventType.CRASH]
        
        return {
            'lap_count': self.lap_detector.lap_count,
            'best_lap_time': self.lap_detector.best_lap_time,
            'has_crash': len(crashes) > 0,
            'events_count': len(self.events)
        }


if __name__ == "__main__":
    # Тест менеджера событий
    print("Testing Event Manager...")
    
    manager = EventManager()
    manager.start_flight()
    
    # Симуляция данных
    sim_time = 0
    lap_time = 0
    
    for i in range(300):
        sim_time += 0.016  # ~60 FPS
        lap_time += 0.016
        
        # Сброс круга каждые ~30 секунд
        if lap_time > 30:
            lap_time = 0
        
        events = manager.process_frame(
            current_time=lap_time,
            speed=50 + np.sin(i * 0.1) * 10,
            altitude=10 + np.cos(i * 0.05) * 2,
            stick_throttle=0.5,
            frame_data={'speed': 50}
        )
        
        for event in events:
            print(f"Event: {event.event_type.value} at {event.flight_time:.2f}s")
            if event.event_type == EventType.LAP_END:
                print(f"  Lap {event.lap_number}: {event.value:.3f}s")
    
    summary = manager.get_summary()
    print(f"\nSummary: {summary}")
