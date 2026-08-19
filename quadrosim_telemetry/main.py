#!/usr/bin/env python3
"""
ТВ-телеметрия Квадросима - Главный модуль запуска
Система автоматического распознавания телеметрии и анализа полётов дронов
"""
import sys
import time
import signal
import threading
from pathlib import Path
from typing import Optional

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigManager, SystemConfig
from capture import ScreenCapture, MultiROICapture, CaptureFrame
from ocr import HUDRecognizer, FieldParser, OCREngine
from sticks import StickDetector, StickAnalyzer
from logger import TelemetryLogger, TelemetryFrame
from events import EventManager, EventType


class QuadrosimTelemetry:
    """Основной класс системы телеметрии"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load()
        
        # Компоненты
        self.capture: Optional[ScreenCapture] = None
        self.ocr: Optional[HUDRecognizer] = None
        self.stick_detector: Optional[StickDetector] = None
        self.logger: Optional[TelemetryLogger] = None
        self.event_manager: Optional[EventManager] = None
        
        # Состояние
        self.is_running = False
        self.frame_count = 0
        self.current_flight_id: Optional[int] = None
        
        # Очереди и потоки
        self.running_thread: Optional[threading.Thread] = None
    
    def initialize(self):
        """Инициализировать все компоненты"""
        print("=== Инициализация системы ===\n")
        
        # Проверка конфигурации
        if not self._check_config():
            raise RuntimeError("Конфигурация не завершена. Запустите калибровку.")
        
        # OCR движок
        ocr_engine = OCREngine(
            whitelist=self.config.ocr.whitelist,
            min_confidence=self.config.ocr.min_confidence,
            median_filter_size=self.config.ocr.median_filter_size
        )
        self.ocr = HUDRecognizer(ocr_engine)
        
        # Детектор стиков
        self.stick_detector = StickDetector(
            dot_color_lower=self.config.sticks.dot_color_lower,
            dot_color_upper=self.config.sticks.dot_color_upper,
            min_area=self.config.sticks.min_area,
            max_area=self.config.sticks.max_area
        )
        
        # Логгер
        self.logger = TelemetryLogger(
            db_path=self.config.logger.db_path,
            batch_size=self.config.logger.batch_size
        )
        
        # Менеджер событий
        self.event_manager = EventManager({
            'min_lap_time': self.config.analysis.min_lap_time,
            'velocity_threshold': self.config.analysis.crash_velocity_threshold
        })
        
        print("✓ Все компоненты инициализированы\n")
    
    def _check_config(self) -> bool:
        """Проверить наличие калиброванных ROI"""
        required_fields = [
            'speed', 'altitude', 'current_time', 'best_time',
            'left_stick', 'right_stick'
        ]
        
        for field in required_fields:
            roi = self.config_manager.get_roi(field)
            if roi is None:
                print(f"⚠ Не откалибрована зона: {field}")
                return False
        
        print("✓ Конфигурация проверена")
        return True
    
    def start_capture(self):
        """Запустить захват экрана"""
        # Собираем все ROI
        rois = {}
        for field in ['speed', 'altitude', 'current_time', 'best_time', 
                      'battery_voltage', 'battery_current', 'flight_mode',
                      'time_limit', 'laps_info', 'left_stick', 'right_stick']:
            roi = self.config_manager.get_roi(field)
            if roi:
                rois[field] = roi.to_tuple()
        
        if not rois:
            raise RuntimeError("Нет ROI для захвата")
        
        # Используем MultiROICapture для эффективного захвата
        self.capture = MultiROICapture(rois=rois, target_fps=self.config.capture.fps)
        self.capture.capture.init()
        self.capture.capture.start_capture_thread()
        
        print(f"✓ Захват запущен: {self.config.capture.fps} FPS")
    
    def stop_capture(self):
        """Остановить захват"""
        if self.capture:
            self.capture.capture.stop_capture()
            print("✓ Захват остановлен")
    
    def process_frame(self, frame_data: CaptureFrame):
        """Обработать один кадр"""
        self.frame_count += 1
        
        # Извлекаем ROI из кадра
        roi_images = self.capture.extract_rois(frame_data.image)
        
        # Распознаём HUD
        hud_data = self.ocr.recognize_all(roi_images) if self.ocr else {}
        
        # Распознаём стики
        stick_data = {}
        if 'left_stick' in roi_images and 'right_stick' in roi_images:
            sticks = self.stick_detector.process_both_sticks(
                roi_images['left_stick'],
                roi_images['right_stick']
            )
            
            if sticks['left'] and sticks['left'].is_valid():
                stick_data['stick_left_x'] = sticks['left'].x
                stick_data['stick_left_y'] = sticks['left'].y
            
            if sticks['right'] and sticks['right'].is_valid():
                stick_data['stick_right_x'] = sticks['right'].x
                stick_data['stick_right_y'] = sticks['right'].y
        
        # Объединяем данные
        all_data = {**hud_data, **stick_data}
        
        # Логируем
        if self.logger and self.current_flight_id:
            self.logger.log_dict(all_data, frame_id=self.frame_count, flight_id=self.current_flight_id)
        
        # Обрабатываем события
        if self.event_manager:
            events = self.event_manager.process_frame(
                current_time=hud_data.get('current_time'),
                speed=hud_data.get('speed'),
                altitude=hud_data.get('altitude'),
                stick_throttle=stick_data.get('stick_left_y'),
                frame_data=all_data
            )
            
            # Обработка событий
            for event in events:
                if event.event_type == EventType.LAP_END:
                    print(f"\n🏁 Круг {event.lap_number}: {event.value:.3f} сек")
                    if event.data and event.data.get('is_best'):
                        print(f"   ⭐ НОВЫЙ ЛУЧШИЙ КРУГ!")
                
                elif event.event_type == EventType.CRASH:
                    print(f"\n💥 КРАШ обнаружен!")
        
        # Live-дельта
        if self.event_manager:
            delta = self.event_manager.get_live_delta()
            if delta is not None:
                sign = '+' if delta > 0 else ''
                # print(f"Δ {sign}{delta:.3f}s", end='\r')
    
    def run(self):
        """Основной цикл работы"""
        self.is_running = True
        
        print("\n=== Запуск записи телеметрии ===")
        print("Нажмите Ctrl+C для остановки\n")
        
        # Начинаем вылет
        self.current_flight_id = self.logger.start_flight()
        self.event_manager.start_flight()
        
        start_time = time.perf_counter()
        
        try:
            while self.is_running:
                # Получаем кадр
                frame = self.capture.capture.get_frame(timeout=0.1)
                
                if frame:
                    self.process_frame(frame)
                    
                    # Статистика каждую секунду
                    elapsed = time.perf_counter() - start_time
                    if elapsed > 1.0:
                        fps = self.frame_count / elapsed
                        # print(f"\rFPS: {fps:.1f}, Frames: {self.frame_count}", end='')
                        start_time = time.perf_counter()
                        self.frame_count = 0
                
        except KeyboardInterrupt:
            print("\n\nОстановка по запросу пользователя...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Корректное завершение работы"""
        self.is_running = False
        
        # Завершаем вылет
        if self.event_manager:
            self.event_manager.end_flight()
        
        if self.logger:
            self.logger.end_flight(self.current_flight_id)
            self.logger.stop()
        
        # Останавливаем захват
        self.stop_capture()
        
        # Отчёт
        summary = self.event_manager.get_summary() if self.event_manager else {}
        print("\n=== Статистика вылета ===")
        print(f"Кругов: {summary.get('lap_count', 0)}")
        print(f"Лучший круг: {summary.get('best_lap_time', 'N/A')}")
        print(f"Краш: {'Да' if summary.get('has_crash') else 'Нет'}")
        
        print("\n✓ Система остановлена")


def main():
    """Точка входа"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ТВ-телеметрия Квадросима")
    parser.add_argument('--config', default='config.json', help='Путь к конфигу')
    parser.add_argument('--calibrate', action='store_true', help='Запустить калибровку')
    
    args = parser.parse_args()
    
    if args.calibrate:
        from calibration import run_calibration
        run_calibration(config_path=args.config)
        return
    
    # Создаём систему
    system = QuadrosimTelemetry(config_path=args.config)
    
    try:
        system.initialize()
        system.start_capture()
        system.run()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
