"""
Калибровка ROI зон через GUI
Пользователь размечает области интереса мышью
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json

from config import ConfigManager, ROI, HUDConfig


@dataclass
class CalibrationState:
    """Состояние процесса калибровки"""
    current_field: str = ""
    drawing: bool = False
    start_point: Optional[Tuple[int, int]] = None
    end_point: Optional[Tuple[int, int]] = None
    roi_complete: bool = False


class ROICalibrator:
    """Калибратор ROI зон с помощью мыши"""
    
    # Список полей для калибровки в порядке обхода
    FIELDS_TO_CALIBRATE = [
        "pilot_info",       # Верх-лево: регион и ФИО
        "datetime",         # Дата/время
        "battery_voltage",  # Напряжение батареи
        "battery_current",  # Ток батареи
        "flight_mode",      # Режим полёта
        "time_limit",       # Верх-право: лимит времени
        "speed",            # Центр-лево: скорость
        "altitude",         # Центр-право: высота
        "laps_info",        # Низ-лево: круги
        "current_time",     # Низ-лево: текущее время
        "best_time",        # Низ-право: лучшее время
        "left_stick",       # Низ-центр: левый стик
        "right_stick",      # Низ-центр: правый стик
    ]
    
    FIELD_DESCRIPTIONS = {
        "pilot_info": "Верх-лево: Регион и ФИО пилота",
        "datetime": "Дата/время (ДД.ММ.ГГ ЧЧ:ММ:СС)",
        "battery_voltage": "Напряжение батареи (XX,XV)",
        "battery_current": "Ток батареи (X,XA)",
        "flight_mode": "Режим полёта (ACRO)",
        "time_limit": "Верх-право: Лимит времени (ММ:СС)",
        "speed": "Центр-лево: Скорость (NN КМ/Ч)",
        "altitude": "Центр-право: Высота (N М)",
        "laps_info": "Низ-лево: Круги (X / Y)",
        "current_time": "Низ-лево: Текущее время (ММ:СС.mmm)",
        "best_time": "Низ-право: Лучшее время (ММ:СС.mmm)",
        "left_stick": "Низ-центр: Левый стик (газ/руль)",
        "right_stick": "Низ-центр: Правый стик (тангаж/крен)",
    }
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.state = CalibrationState()
        self.rois: dict = {}
        self.window_name = "Quadrosim Telemetry - ROI Calibration"
        self.frame: Optional[np.ndarray] = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.drawing = True
            self.state.start_point = (x, y)
            self.state.roi_complete = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.state.drawing:
                self.state.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.state.drawing = False
            self.state.end_point = (x, y)
            self.state.roi_complete = True
            
            # Сохраняем ROI
            if self.state.start_point and self.state.end_point:
                x1, y1 = min(self.state.start_point[0], self.state.end_point[0]), \
                         min(self.state.start_point[1], self.state.end_point[1])
                x2, y2 = max(self.state.start_point[0], self.state.end_point[0]), \
                         max(self.state.start_point[1], self.state.end_point[1])
                
                roi = ROI(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1
                )
                self.rois[self.state.current_field] = roi
                print(f"✓ {self.FIELD_DESCRIPTIONS[self.state.current_field]}: {roi}")
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Нарисовать оверлей с инструкциями и выделением"""
        overlay = frame.copy()
        
        # Инструкция
        instruction = f"Field: {self.state.current_field} - {self.FIELD_DESCRIPTIONS[self.state.current_field]}"
        cv2.putText(overlay, instruction, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        progress = f"Progress: {len(self.rois)}/{len(self.FIELDS_TO_CALIBRATE)}"
        cv2.putText(overlay, progress, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Рисуем прямоугольник при рисовании
        if self.state.drawing and self.state.start_point and self.state.end_point:
            cv2.rectangle(overlay, self.state.start_point, self.state.end_point, 
                         (0, 255, 0), 2)
        
        # Рисуем сохранённые ROI
        for field_name, roi in self.rois.items():
            if field_name != self.state.current_field:
                pt1 = (roi.x, roi.y)
                pt2 = (roi.x + roi.width, roi.y + roi.height)
                color = (0, 255, 255) if field_name in ["left_stick", "right_stick"] else (255, 0, 0)
                cv2.rectangle(overlay, pt1, pt2, color, 1)
                # Подпись
                label = field_name.replace("_", " ")
                cv2.putText(overlay, label, (pt1[0], pt1[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def calibrate(self, screenshot_path: Optional[str] = None) -> bool:
        """
        Запустить процесс калибровки
        
        Args:
            screenshot_path: Путь к скриншоту симулятора (опционально)
                            Если не указан, будет сделан захват экрана
            
        Returns:
            True если калибровка успешна, False если отменена
        """
        import time
        
        # Загружаем или делаем скриншот
        if screenshot_path:
            self.frame = cv2.imread(screenshot_path)
            if self.frame is None:
                print(f"Error: Cannot load screenshot from {screenshot_path}")
                return False
        else:
            print("Делайте скриншот симулятора через 3 секунды...")
            time.sleep(3)
            # Используем mss для захвата экрана
            try:
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[0]  # Все мониторы
                    screenshot = sct.grab(monitor)
                    self.frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGBA2RGB)
            except ImportError:
                print("Error: mss library not installed. Install with: pip install mss")
                return False
            except Exception as e:
                print(f"Error capturing screen: {e}")
                return False
        
        # Настраиваем окно
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== Калибровка ROI зон ===")
        print("Инструкция:")
        print("  1. Для каждой зоны: зажмите ЛКМ и тяните до выделения нужной области")
        print("  2. Отпустите ЛКМ для сохранения зоны")
        print("  3. Нажмите 'n' для перехода к следующей зоне")
        print("  4. Нажмите 'q' для выхода без сохранения")
        print("  5. Нажмите 's' для сохранения текущего состояния\n")
        
        field_index = 0
        
        while field_index < len(self.FIELDS_TO_CALIBRATE):
            self.state.current_field = self.FIELDS_TO_CALIBRATE[field_index]
            
            # Ждём завершения рисования ROI для текущей зоны
            self.state.roi_complete = False
            self.state.drawing = False
            
            while not self.state.roi_complete:
                display_frame = self.draw_overlay(self.frame)
                cv2.imshow(self.window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("\nКалибровка отменена")
                    return False
                elif key == ord('s'):
                    self._save_progress()
                    print("Прогресс сохранён")
                elif key == ord('n') and self.state.roi_complete:
                    break
            
            # Переходим к следующему полю
            field_index += 1
        
        cv2.destroyAllWindows()
        
        # Сохраняем все ROI в конфиг
        self._save_to_config()
        
        print(f"\n✓ Калибровка завершена! Сохранено {len(self.rois)} зон")
        return True
    
    def _save_progress(self):
        """Сохранить текущий прогресс"""
        temp_file = "calibration_temp.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({k: asdict(v) for k, v in self.rois.items()}, f, indent=2)
    
    def _save_to_config(self):
        """Сохранить ROI в основной конфиг"""
        for field_name, roi in self.rois.items():
            self.config_manager.update_roi(field_name, roi)
        
        print(f"Конфигурация сохранена в {self.config_manager.config_path}")


def run_calibration(screenshot_path: Optional[str] = None, config_path: str = "config.json"):
    """
    Запустить калибровку ROI
    
    Args:
        screenshot_path: Путь к скриншоту (опционально)
        config_path: Путь к файлу конфигурации
    """
    from dataclasses import asdict
    
    config_manager = ConfigManager(config_path)
    calibrator = ROICalibrator(config_manager)
    
    success = calibrator.calibrate(screenshot_path)
    
    if success:
        print("\n=== Калибровка успешно завершена ===")
        print(f"Конфигурация сохранена в: {config_manager.config_path.absolute()}")
        print("\nЗапуск основного приложения:")
        print("  python main.py")
    else:
        print("\n=== Калибровка прервана ===")
    
    return success


if __name__ == "__main__":
    import sys
    
    screenshot = sys.argv[1] if len(sys.argv) > 1 else None
    run_calibration(screenshot)
