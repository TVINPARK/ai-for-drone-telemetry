"""
Модуль OCR для распознавания числовых полей HUD
Использует Tesseract/PaddleOCR с предобработкой и валидацией
"""
import re
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import statistics

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Install with: pip install pytesseract")


class MedianFilter:
    """Медианный фильтр для сглаживания выбросов"""
    
    def __init__(self, window_size: int = 5):
        self.window: deque = deque(maxlen=window_size)
    
    def add(self, value: float) -> Optional[float]:
        self.window.append(value)
        if len(self.window) < 3:
            return value
        return statistics.median(self.window)
    
    def reset(self):
        self.window.clear()


class FieldParser:
    """Парсеры для различных типов полей HUD"""
    
    @staticmethod
    def parse_time(text: str) -> Optional[float]:
        """
        Парсинг времени в форматах:
        - ММ:СС.mmm (1:23.456)
        - ММ:СС (1:23)
        - СС.mmm (23.456)
        
        Returns: время в секундах или None
        """
        if not text:
            return None
        
        # Удаляем лишние символы
        text = text.strip().upper()
        
        # Паттерн: ММ:СС.mmm или ММ:СС
        pattern_mm_ss_ms = r'^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$'
        match = re.match(pattern_mm_ss_ms, text)
        
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            ms = int(match.group(3).ljust(3, '0')) / 1000.0 if match.group(3) else 0
            return minutes * 60 + seconds + ms
        
        # Паттерн: только секунды с миллисекундами
        pattern_ss_ms = r'^(\d{1,2})\.(\d{1,3})$'
        match = re.match(pattern_ss_ms, text)
        
        if match:
            seconds = int(match.group(1))
            ms = int(match.group(2).ljust(3, '0')) / 1000.0
            return seconds + ms
        
        # Паттерн: только целые секунды
        pattern_ss = r'^(\d+)$'
        match = re.match(pattern_ss, text)
        
        if match:
            return float(match.group(1))
        
        return None
    
    @staticmethod
    def parse_number(text: str) -> Optional[float]:
        """
        Парсинг числа с возможной запятой/точкой как разделителем
        
        Examples: 123, 123.45, 123,45
        """
        if not text:
            return None
        
        text = text.strip()
        
        # Заменяем запятую на точку
        text = text.replace(',', '.')
        
        # Пробуем распарсить float
        try:
            return float(text)
        except ValueError:
            pass
        
        # Пробуем int
        try:
            return float(int(text))
        except ValueError:
            pass
        
        return None
    
    @staticmethod
    def parse_voltage(text: str) -> Optional[float]:
        """Парсинг напряжения батареи (формат XX,XV)"""
        if not text:
            return None
        
        text = text.strip().upper().replace('V', '')
        return FieldParser.parse_number(text)
    
    @staticmethod
    def parse_current(text: str) -> Optional[float]:
        """Парсинг тока (формат X,XA)"""
        if not text:
            return None
        
        text = text.strip().upper().replace('A', '')
        return FieldParser.parse_number(text)
    
    @staticmethod
    def parse_laps(text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Парсинг информации о кругах (формат "X / Y")
        
        Returns: (текущий_круг, всего_кругов)
        """
        if not text:
            return None, None
        
        # Паттерн: цифра / цифра
        pattern = r'(\d+)\s*/\s*(\d+)'
        match = re.search(pattern, text)
        
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # Только текущий круг
        pattern_single = r'(\d+)'
        match = re.search(pattern_single, text)
        
        if match:
            return int(match.group(1)), None
        
        return None, None
    
    @staticmethod
    def parse_speed(text: str) -> Optional[float]:
        """Парсинг скорости (формат NN КМ/Ч)"""
        if not text:
            return None
        
        text = text.strip().upper()
        # Удаляем единицы измерения
        text = re.sub(r'[КМЧС/\s]', '', text)
        return FieldParser.parse_number(text)
    
    @staticmethod
    def parse_altitude(text: str) -> Optional[float]:
        """Парсинг высоты (формат N М)"""
        if not text:
            return None
        
        text = text.strip().upper().replace('М', '').replace('M', '')
        return FieldParser.parse_number(text)


class OCREngine:
    """Основной движок OCR"""
    
    def __init__(self, 
                 whitelist: str = "0123456789.,:VAMC/",
                 min_confidence: float = 0.7,
                 median_filter_size: int = 5,
                 lang: str = "eng"):
        """
        Args:
            whitelist: Разрешённые символы для OCR
            min_confidence: Минимальная уверенность распознавания
            median_filter_size: Размер окна медианного фильтра
            lang: Язык для Tesseract
        """
        self.whitelist = whitelist
        self.min_confidence = min_confidence
        self.lang = lang
        
        # Медианные фильтры для каждого поля
        self.filters: Dict[str, MedianFilter] = {}
        self.median_filter_size = median_filter_size
        
        # Кэш последних значений для валидации
        self.last_values: Dict[str, Any] = {}
        
        if TESSERACT_AVAILABLE:
            # Настраиваем Tesseract
            custom_config = f'--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}'
            self.tesseract_config = custom_config
        else:
            self.tesseract_config = None
    
    def get_filter(self, field_name: str) -> MedianFilter:
        """Получить или создать фильтр для поля"""
        if field_name not in self.filters:
            self.filters[field_name] = MedianFilter(self.median_filter_size)
        return self.filters[field_name]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для улучшения OCR
        
        - Конвертация в оттенки серого
        - Бинаризация (адаптивный порог)
        - Увеличение контраста
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Увеличиваем размер для лучшей читаемости
        scale = 2.0
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Применяем адаптивный порог
        binary = cv2.adaptiveThreshold(
            resized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Морфологические операции для удаления шума
        kernel = np.ones((1, 1), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return denoised
    
    def recognize_text(self, image: np.ndarray, field_name: str = "unknown") -> Tuple[str, float]:
        """
        Распознать текст на изображении
        
        Args:
            image: Изображение ROI зоны
            field_name: Имя поля для логгирования
        
        Returns:
            (распознанный_текст, уверенность)
        """
        if not TESSERACT_AVAILABLE:
            return "", 0.0
        
        # Предобработка
        processed = self.preprocess_image(image)
        
        try:
            # Распознавание с кастомной конфигурацией
            result = pytesseract.image_to_string(
                processed,
                config=self.tesseract_config,
                lang=self.lang
            )
            
            # Получаем данные о уверенности
            data = pytesseract.image_to_data(
                processed,
                config=self.tesseract_config,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )
            
            # Средняя уверенность по всем распознанным блокам
            confidences = [c for c in data['conf'] if c > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            text = result.strip()
            return text, avg_confidence / 100.0
            
        except Exception as e:
            print(f"OCR error for {field_name}: {e}")
            return "", 0.0
    
    def recognize_field(self, 
                        image: np.ndarray, 
                        field_name: str,
                        parser_func=None) -> Optional[Any]:
        """
        Распознать и распарсить поле HUD
        
        Args:
            image: Изображение ROI зоны
            field_name: Имя поля (для фильтра)
            parser_func: Функция парсинга (например, FieldParser.parse_time)
        
        Returns:
            Распарсенное значение или None
        """
        # Распознаём текст
        text, confidence = self.recognize_text(image, field_name)
        
        if confidence < self.min_confidence or not text:
            return None
        
        # Если есть функция парсера - используем её
        if parser_func:
            value = parser_func(text)
        else:
            value = FieldParser.parse_number(text)
        
        # Применяем медианный фильтр
        if value is not None:
            filter_obj = self.get_filter(field_name)
            filtered_value = filter_obj.add(value)
            self.last_values[field_name] = filtered_value
            return filtered_value
        
        return None
    
    def reset_filters(self):
        """Сбросить все фильтры (для нового вылета)"""
        for filter_obj in self.filters.values():
            filter_obj.reset()
        self.last_values.clear()


class HUDRecognizer:
    """Распознаватель всех полей HUD"""
    
    def __init__(self, ocr_engine: Optional[OCREngine] = None):
        self.ocr = ocr_engine or OCREngine()
        
        # Маппинг полей к функциям парсинга
        self.field_parsers = {
            'current_time': FieldParser.parse_time,
            'best_time': FieldParser.parse_time,
            'time_limit': FieldParser.parse_time,
            'battery_voltage': FieldParser.parse_voltage,
            'battery_current': FieldParser.parse_current,
            'speed': FieldParser.parse_speed,
            'altitude': FieldParser.parse_altitude,
            'laps_info': FieldParser.parse_laps,
            'datetime': None,  # Текстовое поле
            'pilot_info': None,  # Текстовое поле
            'flight_mode': None,  # Текстовое поле
        }
    
    def recognize_all(self, roi_images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Распознать все поля HUD из словаря ROI изображений
        
        Args:
            roi_images: {field_name: image}
        
        Returns:
            Словарь распознанных значений
        """
        results = {}
        
        for field_name, image in roi_images.items():
            parser_func = self.field_parsers.get(field_name)
            value = self.ocr.recognize_field(image, field_name, parser_func)
            
            if value is not None:
                results[field_name] = value
        
        return results
    
    def recognize_single(self, image: np.ndarray, field_name: str) -> Any:
        """Распознать одно поле"""
        parser_func = self.field_parsers.get(field_name)
        return self.ocr.recognize_field(image, field_name, parser_func)


if __name__ == "__main__":
    # Тест OCR на примере
    print("Testing OCR module...")
    
    # Создаём тестовое изображение с цифрами
    test_img = np.zeros((50, 100), dtype=np.uint8)
    cv2.putText(test_img, "1:23.456", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    ocr = OCREngine()
    
    if TESSERACT_AVAILABLE:
        text, conf = ocr.recognize_text(test_img, "test")
        print(f"Recognized: '{text}' (confidence: {conf:.2f})")
        
        time_val = ocr.recognize_field(test_img, "current_time", FieldParser.parse_time)
        print(f"Parsed time: {time_val} seconds")
    else:
        print("Tesseract not available, skipping OCR test")
    
    # Тест парсеров
    print("\nParser tests:")
    print(f"  '1:23.456' -> {FieldParser.parse_time('1:23.456')} sec")
    print(f"  '1:23' -> {FieldParser.parse_time('1:23')} sec")
    print(f"  '23.456' -> {FieldParser.parse_time('23.456')} sec")
    print(f"  '123,45' -> {FieldParser.parse_number('123,45')}")
    print(f"  '2 / 5' -> {FieldParser.parse_laps('2 / 5')}")