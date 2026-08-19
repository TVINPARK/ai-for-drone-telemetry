"""
Модуль распознавания стиков управления
Использует OpenCV для детекции цветных точек относительно центра креста
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class StickPosition:
    """Позиция стика с нормализованными координатами [-1, 1]"""
    x: float  # Руль/Крен (-1 = влево/влево, 1 = вправо/вправо)
    y: float  # Газ/Тангаж (-1 = вниз/вперёд, 1 = вверх/назад)
    confidence: float
    raw_x: int  # Сырые координаты в пикселях
    raw_y: int
    
    def is_valid(self) -> bool:
        return self.confidence > 0.5 and -1.0 <= self.x <= 1.0 and -1.0 <= self.y <= 1.0


class StickDetector:
    """
    Детектор позиции стиков по цветным точкам
    
    Алгоритм:
    1. Конвертация в HSV
    2. Пороговая обработка по цвету точки
    3. Поиск контуров
    4. Фильтрация по площади
    5. Вычисление центра точки
    6. Нормализация относительно центра креста
    """
    
    def __init__(self,
                 dot_color_lower: np.ndarray = None,
                 dot_color_upper: np.ndarray = None,
                 min_area: int = 10,
                 max_area: int = 500,
                 cross_detection: bool = True):
        """
        Args:
            dot_color_lower: Нижняя граница HSV для цвета точки
            dot_color_upper: Верхняя граница HSV для цвета точки
            min_area: Минимальная площадь точки
            max_area: Максимальная площадь точки
            cross_detection: Пытаться ли детектить крест для автокалибровки
        """
        self.dot_color_lower = dot_color_lower or np.array([0, 100, 100])
        self.dot_color_upper = dot_color_upper or np.array([15, 255, 255])
        self.min_area = min_area
        self.max_area = max_area
        self.cross_detection = cross_detection
        
        # Центры крестов (задаются калибровкой или автодетектом)
        self.left_cross_center: Optional[Tuple[int, int]] = None
        self.right_cross_center: Optional[Tuple[int, int]] = None
        
        # Размеры зон стиков (для нормализации)
        self.left_stick_radius: int = 50
        self.right_stick_radius: int = 50
    
    def set_cross_centers(self, 
                          left_center: Tuple[int, int], 
                          right_center: Tuple[int, int],
                          left_radius: int = 50,
                          right_radius: int = 50):
        """
        Установить центры крестов и радиусы нормализации
        
        Args:
            left_center: Центр левого креста (x, y)
            right_center: Центр правого креста (x, y)
            left_radius: Радиус левого стика (макс отклонение точки)
            right_radius: Радиус правого стика
        """
        self.left_cross_center = left_center
        self.right_cross_center = right_center
        self.left_stick_radius = left_radius
        self.right_stick_radius = right_radius
    
    def detect_dot(self, image: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """
        Детектировать точку на изображении
        
        Args:
            image: ROI изображение зоны стика
            
        Returns:
            (center_x, center_y, confidence) или None
        """
        if image is None or image.size == 0:
            return None
        
        # Конвертация в HSV
        if len(image.shape) == 2:
            hsv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Пороговая обработка по цвету
        mask = cv2.inRange(hsv, self.dot_color_lower, self.dot_color_upper)
        
        # Морфологические операции
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Находим самый большой подходящий контур
        best_contour = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area <= area <= self.max_area:
                if area > best_area:
                    best_area = area
                    best_contour = contour
        
        if best_contour is None:
            # Пробуем найти любой контур если ничего не подошло
            if contours:
                best_contour = max(contours, key=cv2.contourArea)
                best_area = cv2.contourArea(best_contour)
            else:
                return None
        
        # Вычисляем центр контура
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Confidence на основе площади и круглости
        perimeter = cv2.arcLength(best_contour, True)
        if perimeter == 0:
            confidence = 0.5
        else:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            confidence = min(1.0, circularity * (best_area / self.max_area))
        
        return (center_x, center_y, confidence)
    
    def detect_cross_center(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Детектировать центр креста по геометрии
        
        Args:
            image: ROI изображение зоны стика
            
        Returns:
            (center_x, center_y) или None
        """
        if image is None or image.size == 0:
            return None
        
        # Конвертация в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Детекция линий (крест состоит из двух пересекающихся линий)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
        
        if lines is None:
            # Fallback: используем центр изображения
            h, w = gray.shape[:2]
            return (w // 2, h // 2)
        
        # Собираем горизонтальные и вертикальные линии
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            rho, theta = line[0]
            if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                vertical_lines.append((rho, theta))
            elif np.pi / 4 < theta < 3 * np.pi / 4:
                horizontal_lines.append((rho, theta))
        
        # Если нашли и горизонтальные и вертикальные линии
        if horizontal_lines and vertical_lines:
            # Усредняем позиции
            avg_h_rho = sum(l[0] for l in horizontal_lines) / len(horizontal_lines)
            avg_v_rho = sum(l[0] for l in vertical_lines) / len(vertical_lines)
            
            # Приблизительный центр (упрощённо)
            h, w = gray.shape[:2]
            return (w // 2, h // 2)
        
        # Fallback: центр изображения
        h, w = gray.shape[:2]
        return (w // 2, h // 2)
    
    def normalize_position(self, 
                           point: Tuple[int, int], 
                           center: Tuple[int, int], 
                           radius: int) -> Tuple[float, float]:
        """
        Нормализовать позицию точки относительно центра креста
        
        Args:
            point: Координаты точки (x, y)
            center: Центр креста (x, y)
            radius: Максимальный радиус отклонения
            
        Returns:
            (norm_x, norm_y) в диапазоне [-1, 1]
        """
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        
        # Нормализация с ограничением
        norm_x = np.clip(dx / radius, -1.0, 1.0)
        norm_y = np.clip(-dy / radius, -1.0, 1.0)  # Инвертируем Y (вверх = положительно)
        
        return (norm_x, norm_y)
    
    def process_stick(self, 
                      image: np.ndarray, 
                      cross_center: Optional[Tuple[int, int]],
                      radius: int) -> Optional[StickPosition]:
        """
        Обработать изображение одного стика
        
        Args:
            image: ROI изображение зоны стика
            cross_center: Центр креста (или None для автодетекта)
            radius: Радиус для нормализации
            
        Returns:
            StickPosition или None
        """
        # Детектируем точку
        dot_result = self.detect_dot(image)
        
        if dot_result is None:
            return None
        
        dot_x, dot_y, confidence = dot_result
        
        # Если центр не задан, используем центр изображения
        if cross_center is None:
            h, w = image.shape[:2]
            cross_center = (w // 2, h // 2)
        
        # Нормализуем позицию
        norm_x, norm_y = self.normalize_position(
            (dot_x, dot_y), 
            cross_center, 
            radius
        )
        
        return StickPosition(
            x=norm_x,
            y=norm_y,
            confidence=confidence,
            raw_x=dot_x,
            raw_y=dot_y
        )
    
    def process_both_sticks(self, 
                            left_image: np.ndarray, 
                            right_image: np.ndarray) -> Dict[str, Optional[StickPosition]]:
        """
        Обработать оба стика одновременно
        
        Args:
            left_image: ROI левого стика
            right_image: ROI правого стика
            
        Returns:
            {'left': StickPosition, 'right': StickPosition}
        """
        left_pos = self.process_stick(
            left_image, 
            self.left_cross_center, 
            self.left_stick_radius
        )
        
        right_pos = self.process_stick(
            right_image, 
            self.right_cross_center, 
            self.right_stick_radius
        )
        
        return {
            'left': left_pos,
            'right': right_pos
        }


class StickAnalyzer:
    """Анализатор данных стиков для пост-обработки"""
    
    @staticmethod
    def calculate_smoothness(positions: list) -> float:
        """
        Вычислить плавность управления (RMS производной)
        
        Args:
            positions: Список позиций StickPosition
            
        Returns:
            RMS производной (меньше = плавнее)
        """
        if len(positions) < 2:
            return 0.0
        
        derivatives = []
        for i in range(1, len(positions)):
            if positions[i] and positions[i-1]:
                dx = positions[i].x - positions[i-1].x
                dy = positions[i].y - positions[i-1].y
                derivative = np.sqrt(dx**2 + dy**2)
                derivatives.append(derivative)
        
        if not derivatives:
            return 0.0
        
        return np.sqrt(np.mean(np.array(derivatives)**2))
    
    @staticmethod
    def count_corrections(positions: list, axis: str = 'both') -> int:
        """
        Подсчитать количество коррекций (смен знака производной)
        
        Args:
            positions: Список позиций StickPosition
            axis: 'x', 'y', или 'both'
            
        Returns:
            Количество коррекций
        """
        if len(positions) < 3:
            return 0
        
        corrections = 0
        
        if axis in ['x', 'both']:
            prev_sign_x = 0
            for i in range(1, len(positions)):
                if positions[i] and positions[i-1]:
                    dx = positions[i].x - positions[i-1].x
                    sign = np.sign(dx)
                    if sign != 0 and prev_sign_x != 0 and sign != prev_sign_x:
                        corrections += 1
                    prev_sign_x = sign
        
        if axis in ['y', 'both']:
            prev_sign_y = 0
            for i in range(1, len(positions)):
                if positions[i] and positions[i-1]:
                    dy = positions[i].y - positions[i-1].y
                    sign = np.sign(dy)
                    if sign != 0 and prev_sign_y != 0 and sign != prev_sign_y:
                        corrections += 1
                    prev_sign_y = sign
        
        return corrections
    
    @staticmethod
    def get_statistics(positions: list) -> Dict[str, float]:
        """
        Получить статистику по позициям стика
        
        Returns:
            Dict со статистикой (avg, max, min для x и y)
        """
        valid_positions = [p for p in positions if p and p.is_valid()]
        
        if not valid_positions:
            return {}
        
        x_values = [p.x for p in valid_positions]
        y_values = [p.y for p in valid_positions]
        
        return {
            'avg_x': np.mean(x_values),
            'max_x': np.max(x_values),
            'min_x': np.min(x_values),
            'avg_y': np.mean(y_values),
            'max_y': np.max(y_values),
            'min_y': np.min(y_values),
            'count': len(valid_positions)
        }


if __name__ == "__main__":
    # Тест детектора стиков
    print("Testing Stick Detector...")
    
    detector = StickDetector()
    
    # Создаём тестовое изображение с точкой
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Рисуем оранжевую точку в центре
    cv2.circle(test_img, (60, 40), 8, (0, 140, 255), -1)  # BGR orange
    
    result = detector.detect_dot(test_img)
    print(f"Dot detection: {result}")
    
    # Тест нормализации
    if result:
        norm = detector.normalize_position(
            (result[0], result[1]),
            (50, 50),
            50
        )
        print(f"Normalized position: {norm}")
    
    # Тест анализатора
    analyzer = StickAnalyzer()
    
    # Создаём фейковые данные
    fake_positions = [
        StickPosition(x=0.0, y=0.0, confidence=1.0, raw_x=50, raw_y=50),
        StickPosition(x=0.1, y=0.1, confidence=1.0, raw_x=55, raw_y=45),
        StickPosition(x=-0.1, y=0.0, confidence=1.0, raw_x=45, raw_y=50),
        StickPosition(x=0.2, y=-0.1, confidence=1.0, raw_x=60, raw_y=55),
    ]
    
    smoothness = analyzer.calculate_smoothness(fake_positions)
    corrections = analyzer.count_corrections(fake_positions)
    stats = analyzer.get_statistics(fake_positions)
    
    print(f"\nAnalyzer tests:")
    print(f"  Smoothness (RMS): {smoothness:.4f}")
    print(f"  Corrections: {corrections}")
    print(f"  Statistics: {stats}")
