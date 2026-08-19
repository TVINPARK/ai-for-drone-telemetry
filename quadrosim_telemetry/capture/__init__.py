"""
Модуль захвата экрана (Capture Module)
Использует dxcam для высокоскоростного захвата с поддержкой ROI
"""
import time
import threading
import queue
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    print("Warning: dxcam not available. Install with: pip install dxcam")

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


@dataclass
class CaptureFrame:
    """Кадр с метаданными"""
    image: np.ndarray
    timestamp: float  # performance_counter
    frame_id: int
    fps: float


class ScreenCapture:
    """
    Класс для захвата экрана с поддержкой ROI
    
    Поддерживает два бэкенда:
    - dxcam: DirectX-based, высокая производительность на Windows
    - mss: кроссплатформенный, хорошая производительность
    """
    
    def __init__(self, 
                 monitor_index: int = 0,
                 roi: Optional[tuple] = None,
                 target_fps: int = 60,
                 backend: str = "auto"):
        """
        Args:
            monitor_index: Индекс монитора (0 = все мониторы для mss, 1 = первый и т.д.)
            roi: Region of Interest (x1, y1, x2, y2) или None для полного экрана
            target_fps: Целевая частота кадров
            backend: "dxcam", "mss", или "auto" для автовыбора
        """
        self.monitor_index = monitor_index
        self.roi = roi
        self.target_fps = target_fps
        self.backend = backend
        
        self.camera = None
        self.is_capturing = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self.frame_id = 0
        self.last_frame_time = 0
        self.fps_history: list = []
        
        self._select_backend()
    
    def _select_backend(self):
        """Выбрать лучший доступный бэкенд"""
        if self.backend == "auto":
            if DXCAM_AVAILABLE:
                self.backend = "dxcam"
            elif MSS_AVAILABLE:
                self.backend = "mss"
            else:
                raise RuntimeError("No capture backend available. Install dxcam or mss.")
        
        if self.backend == "dxcam" and not DXCAM_AVAILABLE:
            raise RuntimeError("dxcam requested but not available")
        if self.backend == "mss" and not MSS_AVAILABLE:
            raise RuntimeError("mss requested but not available")
    
    def init(self):
        """Инициализировать захват"""
        if self.backend == "dxcam":
            self._init_dxcam()
        elif self.backend == "mss":
            self._init_mss()
    
    def _init_dxcam(self):
        """Инициализация dxcam"""
        # Создаём устройство
        self.camera = dxcam.create(device_idx=0, max_buffer_len=4)
        
        if self.roi:
            # Для dxcam ROI задаётся при захвате
            pass
        
        print(f"DXCam initialized: {self.camera.width}x{self.camera.height}")
    
    def _init_mss(self):
        """Инициализация mss"""
        self.sct = mss.mss()
        
        # Определяем монитор
        if self.monitor_index == 0:
            # Все мониторы
            self.monitor = self.sct.monitors[0]
        else:
            idx = min(self.monitor_index, len(self.sct.monitors) - 1)
            self.monitor = self.sct.monitors[idx]
        
        # Применяем ROI если указан
        if self.roi:
            x1, y1, x2, y2 = self.roi
            self.monitor = {
                "left": self.monitor["left"] + x1,
                "top": self.monitor["top"] + y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        
        print(f"MSS initialized: {self.monitor['width']}x{self.monitor['height']}")
    
    def capture(self) -> Optional[np.ndarray]:
        """Сделать один захват кадра"""
        if self.backend == "dxcam":
            frame = self.camera.grab(region=self.roi, color_mode="RGB")
            if frame is not None:
                return np.array(frame)
        elif self.backend == "mss":
            screenshot = self.sct.grab(self.monitor)
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2RGB)
        
        return None
    
    def start_capture_thread(self):
        """Запустить поток захвата"""
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"Capture started: {self.target_fps} FPS, backend={self.backend}")
    
    def stop_capture(self):
        """Остановить захват"""
        self.is_capturing = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        print("Capture stopped")
    
    def _capture_loop(self):
        """Основной цикл захвата в отдельном потоке"""
        frame_interval = 1.0 / self.target_fps
        last_time = time.perf_counter()
        
        while self.is_capturing:
            current_time = time.perf_counter()
            
            # Ограничиваем FPS
            elapsed = current_time - last_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue
            
            # Захват кадра
            frame = self.capture()
            if frame is not None:
                self.frame_id += 1
                
                # Вычисляем FPS
                dt = current_time - self.last_frame_time if self.last_frame_time > 0 else frame_interval
                current_fps = 1.0 / dt if dt > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                
                # Создаём объект кадра
                capture_frame = CaptureFrame(
                    image=frame,
                    timestamp=current_time,
                    frame_id=self.frame_id,
                    fps=np.mean(self.fps_history[-10:]) if self.fps_history else 0
                )
                
                # Пытаемся добавить в очередь
                try:
                    self.frame_queue.put_nowait(capture_frame)
                except queue.Full:
                    # Пропускаем кадр если очередь полна
                    pass
                
                self.last_frame_time = current_time
            
            last_time = current_time
    
    def get_frame(self, timeout: float = 0.1) -> Optional[CaptureFrame]:
        """Получить следующий кадр из очереди"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_current_fps(self) -> float:
        """Получить текущий FPS"""
        if not self.fps_history:
            return 0.0
        return np.mean(self.fps_history[-10:])


class MultiROICapture:
    """
    Захват нескольких ROI одновременно
    
    Оптимизировано для захвата только нужных зон HUD
    """
    
    def __init__(self, rois: Dict[str, tuple], target_fps: int = 60):
        """
        Args:
            rois: Словарь {имя_зоны: (x1, y1, x2, y2)}
            target_fps: Целевая частота кадров
        """
        self.rois = rois
        self.target_fps = target_fps
        
        # Объединяем все ROI в одну bounding box для захвата
        all_coords = []
        for roi in rois.values():
            all_coords.extend(roi)
        
        self.global_roi = (
            min(all_coords[0::2]),  # x1
            min(all_coords[1::2]),  # y1
            max(all_coords[0::2]),  # x2
            max(all_coords[1::2])   # y2
        )
        
        self.capture = ScreenCapture(roi=self.global_roi, target_fps=target_fps)
        
        # Смещения для каждой зоны относительно global_roi
        self.offsets = {}
        for name, roi in rois.items():
            self.offsets[name] = (
                roi[0] - self.global_roi[0],
                roi[1] - self.global_roi[1],
                roi[2] - self.global_roi[0],
                roi[3] - self.global_roi[1]
            )
    
    def extract_rois(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Извлечь все ROI из кадра"""
        result = {}
        for name, offset in self.offsets.items():
            x1, y1, x2, y2 = offset
            result[name] = frame[y1:y2, x1:x2]
        return result


if __name__ == "__main__":
    # Тест захвата
    import cv2
    
    print("Testing screen capture...")
    
    capture = ScreenCapture(target_fps=30)
    capture.init()
    capture.start_capture_thread()
    
    try:
        for i in range(100):
            frame_data = capture.get_frame(timeout=0.1)
            if frame_data:
                print(f"Frame {frame_data.frame_id}: {frame_data.image.shape}, FPS: {frame_data.fps:.1f}")
                
                # Показать кадр (опционально)
                # cv2.imshow("Capture", cv2.cvtColor(frame_data.image, cv2.COLOR_RGB2BGR))
                # if cv2.waitKey(1) == ord('q'):
                #     break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop_capture()
        cv2.destroyAllWindows()
