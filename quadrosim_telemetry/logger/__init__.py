"""
Модуль логгирования телеметрии в SQLite
Записывает данные с высокой частотой, поддерживает пакетную запись
"""
import sqlite3
import time
import threading
import queue
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class TelemetryFrame:
    """Кадр телеметрии со всеми данными HUD и стиков"""
    timestamp: float  # performance_counter
    frame_id: int
    
    # Поля HUD
    pilot_info: Optional[str] = None
    datetime_str: Optional[str] = None
    battery_voltage: Optional[float] = None
    battery_current: Optional[float] = None
    flight_mode: Optional[str] = None
    time_limit: Optional[float] = None
    speed: Optional[float] = None
    altitude: Optional[float] = None
    laps_current: Optional[int] = None
    laps_total: Optional[int] = None
    current_time: Optional[float] = None
    best_time: Optional[float] = None
    
    # Стики (4 оси)
    stick_left_x: Optional[float] = None  # Руль
    stick_left_y: Optional[float] = None  # Газ
    stick_right_x: Optional[float] = None  # Крен
    stick_right_y: Optional[float] = None  # Тангаж
    
    # Метаданные
    flight_id: Optional[int] = None  # ID текущего вылета
    is_active: bool = True  # Активен ли полёт


class TelemetryLogger:
    """
    Логгер телеметрии с пакетной записью в SQLite
    
    Архитектура:
    - Основной поток добавляет кадры в очередь
    - Фоновый поток пакетно записывает в БД
    """
    
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS telemetry_frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        frame_id INTEGER NOT NULL,
        flight_id INTEGER,
        
        -- HUD fields
        pilot_info TEXT,
        datetime_str TEXT,
        battery_voltage REAL,
        battery_current REAL,
        flight_mode TEXT,
        time_limit REAL,
        speed REAL,
        altitude REAL,
        laps_current INTEGER,
        laps_total INTEGER,
        current_time REAL,
        best_time REAL,
        
        -- Stick axes
        stick_left_x REAL,
        stick_left_y REAL,
        stick_right_x REAL,
        stick_right_y REAL,
        
        -- Meta
        is_active INTEGER DEFAULT 1,
        
        -- Indexes for fast queries
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_timestamp ON telemetry_frames(timestamp);
    CREATE INDEX IF NOT EXISTS idx_flight_id ON telemetry_frames(flight_id);
    CREATE INDEX IF NOT EXISTS idx_frame_id ON telemetry_frames(frame_id);
    """
    
    CREATE_FLIGHTS_TABLE = """
    CREATE TABLE IF NOT EXISTS flights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time REAL NOT NULL,
        end_time REAL,
        duration REAL,
        lap_count INTEGER DEFAULT 0,
        best_lap_time REAL,
        status TEXT DEFAULT 'active',
        notes TEXT
    );
    """
    
    def __init__(self, 
                 db_path: str = "data/telemetry.db",
                 batch_size: int = 100,
                 flush_interval: float = 1.0):
        """
        Args:
            db_path: Путь к SQLite базе данных
            batch_size: Размер пакета для записи
            flush_interval: Интервал принудительной записи (сек)
        """
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Очередь кадров
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        # Буфер для пакетной записи
        self.write_buffer: List[TelemetryFrame] = []
        self.buffer_lock = threading.Lock()
        
        # Состояние
        self.is_running = False
        self.current_flight_id: Optional[int] = None
        
        # Статистика
        self.frames_written = 0
        self.frames_dropped = 0
        
        # Инициализация БД
        self._init_database()
        
        # Запуск фонового потока записи
        self._start_writer_thread()
    
    def _init_database(self):
        """Инициализировать базу данных"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Разделяем SQL на отдельные statements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                frame_id INTEGER NOT NULL,
                flight_id INTEGER,
                
                -- HUD fields
                pilot_info TEXT,
                datetime_str TEXT,
                battery_voltage REAL,
                battery_current REAL,
                flight_mode TEXT,
                time_limit REAL,
                speed REAL,
                altitude REAL,
                laps_current INTEGER,
                laps_total INTEGER,
                current_time REAL,
                best_time REAL,
                
                -- Stick axes
                stick_left_x REAL,
                stick_left_y REAL,
                stick_right_x REAL,
                stick_right_y REAL,
                
                -- Meta
                is_active INTEGER DEFAULT 1,
                
                -- Indexes for fast queries
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON telemetry_frames(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flight_id ON telemetry_frames(flight_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_id ON telemetry_frames(frame_id)")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL NOT NULL,
                end_time REAL,
                duration REAL,
                lap_count INTEGER DEFAULT 0,
                best_lap_time REAL,
                status TEXT DEFAULT 'active',
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _start_writer_thread(self):
        """Запустить фоновый поток записи"""
        self.is_running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        """Фоновый цикл записи в БД"""
        last_flush = time.perf_counter()
        
        while self.is_running:
            # Получаем кадр из очереди
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                with self.buffer_lock:
                    self.write_buffer.append(frame)
                    
                    # Пакетная запись если набралось достаточно кадров
                    if len(self.write_buffer) >= self.batch_size:
                        self._flush_buffer()
                        last_flush = time.perf_counter()
                        
            except queue.Empty:
                pass
            
            # Принудительная запись по таймеру
            current_time = time.perf_counter()
            if current_time - last_flush > self.flush_interval:
                with self.buffer_lock:
                    if self.write_buffer:
                        self._flush_buffer()
                last_flush = current_time
    
    def _flush_buffer(self):
        """Сбросить буфер в БД"""
        if not self.write_buffer:
            return
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        insert_sql = """
        INSERT INTO telemetry_frames 
        (timestamp, frame_id, flight_id,
         pilot_info, datetime_str, battery_voltage, battery_current,
         flight_mode, time_limit, speed, altitude,
         laps_current, laps_total, current_time, best_time,
         stick_left_x, stick_left_y, stick_right_x, stick_right_y,
         is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        data = [
            (
                f.timestamp, f.frame_id, f.flight_id,
                f.pilot_info, f.datetime_str, f.battery_voltage, f.battery_current,
                f.flight_mode, f.time_limit, f.speed, f.altitude,
                f.laps_current, f.laps_total, f.current_time, f.best_time,
                f.stick_left_x, f.stick_left_y, f.stick_right_x, f.stick_right_y,
                1 if f.is_active else 0
            )
            for f in self.write_buffer
        ]
        
        cursor.executemany(insert_sql, data)
        conn.commit()
        conn.close()
        
        self.frames_written += len(self.write_buffer)
        self.write_buffer.clear()
    
    def log_frame(self, frame: TelemetryFrame):
        """
        Добавить кадр телеметрии в очередь
        
        Args:
            frame: Кадр с данными
        """
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.frames_dropped += 1
    
    def log_dict(self, data: Dict[str, Any], frame_id: int, flight_id: Optional[int] = None):
        """
        Добавить кадр из словаря
        
        Args:
            data: Словарь с данными
            frame_id: ID кадра
            flight_id: ID вылета
        """
        frame = TelemetryFrame(
            timestamp=time.perf_counter(),
            frame_id=frame_id,
            flight_id=flight_id or self.current_flight_id,
            **{k: v for k, v in data.items() if k in TelemetryFrame.__dataclass_fields__}
        )
        self.log_frame(frame)
    
    def start_flight(self) -> int:
        """
        Начать новый вылет
        
        Returns:
            ID вылета
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO flights (start_time, status) VALUES (?, 'active')",
            (time.perf_counter(),)
        )
        
        self.current_flight_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Flight started: ID={self.current_flight_id}")
        return self.current_flight_id
    
    def end_flight(self, flight_id: Optional[int] = None, notes: str = ""):
        """
        Завершить вылет
        
        Args:
            flight_id: ID вылета (или текущий)
            notes: Заметки
        """
        fid = flight_id or self.current_flight_id
        if fid is None:
            return
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        end_time = time.perf_counter()
        
        # Обновляем статус вылета
        cursor.execute("""
            UPDATE flights 
            SET end_time = ?, duration = ?, status = 'completed', notes = ?
            WHERE id = ?
        """, (end_time, end_time - cursor.execute("SELECT start_time FROM flights WHERE id = ?", (fid,)).fetchone()[0], notes, fid))
        
        # Помечаем кадры как неактивные
        cursor.execute(
            "UPDATE telemetry_frames SET is_active = 0 WHERE flight_id = ?",
            (fid,)
        )
        
        conn.commit()
        conn.close()
        
        # Сбрасываем буфер
        with self.buffer_lock:
            self._flush_buffer()
        
        self.current_flight_id = None
        print(f"Flight ended: ID={fid}")
    
    def get_flight_data(self, flight_id: int) -> List[Dict[str, Any]]:
        """
        Получить данные вылета
        
        Args:
            flight_id: ID вылета
            
        Returns:
            Список словарей с данными кадров
        """
        # Сначала сбрасываем буфер
        with self.buffer_lock:
            self._flush_buffer()
        
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM telemetry_frames WHERE flight_id = ? ORDER BY timestamp",
            (flight_id,)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def stop(self):
        """Остановить логгер"""
        self.is_running = False
        
        if hasattr(self, 'writer_thread'):
            self.writer_thread.join(timeout=2.0)
        
        # Финальная запись буфера
        with self.buffer_lock:
            if self.write_buffer:
                self._flush_buffer()
        
        print(f"Logger stopped. Frames written: {self.frames_written}, dropped: {self.frames_dropped}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FlightDataExporter:
    """Экспорт данных вылета в различные форматы"""
    
    @staticmethod
    def to_pandas(flight_data: List[Dict]) -> 'pd.DataFrame':
        """Конвертировать в DataFrame"""
        import pandas as pd
        return pd.DataFrame(flight_data)
    
    @staticmethod
    def to_csv(flight_data: List[Dict], output_path: str):
        """Экспорт в CSV"""
        df = FlightDataExporter.to_pandas(flight_data)
        df.to_csv(output_path, index=False)
    
    @staticmethod
    def to_json(flight_data: List[Dict], output_path: str):
        """Экспорт в JSON"""
        with open(output_path, 'w') as f:
            json.dump(flight_data, f, indent=2)


if __name__ == "__main__":
    # Тест логгера
    print("Testing Telemetry Logger...")
    
    import os
    test_db = "data/test_telemetry.db"
    
    # Удаляем старый тестовый файл
    if os.path.exists(test_db):
        os.remove(test_db)
    
    with TelemetryLogger(db_path=test_db, batch_size=10) as logger:
        # Начинаем вылет
        flight_id = logger.start_flight()
        
        # Пишем тестовые данные
        for i in range(50):
            frame = TelemetryFrame(
                timestamp=time.perf_counter(),
                frame_id=i,
                flight_id=flight_id,
                speed=50 + i * 0.5,
                altitude=10 + i * 0.1,
                current_time=i * 0.1,
                stick_left_x=0.1 * (i % 10),
                stick_left_y=0.5 + 0.1 * (i % 5),
                stick_right_x=-0.2 * (i % 7),
                stick_right_y=0.3 * (i % 3)
            )
            logger.log_frame(frame)
            
            if i % 10 == 0:
                print(f"Logged frame {i}")
            
            time.sleep(0.01)
        
        # Завершаем вылет
        logger.end_flight(notes="Test flight")
    
    # Читаем данные обратно
    logger2 = TelemetryLogger(db_path=test_db)
    data = logger2.get_flight_data(flight_id)
    print(f"\nRetrieved {len(data)} frames from flight {flight_id}")
    
    if data:
        print(f"First frame: speed={data[0]['speed']}, altitude={data[0]['altitude']}")
        print(f"Last frame: speed={data[-1]['speed']}, altitude={data[-1]['altitude']}")
    
    logger2.stop()
    
    # Cleanup
    os.remove(test_db)
    print("\nTest completed!")
