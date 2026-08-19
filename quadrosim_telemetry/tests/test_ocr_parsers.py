"""
Юнит-тесты для парсеров OCR
Тестирует корректность распознавания различных форматов данных
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr import FieldParser


class TestFieldParser:
    """Тесты для парсеров полей HUD"""
    
    def test_parse_time_mm_ss_ms(self):
        """Парсинг времени в формате ММ:СС.mmm"""
        assert FieldParser.parse_time("1:23.456") == 83.456
        assert FieldParser.parse_time("0:45.123") == 45.123
        assert FieldParser.parse_time("10:00.000") == 600.0
        print("✓ test_parse_time_mm_ss_ms passed")
    
    def test_parse_time_mm_ss(self):
        """Парсинг времени в формате ММ:СС"""
        assert FieldParser.parse_time("1:23") == 83.0
        assert FieldParser.parse_time("0:45") == 45.0
        assert FieldParser.parse_time("10:00") == 600.0
        print("✓ test_parse_time_mm_ss passed")
    
    def test_parse_time_ss_ms(self):
        """Парсинг времени в формате СС.mmm"""
        assert FieldParser.parse_time("23.456") == 23.456
        assert FieldParser.parse_time("5.1") == 5.1
        print("✓ test_parse_time_ss_ms passed")
    
    def test_parse_time_invalid(self):
        """Парсинг некорректного времени"""
        assert FieldParser.parse_time("") is None
        assert FieldParser.parse_time("abc") is None
        assert FieldParser.parse_time("1:2:3") is None
        print("✓ test_parse_time_invalid passed")
    
    def test_parse_number_integer(self):
        """Парсинг целых чисел"""
        assert FieldParser.parse_number("123") == 123.0
        assert FieldParser.parse_number("0") == 0.0
        assert FieldParser.parse_number("999") == 999.0
        print("✓ test_parse_number_integer passed")
    
    def test_parse_number_decimal_dot(self):
        """Парсинг чисел с точкой"""
        assert FieldParser.parse_number("123.45") == 123.45
        assert FieldParser.parse_number("0.5") == 0.5
        print("✓ test_parse_number_decimal_dot passed")
    
    def test_parse_number_decimal_comma(self):
        """Парсинг чисел с запятой"""
        assert FieldParser.parse_number("123,45") == 123.45
        assert FieldParser.parse_number("0,5") == 0.5
        print("✓ test_parse_number_decimal_comma_passed")
    
    def test_parse_voltage(self):
        """Парсинг напряжения батареи"""
        assert FieldParser.parse_voltage("22,2V") == 22.2
        assert FieldParser.parse_voltage("25.5V") == 25.5
        assert FieldParser.parse_voltage("18,8") == 18.8
        print("✓ test_parse_voltage passed")
    
    def test_parse_current(self):
        """Парсинг тока батареи"""
        assert FieldParser.parse_current("15,5A") == 15.5
        assert FieldParser.parse_current("8.2A") == 8.2
        assert FieldParser.parse_current("20,0") == 20.0
        print("✓ test_parse_current passed")
    
    def test_parse_laps(self):
        """Парсинг информации о кругах"""
        assert FieldParser.parse_laps("2 / 5") == (2, 5)
        assert FieldParser.parse_laps("1/10") == (1, 10)
        assert FieldParser.parse_laps("5 /  3") == (5, 3)
        assert FieldParser.parse_laps("7") == (7, None)
        print("✓ test_parse_laps passed")
    
    def test_parse_speed(self):
        """Парсинг скорости"""
        assert FieldParser.parse_speed("123 КМ/Ч") == 123.0
        assert FieldParser.parse_speed("85 км/ч") == 85.0
        assert FieldParser.parse_speed("45") == 45.0
        print("✓ test_parse_speed passed")
    
    def test_parse_altitude(self):
        """Парсинг высоты"""
        assert FieldParser.parse_altitude("50 М") == 50.0
        assert FieldParser.parse_altitude("120 м") == 120.0
        assert FieldParser.parse_altitude("75 M") == 75.0
        assert FieldParser.parse_altitude("30") == 30.0
        print("✓ test_parse_altitude passed")
    
    def run_all(self):
        """Запустить все тесты"""
        print("\n=== Running OCR Parser Tests ===\n")
        
        self.test_parse_time_mm_ss_ms()
        self.test_parse_time_mm_ss()
        self.test_parse_time_ss_ms()
        self.test_parse_time_invalid()
        self.test_parse_number_integer()
        self.test_parse_number_decimal_dot()
        self.test_parse_number_decimal_comma()
        self.test_parse_voltage()
        self.test_parse_current()
        self.test_parse_laps()
        self.test_parse_speed()
        self.test_parse_altitude()
        
        print("\n=== All tests passed! ===\n")


if __name__ == "__main__":
    tester = TestFieldParser()
    tester.run_all()
