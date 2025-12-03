@echo off
echo ========================================
echo Тестирование ML API
echo ========================================
echo.

echo 1. Проверка здоровья сервера...
curl http://localhost:8080/health
if errorlevel 1 (
    echo ❌ Сервер не отвечает!
    echo Запустите сервер: run.bat
    pause
    exit /b 1
)

echo.
echo 2. Тест предсказания...
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"features\": [0.5, 0.3, 0.8]}"

echo.
echo 3. Дополнительные тесты...
echo.
echo Тест 1: [0.1, 0.2, 0.3]
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"features\": [0.1, 0.2, 0.3]}"

echo.
echo Тест 2: [0.9, 0.8, 0.7]
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"features\": [0.9, 0.8, 0.7]}"

echo.
echo ========================================
echo ✅ Тестирование завершено!
echo ========================================
pause