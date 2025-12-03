@echo off
echo ========================================
echo Запуск ML API сервера
echo ========================================
echo.

echo 🚀 Сервер запускается...
echo 📍 Адрес: http://localhost:8080
echo 🛑 Для остановки: Ctrl+C
echo.

if not exist decision_tree_model.jls (
    echo ⚠️  Модель не найдена!
    echo Сначала выполните: train.bat
    pause
    exit /b 1
)

julia --project=. working_final_api.jl

pause