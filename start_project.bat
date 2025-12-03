@echo off
echo ========================================
echo Запуск ML Web App проекта на Julia
echo ========================================
echo.

echo 1. Настройка проекта...
julia --project=. setup.jl
echo.

echo 2. Обучение модели...
julia --project=. simple_model_fixed.jl
echo.

echo 3. Запуск API сервера...
echo    Сервер запускается на http://localhost:8080
echo    Для остановки нажмите Ctrl+C в этом окне
echo.
julia --project=. working_final_api.jl

pause