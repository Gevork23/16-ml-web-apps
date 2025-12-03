@echo off
echo ========================================
echo Установка зависимостей ML проекта
echo ========================================
echo.

echo 1. Проверка Julia...
julia --version
if errorlevel 1 (
    echo ❌ Julia не установлена!
    echo Скачайте с: https://julialang.org/downloads/
    pause
    exit /b 1
)

echo.
echo 2. Установка пакетов Julia...
julia --project=. -e "using Pkg; Pkg.instantiate(); println('✅ Пакеты установлены')"
if errorlevel 1 (
    echo ❌ Ошибка установки пакетов
    pause
    exit /b 1
)

echo.
echo 3. Запуск setup...
julia --project=. setup.jl

echo.
echo ========================================
echo ✅ Установка завершена!
echo ========================================
pause