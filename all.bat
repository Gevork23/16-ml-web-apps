@echo off
echo ========================================
echo ПОЛНЫЙ ЗАПУСК ML ПРОЕКТА
echo ========================================
echo.

call install.bat
if errorlevel 1 exit /b 1

call train.bat
if errorlevel 1 exit /b 1

echo.
echo 🚀 Запуск сервера...
echo Откройте браузер: http://localhost:8080
echo Или протестируйте в другом окне: test.bat
echo.

call run.bat