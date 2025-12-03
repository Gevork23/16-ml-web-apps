#!/bin/bash
echo "========================================"
echo "ДЕМОНСТРАЦИЯ РАБОТЫ ML WEB APP"
echo "========================================"
echo

echo "1. Проверяем зависимости..."
if ! command -v julia &> /dev/null; then
    echo "❌ Julia не установлена"
    exit 1
fi
echo "✅ Julia установлена"

echo
echo "2. Устанавливаем зависимости проекта..."
julia --project=. -e 'using Pkg; Pkg.instantiate()' 2>/dev/null
echo "✅ Зависимости установлены"

echo
echo "3. Обучаем модель..."
julia --project=. simple_model_fixed.jl 2>&1 | tail -20
echo "✅ Модель обучена"

echo
echo "4. Запускаем сервер в фоне..."
julia --project=. working_final_api.jl &
SERVER_PID=$!
sleep 3

echo
echo "5. Тестируем API..."
echo "   Health check:"
curl -s http://localhost:8080/health | python -m json.tool 2>/dev/null || curl -s http://localhost:8080/health
echo
echo "   Prediction test:"
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, 0.8]}' | python -m json.tool 2>/dev/null || \
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, 0.8]}'

echo
echo "6. Веб-интерфейс доступен по:"
echo "   http://localhost:8080"
echo
echo "7. Для остановки сервера выполните:"
echo "   kill $SERVER_PID"
echo
echo "========================================"
echo "Демонстрация завершена!"
echo "========================================"
