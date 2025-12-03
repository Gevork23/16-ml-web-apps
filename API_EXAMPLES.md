# Примеры использования API

## Базовый URL
http://localhost:8080

## 1. Проверка здоровья сервера

### Запрос:
```bash
curl http://localhost:8080/health
Ответ:
{
  "status": "healthy",
  "model": "DecisionTree",
  "accuracy": "96.7%",
  "timestamp": "2025-12-03T20:15:30.123"
}
2. Предсказание класса
Запрос:
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.5, 0.3, 0.8]}'
Ответ:
{
  "success": true,
  "prediction": 0,
  "features": [0.5, 0.3, 0.8],
  "timestamp": "2025-12-03T20:15:35.456"
}
Несколько примеров данных
Пример 1 (вероятно класс 0):
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.1, 0.2, 0.3]}'
Пример 2 (вероятно класс 1):
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.9, 0.8, 0.7]}'
Пример 3 (пограничный случай):
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.5, 0.5, 0.5]}'
4. Веб-интерфейс
Откройте в браузере:
http://localhost:8080
5. Python клиент
import requests
import json

# Базовый URL
BASE_URL = "http://localhost:8080"

# 1. Проверка здоровья
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. Предсказание
data = {"features": [0.5, 0.3, 0.8]}
response = requests.post(
    f"{BASE_URL}/predict",
    json=data,
    headers={"Content-Type": "application/json"}
)
print("Prediction:", response.json())
6. JavaScript клиент (для веб-приложений)
// Проверка здоровья
async function checkHealth() {
    const response = await fetch('http://localhost:8080/health');
    const data = await response.json();
    console.log('Server health:', data);
}

// Предсказание
async function predict(features) {
    const response = await fetch('http://localhost:8080/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: features })
    });
    return await response.json();
}

// Использование
predict([0.5, 0.3, 0.8]).then(result => {
    console.log('Prediction result:', result);
});
Ошибки API
400 Bad Request - Неверный формат данных
{
  "success": false,
  "error": "Missing 'features' field"
}
400 Bad Request - Неверное количество признаков
{
  "success": false, 
  "error": "Expected 3 features, got 2"
}
404 Not Found - Неверный URL
404 Not Found
Доступные пути: GET /, GET /health, POST /predict
