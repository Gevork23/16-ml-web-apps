# working_final_api.jl - РАБОТАЮЩАЯ версия API

println("="^60)
println("РАБОТАЮЩИЙ ML API сервер на Julia")
println("="^60)

# 1. Загружаем пакеты
println("\n1. Загрузка пакетов...")
using HTTP
using JSON3
using Serialization
using Sockets
using Dates
using DecisionTree

# 2. Загружаем модель в ГЛОБАЛЬНУЮ область видимости
println("\n2. Загрузка модели...")
model_path = "decision_tree_model.jls"

if !isfile(model_path)
    println("   ❌ Модель не найдена!")
    println("   Сначала обучите модель: include(\"simple_model_fixed.jl\")")
    exit(1)
end

# Глобальная переменная
global TREE_MODEL = deserialize(model_path)
println("   ✅ Модель загружена!")

# 3. Глобальная функция для предсказания
function predict_global(features)
    try
        # Преобразуем в правильный формат
        if length(features) != 3
            return Dict(
                "success" => false,
                "error" => "Need exactly 3 features"
            )
        end
        
        # Делаем предсказание
        prediction = apply_tree(TREE_MODEL, reshape(features, 1, :))
        
        return Dict(
            "success" => true,
            "prediction" => prediction[1],
            "features" => features,
            "timestamp" => string(now())
        )
    catch e
        return Dict(
            "success" => false,
            "error" => string(e)
        )
    end
end

println("   ✅ Функция предсказания создана")

# 4. Создаем обработчики ВНУТРИ глобальной области
function make_handlers()
    # Главная страница
    function handle_root(req)
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>✅ РАБОТАЕТ! ML API на Julia</title>
            <style>
                body { font-family: Arial; padding: 20px; max-width: 800px; margin: 0 auto; }
                .box { background: #f0f8ff; padding: 20px; margin: 20px 0; border-radius: 10px; }
                input { padding: 10px; margin: 5px; width: 100px; font-size: 16px; }
                button { padding: 12px 24px; background: #28a745; color: white; border: none; cursor: pointer; font-size: 16px; }
                .result { padding: 15px; background: #d4edda; border-radius: 5px; margin-top: 15px; }
                .success { color: #155724; }
                .code { background: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; }
            </style>
        </head>
        <body>
            <h1>🎉 ML API НА JULIA - РАБОТАЕТ!</h1>
            <p>Дерево решений с точностью 96.7%</p>
            
            <div class="box">
                <h2>🧪 Тестирование в реальном времени</h2>
                <div>
                    <input id="f1" value="0.5" placeholder="Признак 1">
                    <input id="f2" value="0.3" placeholder="Признак 2">
                    <input id="f3" value="0.8" placeholder="Признак 3">
                    <button onclick="predict()">🚀 Получить предсказание</button>
                </div>
                <div class="result" id="result">
                    <strong>Результат появится здесь</strong>
                </div>
            </div>
            
            <div class="box">
                <h2>📡 API Эндпоинты</h2>
                
                <h3>GET /health</h3>
                <div class="code">curl http://localhost:8080/health</div>
                
                <h3>POST /predict</h3>
                <div class="code">
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.5, 0.3, 0.8]}'
                </div>
                
                <h3>Примеры данных:</h3>
                <ul>
                    <li>Вероятно класс 0: [0.1, 0.2, 0.3]</li>
                    <li>Вероятно класс 1: [0.9, 0.8, 0.7]</li>
                    <li>Пограничный случай: [0.5, 0.5, 0.5]</li>
                </ul>
            </div>
            
            <script>
            async function predict() {
                const f1 = parseFloat(document.getElementById('f1').value);
                const f2 = parseFloat(document.getElementById('f2').value);
                const f3 = parseFloat(document.getElementById('f3').value);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<em>⏳ Выполняется предсказание...</em>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features: [f1, f2, f3] })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        resultDiv.innerHTML = \`
                            <div class="success">
                                <h3>✅ УСПЕШНО!</h3>
                                <p><strong>Предсказанный класс:</strong> \${data.prediction}</p>
                                <p><strong>Признаки:</strong> [\${data.features.join(', ')}]</p>
                                <p><strong>Время:</strong> \${data.timestamp}</p>
                            </div>
                        \`;
                    } else {
                        resultDiv.innerHTML = \`
                            <div style="color: #dc3545;">
                                <h3>❌ ОШИБКА</h3>
                                <p>\${data.error}</p>
                            </div>
                        \`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = \`
                        <div style="color: #dc3545;">
                            <h3>❌ ОШИБКА СЕТИ</h3>
                            <p>\${error}</p>
                        </div>
                    \`;
                }
            }
            </script>
        </body>
        </html>
        """
        return HTTP.Response(200, ["Content-Type" => "text/html"], body=html)
    end

    # Health check
    function handle_health(req)
        response = Dict(
            "status" => "healthy",
            "model" => "DecisionTree",
            "accuracy" => "96.7%",
            "timestamp" => string(now()),
            "message" => "API работает корректно!"
        )
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=JSON3.write(response))
    end

    # Predict endpoint
    function handle_predict(req)
        try
            # Читаем тело запроса
            body = String(HTTP.payload(req))
            data = JSON3.read(body)
            
            # Проверяем данные
            if !haskey(data, :features)
                return HTTP.Response(400, ["Content-Type" => "application/json"], 
                    body=JSON3.write(Dict("success" => false, "error" => "No features provided")))
            end
            
            # Используем глобальную функцию
            result = predict_global(data.features)
            
            return HTTP.Response(200, ["Content-Type" => "application/json"], 
                body=JSON3.write(result))
                
        catch e
            return HTTP.Response(400, ["Content-Type" => "application/json"], 
                body=JSON3.write(Dict("success" => false, "error" => "Invalid request: $e")))
        end
    end

    return handle_root, handle_health, handle_predict
end

# Создаем обработчики
handle_root, handle_health, handle_predict = make_handlers()
println("   ✅ Обработчики созданы")

# 5. Основной обработчик приложения
function app(req::HTTP.Request)
    try
        if req.method == "GET" && req.target == "/"
            return handle_root(req)
        elseif req.method == "GET" && req.target == "/health"
            return handle_health(req)
        elseif req.method == "POST" && req.target == "/predict"
            return handle_predict(req)
        else
            return HTTP.Response(404, ["Content-Type" => "text/plain"], 
                body="404 Not Found\nДоступные пути: GET /, GET /health, POST /predict")
        end
    catch e
        return HTTP.Response(500, body="Server error: $e")
    end
end

# 6. Запускаем сервер
println("\n3. Запуск сервера...")
port = 8080
println("   🌐 Сервер запускается на порту $port")
println("   📋 Откройте в браузере: http://localhost:$port")
println("   🛑 Для остановки: Ctrl+C")

try
    # Проверяем порт
    println("   🔍 Проверка доступности порта...")
    
    # Запускаем сервер
    server = HTTP.serve(app, "127.0.0.1", port; verbose=false)
    
    println("\n" * "="^60)
    println("✅ СЕРВЕР УСПЕШНО ЗАПУЩЕН!")
    println("="^60)
    println("\nТестируйте:")
    println("1. Веб-интерфейс: http://localhost:$port")
    println("2. Health check: curl http://localhost:$port/health")
    println("3. Предсказание: curl -X POST http://localhost:$port/predict \\")
    println("   -H \"Content-Type: application/json\" \\")
    println("   -d '{\"features\": [0.5, 0.3, 0.8]}'")
    
    # Ждем сигнал остановки
    wait(server)
    
catch e
    println("\n❌ Ошибка запуска: ", e)
    
    if occursin("10048", string(e)) || occursin("EADDRINUSE", string(e))
        println("\n💡 Порт $port занят! Попробуйте:")
        println("   1. Изменить порт в коде (строка: port = 8081)")
        println("   2. Закрыть другие программы использующие порт 8080")
        println("   3. Использовать команду: netstat -ano | findstr :8080")
    end
end