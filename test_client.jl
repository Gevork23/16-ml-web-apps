# test_client.jl - Тестирование API

println("="^50)
println("Тестирование ML API")
println("="^50)

using HTTP
using JSON3

const BASE_URL = "http://127.0.0.1:8080"

function test_server()
    println("\n1. Проверка сервера...")
    
    # Тест 1: Проверка здоровья
    try
        response = HTTP.get("$BASE_URL/health")
        if response.status == 200
            data = JSON3.read(String(response.body))
            println("   ✅ Сервер работает:")
            for (key, value) in data
                println("      $key: $value")
            end
        else
            println("   ❌ Сервер вернул статус: $(response.status)")
        end
    catch e
        println("   ❌ Не удалось подключиться к серверу: ", e)
        println("   Убедитесь что сервер запущен: julia working_api.jl")
        return false
    end
    
    # Тест 2: Предсказание через POST
    println("\n2. Тест предсказания через POST...")
    test_cases = [
        [0.1, 0.2, 0.3],
        [0.7, 0.8, 0.9],
        [0.5, 0.5, 0.5]
    ]
    
    for (i, features) in enumerate(test_cases)
        println("   Тест $i: features = $features")
        try
            request_body = JSON3.write(Dict("features" => features))
            response = HTTP.post(
                "$BASE_URL/predict",
                ["Content-Type" => "application/json"],
                request_body
            )
            
            if response.status == 200
                data = JSON3.read(String(response.body))
                if get(data, :success, false)
                    println("      ✅ Класс: $(data[:prediction])")
                                    else
                    println("      ❌ Ошибка: $(data[:error])")
                end
            else
                println("      ❌ HTTP ошибка: $(response.status)")
            end
        catch e
            println("      ❌ Ошибка запроса: ", e)
        end
    end
    
    # Тест 3: Предсказание через GET
    println("\n3. Тест предсказания через GET...")
    try
        url = "$BASE_URL/predict?f1=0.6&f2=0.4&f3=0.7"
        response = HTTP.get(url)
        
        if response.status == 200
            data = JSON3.read(String(response.body))
            println("   ✅ GET запрос успешен:")
            println("      Предсказание: $(data[:prediction])")
                    else
            println("   ❌ GET запрос не удался: $(response.status)")
        end
    catch e
        println("   ❌ Ошибка GET запроса: ", e)
    end
    
    println("\n" * "="^50)
    println("✅ Тестирование завершено!")
    println("="^50)
    
    return true
end

# Запускаем тесты
if abspath(PROGRAM_FILE) == @__FILE__
    test_server()
    else
    println("Для запуска тестов выполните: include(\"test_client.jl\")")
end
