# simple_model_fixed.jl - Упрощенная рабочая версия

println("="^50)
println("Упрощенная ML модель на Julia")
println("="^50)

# Используем DecisionTree напрямую, без MLJ
using DecisionTree
using Random
using Serialization

# 1. Создаем синтетические данные
println("\n1. Создание данных...")
Random.seed!(42)

n_samples = 100
X = rand(n_samples, 3)  # 3 признака
y = Int.(X[:, 1] .+ X[:, 2] * 2 .> 1.2)

println("   Создано $n_samples примеров")
println("   Размер X: $(size(X))")
println("   Пример X[1,:]: $(round.(X[1,:], digits=2))")
println("   Пример y[1]: $(y[1])")
println("   Баланс классов: $(sum(y)) положительных из $(length(y))")

# 2. Разделяем данные
println("\n2. Разделение данных...")
train_ratio = 0.7
n_train = Int(floor(train_ratio * n_samples))

X_train = X[1:n_train, :]
y_train = y[1:n_train]
X_test = X[n_train+1:end, :]
y_test = y[n_train+1:end]

println("   Обучающая выборка: $(size(X_train)) примеров")
println("   Тестовая выборка: $(size(X_test)) примеров")

# 3. Обучаем модель (ПРОСТОЙ СПОСОБ!)
println("\n3. Обучение модели DecisionTree...")

# Используем DecisionTree.jl напрямую
model = build_tree(y_train, X_train)
println("   ✅ Модель обучена!")

# 4. Предсказания
println("\n4. Предсказания...")
y_pred = apply_tree(model, X_test)

println("   Пример предсказаний:")
println("   Фактические: $(y_test[1:5])")
println("   Предсказанные: $(y_pred[1:5])")

# 5. Оценка точности
println("\n5. Оценка модели...")
accuracy = sum(y_test .== y_pred) / length(y_test)
println("   Точность: $(round(accuracy * 100, digits=1))%")

# Матрица ошибок
println("\n   Матрица ошибок:")
for true_class in 0:1
    for pred_class in 0:1
        count = sum((y_test .== true_class) .& (y_pred .== pred_class))
        print("   True=$true_class → Pred=$pred_class: $count\t")
    end
    println()
end

# 6. Сохраняем модель
println("\n6. Сохранение модели...")
serialize("decision_tree_model.jls", model)
println("   ✅ Модель сохранена как decision_tree_model.jls")

# 7. Пример использования
println("\n7. Пример предсказания для новых данных:")
new_sample = [0.6, 0.4, 0.3]  # Новые данные
prediction = apply_tree(model, reshape(new_sample, 1, :))
println("   Входные данные: $new_sample")
println("   Предсказанный класс: $prediction")

# 8. Создаем функцию для API
println("\n8. Создание функции для API...")
function predict_tree(features::Vector{Float64})
    # Загружаем модель если нужно
    if !@isdefined(model)
        if isfile("decision_tree_model.jls")
            model = deserialize("decision_tree_model.jls")
        else
            error("Модель не найдена!")
        end
    end
    
    # Делаем предсказание
    pred = apply_tree(model, reshape(features, 1, :))
    return Dict(
        "features" => features,
        "prediction" => pred[1],
        "confidence" => 0.85  # Для простоты
    )
end

# Тестируем функцию
println("\n   Тест функции predict_tree:")
test_input = [0.8, 0.2, 0.5]
result = predict_tree(test_input)
println("   Вход: $test_input")
println("   Результат: $result")

println("\n" * "="^50)
println("✅ Упрощенная модель готова!")
println("="^50)
