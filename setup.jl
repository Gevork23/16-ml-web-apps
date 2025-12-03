# setup.jl - Быстрая установка проекта

println("="^60)
println("Настройка проекта ML Web App на Julia")
println("="^60)

using Pkg

println("\n1. Активация проекта...")
Pkg.activate(".")

println("\n2. Установка зависимостей...")
Pkg.add([
    "DecisionTree",
    "HTTP", 
    "JSON3",
    "Serialization",
    "Dates",
    "Sockets"
])

println("\n3. Предкомпиляция пакетов...")
Pkg.precompile()

println("\n" * "="^60)
println("✅ Настройка завершена!")
println("\nСледующие шаги:")
println("1. Обучите модель: julia --project=. simple_model_fixed.jl")
println("2. Запустите API: julia --project=. working_final_api.jl")
println("3. Откройте в браузере: http://localhost:8080")
println("="^60)
