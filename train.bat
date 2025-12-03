@echo off
echo ========================================
echo Обучение ML модели
echo ========================================
echo.

if exist decision_tree_model.jls (
    echo ✅ Модель уже обучена
    echo Удалите decision_tree_model.jls для переобучения
) else (
    echo 🎯 Обучение новой модели...
    julia --project=. simple_model_fixed.jl
)

echo.
echo ========================================
pause