import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os

# Создаем тестовые данные
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=42,
    n_clusters_per_class=1
)

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сохраняем модель
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')

# Сохраняем пример данных
sample_data = {
    'features': X_test[:5].tolist(),
    'labels': y_test[:5].tolist()
}

print("="*50)
print("✅ Модель успешно создана и сохранена в models/model.pkl")
print(f"📊 Точность на тестовых данных: {model.score(X_test, y_test):.2%}")
print(f"📦 Размер обучающей выборки: {X_train.shape[0]} образцов")
print(f"🔍 Размер тестовой выборки: {X_test.shape[0]} образцов")
print(f"🎯 Классы: {np.unique(y)}")
print("\nПример данных для тестирования API:")
print(f"  Признаки: {X_test[0].tolist()}")
print(f"  Метка: {y_test[0]}")
print("="*50)