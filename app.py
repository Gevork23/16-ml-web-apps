"""
Python аналог нашего Julia проекта для сравнения
"""
from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Создаем и обучаем модель
def train_model():
    X, y = make_classification(n_samples=100, n_features=3, random_state=42)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Сохраняем модель
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, X, y

# Загружаем или обучаем модель
if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Модель загружена из model.pkl")
else:
    print("��� Обучаем новую модель...")
    model, X, y = train_model()
    accuracy = model.score(X, y)
    print(f"✅ Модель обучена с точностью: {accuracy:.1%}")

@app.route('/')
def index():
    return '''
    <h1>ML API на Python (Flask)</h1>
    <p>Аналог Julia проекта</p>
    <p>Отправьте POST на /predict с JSON {"features": [0.5, 0.3, 0.8]}</p>
    '''

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "framework": "Flask",
        "model": "DecisionTree",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "features": data['features'],
            "timestamp": datetime.now().isoformat(),
            "framework": "Python/Flask"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    print("��� Запуск Flask сервера на http://localhost:5000")
    app.run(debug=True, port=5000)
