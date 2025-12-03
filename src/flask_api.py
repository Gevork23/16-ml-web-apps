from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sys
import os

# Добавляем путь к корневой директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех доменов

# Загрузка модели
try:
    model = joblib.load('../models/model.pkl')
    print("✅ Модель успешно загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({
        "message": "ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Эта страница",
            "GET /health": "Проверка работоспособности",
            "POST /predict": "Предсказание по данным",
            "POST /batch_predict": "Предсказание для нескольких образцов"
        },
        "model_info": {
            "loaded": model is not None,
            "type": "RandomForestClassifier" if model else None,
            "n_features": 4 if model else None
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Модель не загружена"}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Нет данных в запросе"}), 400
        
        # Поддержка разных форматов входных данных
        if 'features' in data:
            features = data['features']
        elif 'data' in data:
            features = data['data']
        else:
            return jsonify({
                "error": "Неверный формат данных. Ожидается JSON с ключом 'features' или 'data'"
            }), 400
        
        # Преобразуем в numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Проверяем размерность
        if features_array.shape[1] != 4:
            return jsonify({
                "error": f"Ожидается 4 признака, получено {features_array.shape[1]}. Пример: [1.2, -0.5, 0.3, 2.1]"
            }), 400
        
        # Делаем предсказание
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": {
                "class_0": float(probability[0][0]),
                "class_1": float(probability[0][1])
            },
            "features": features_array[0].tolist(),
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({"error": "Модель не загружена"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({"error": "Нет данных или ключа 'samples' в запросе"}), 400
        
        samples = np.array(data['samples'])
        
        if len(samples.shape) != 2 or samples.shape[1] != 4:
            return jsonify({
                "error": f"Ожидается массив размером [n, 4], получено {samples.shape}"
            }), 400
        
        # Делаем предсказания для всех образцов
        predictions = model.predict(samples)
        probabilities = model.predict_proba(samples)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "sample_id": i,
                "prediction": int(pred),
                "probability": {
                    "class_0": float(prob[0]),
                    "class_1": float(prob[1])
                },
                "features": samples[i].tolist()
            })
        
        return jsonify({
            "results": results,
            "count": len(results),
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400

if __name__ == '__main__':
    print("🚀 Запуск Flask API...")
    print("📡 Адрес: http://localhost:5000")
    print("📊 Адрес с документацией: http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)