import gradio as gr
import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Загрузка модели
try:
    model = joblib.load('../models/model.pkl')
    print("✅ Модель загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None

def predict(f1, f2, f3, f4):
    if model is None:
        return "Модель не загружена", {}
    
    try:
        features = np.array([[f1, f2, f3, f4]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        result = {
            "prediction": int(prediction),
            "probabilities": {
                "class_0": float(probability[0]),
                "class_1": float(probability[1])
            },
            "features": [f1, f2, f3, f4]
        }
        
        text_result = f"🎯 Класс: {prediction}\n"
        text_result += f"📊 Вероятности: Класс 0={probability[0]:.1%}, Класс 1={probability[1]:.1%}"
        
        return text_result, result
    
    except Exception as e:
        return f"Ошибка: {str(e)}", {}

# Простой интерфейс
with gr.Blocks(title="🤖 Простой ML Predictor") as demo:
    gr.Markdown("# 🎯 Простой ML Predictator")
    
    with gr.Row():
        f1 = gr.Slider(-3, 3, value=0, label="Признак 1", step=0.1)
        f2 = gr.Slider(-3, 3, value=0, label="Признак 2", step=0.1)
        f3 = gr.Slider(-3, 3, value=0, label="Признак 3", step=0.1)
        f4 = gr.Slider(-3, 3, value=0, label="Признак 4", step=0.1)
    
    btn = gr.Button("Предсказать", variant="primary")
    
    with gr.Row():
        output_text = gr.Textbox(label="Результат", lines=3)
        output_json = gr.JSON(label="JSON ответ")
    
    btn.click(predict, inputs=[f1, f2, f3, f4], outputs=[output_text, output_json])

if __name__ == "__main__":
    print("🚀 Запуск Gradio на http://localhost:7860")
    demo.launch(server_port=7860)