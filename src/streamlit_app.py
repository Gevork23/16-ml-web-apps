import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Настройка пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка страницы
st.set_page_config(
    page_title="ML Predictor Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../models/model.pkl')
        st.success("✅ Модель успешно загружена!")
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# Инициализация состояния сессии
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Сайдбар
with st.sidebar:
    st.title("🎛️ Панель управления")
    st.markdown("---")
    
    st.subheader("Информация о модели")
    if st.session_state.model is not None:
        st.info("**Модель:** RandomForestClassifier")
        st.info(f"**Количество признаков:** 4")
    else:
        st.warning("Модель не загружена")
    
    st.markdown("---")
    
    st.subheader("Настройки предсказания")
    show_probabilities = st.checkbox("Показывать вероятности", value=True)
    show_3d = st.checkbox("3D визуализация", value=False)
    
    st.markdown("---")
    
    st.subheader("Примеры данных")
    example_data = {
        "Пример 1 (Класс 0)": [1.2, -0.5, 0.3, 2.1],
        "Пример 2 (Класс 1)": [-0.8, 1.5, -1.2, 0.7],
        "Пример 3 (Смешанный)": [0.5, 0.5, 0.5, 0.5]
    }
    
    selected_example = st.selectbox(
        "Выберите пример",
        list(example_data.keys())
    )
    
    if st.button("Загрузить пример"):
        example_features = example_data[selected_example]
        for i in range(4):
            st.session_state[f'feature_{i}'] = example_features[i]
    
    st.markdown("---")
    st.caption("© ML Predictor Dashboard v1.0")

# Основной контент
st.title("🤖 ML Predictor Dashboard")
st.markdown("Интерактивная панель для предсказания с использованием ML модели")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["📊 Предсказание", "📈 Визуализация", "ℹ️ Информация"])

with tab1:
    st.header("Ввод данных и предсказание")
    
    # Создаем колонки для ввода данных
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Введите значения признаков")
        
        # 4 поля для ввода с использованием session_state
        features = []
        for i in range(4):
            if f'feature_{i}' not in st.session_state:
                st.session_state[f'feature_{i}'] = 0.0
            
            feature = st.slider(
                f"Признак {i+1}",
                -3.0, 3.0,
                value=st.session_state[f'feature_{i}'],
                step=0.1,
                key=f"slider_{i}"
            )
            features.append(feature)
            st.session_state[f'feature_{i}'] = feature
        
        # Кнопки действий
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_btn = st.button("🎯 Сделать предсказание", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Сбросить", use_container_width=True):
                for i in range(4):
                    st.session_state[f'feature_{i}'] = 0.0
                st.rerun()
        with col_btn3:
            random_btn = st.button("🎲 Случайные данные", use_container_width=True)
            if random_btn:
                random_features = np.random.uniform(-2, 2, 4)
                for i in range(4):
                    st.session_state[f'feature_{i}'] = float(random_features[i])
                st.rerun()
    
    with col2:
        st.subheader("Визуализация признаков")
        
        # График значений признаков
        fig_bar = go.Figure(data=[
            go.Bar(
                x=[f'Признак {i+1}' for i in range(4)],
                y=features,
                marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
            )
        ])
        
        fig_bar.update_layout(
            title="Значения признаков",
            yaxis_title="Значение",
            height=300,
            margin=dict(t=50, b=20, l=40, r=20)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Радар-диаграмма
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=features,
            theta=[f'Признак {i+1}' for i in range(4)],
            fill='toself',
            line_color='#FF6B6B'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-3, 3])
            ),
            showlegend=False,
            height=300,
            margin=dict(t=50, b=20, l=40, r=20)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Обработка предсказания
    if predict_btn and st.session_state.model is not None:
        try:
            features_array = np.array(features).reshape(1, -1)
            
            # Предсказание
            prediction = st.session_state.model.predict(features_array)
            probabilities = st.session_state.model.predict_proba(features_array)
            
            # Результаты
            st.markdown("---")
            st.subheader("📊 Результаты предсказания")
            
            # Метрики
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric(
                    label="🎯 Предсказанный класс",
                    value=int(prediction[0]),
                    delta="Класс 1" if prediction[0] == 1 else "Класс 0"
                )
            
            with col_res2:
                st.metric(
                    label="📈 Уверенность модели",
                    value=f"{max(probabilities[0]):.1%}",
                    delta="Высокая" if max(probabilities[0]) > 0.7 else "Средняя" if max(probabilities[0]) > 0.5 else "Низкая"
                )
            
            with col_res3:
                st.metric(
                    label="🔢 Количество признаков",
                    value="4",
                    delta="Все признаки использованы"
                )
            
            # Визуализация вероятностей
            if show_probabilities:
                st.subheader("Вероятности классов")
                
                fig_prob = go.Figure(data=[
                    go.Bar(
                        x=['Класс 0', 'Класс 1'],
                        y=probabilities[0],
                        marker_color=['#636EFA', '#EF553B'],
                        text=[f'{p:.1%}' for p in probabilities[0]],
                        textposition='auto'
                    )
                ])
                
                fig_prob.update_layout(
                    yaxis_title="Вероятность",
                    yaxis_tickformat=".0%",
                    height=300
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
            
            # Детальная информация
            with st.expander("📋 Детали предсказания"):
                st.write("**Входные данные:**")
                st.json({f"Признак {i+1}": float(features[i]) for i in range(4)})
                
                st.write("**Вероятности:**")
                prob_df = pd.DataFrame({
                    'Класс': ['Класс 0', 'Класс 1'],
                    'Вероятность': [f'{probabilities[0][0]:.3%}', f'{probabilities[0][1]:.3%}'],
                    'Значение': [float(probabilities[0][0]), float(probabilities[0][1])]
                })
                st.dataframe(prob_df, hide_index=True)
                
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
    
    elif predict_btn:
        st.warning("⚠️ Модель не загружена. Сначала создайте модель (запустите create_model.py).")

with tab2:
    st.header("Визуализация данных")
    
    # Генерация случайных данных для визуализации
    np.random.seed(42)
    n_samples = 100
    X_viz = np.random.uniform(-2, 2, (n_samples, 4))
    
    if st.session_state.model is not None:
        y_viz = st.session_state.model.predict(X_viz)
        
        # 3D визуализация
        if show_3d:
            fig_3d = go.Figure(data=[
                go.Scatter3d(
                    x=X_viz[:, 0],
                    y=X_viz[:, 1],
                    z=X_viz[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=y_viz,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"Класс: {int(y)}" for y in y_viz]
                )
            ])
            
            fig_3d.update_layout(
                title="3D Визуализация предсказаний",
                scene=dict(
                    xaxis_title="Признак 1",
                    yaxis_title="Признак 2",
                    zaxis_title="Признак 3"
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            # 2D scatter plot
            fig_scatter = go.Figure()
            
            for class_label in [0, 1]:
                mask = y_viz == class_label
                fig_scatter.add_trace(go.Scatter(
                    x=X_viz[mask, 0],
                    y=X_viz[mask, 1],
                    mode='markers',
                    name=f'Класс {class_label}',
                    marker=dict(size=10)
                ))
            
            fig_scatter.update_layout(
                title="2D Визуализация предсказаний (Признак 1 vs Признак 2)",
                xaxis_title="Признак 1",
                yaxis_title="Признак 2",
                height=500
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Матрица корреляции
        st.subheader("Матрица корреляции признаков")
        
        df_viz = pd.DataFrame(X_viz, columns=[f'Признак {i+1}' for i in range(4)])
        df_viz['Класс'] = y_viz
        
        corr_matrix = df_viz.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig_corr.update_layout(
            title="Корреляционная матрица",
            height=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Загрузите модель для визуализации данных")

with tab3:
    st.header("Информация о проекте")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("О проекте")
        st.markdown("""
        Это демонстрационное ML-приложение с тремя интерфейсами:
        
        **1. Flask REST API** - для программистов и интеграций
        **2. Streamlit Dashboard** - для аналитиков и исследователей
        **3. Gradio Interface** - для быстрого прототипирования
        
        **Используемые технологии:**
        - Python 3.12
        - Scikit-learn для ML
        - Flask для REST API
        - Streamlit для дашборда
        - Gradio для интерактивного интерфейса
        - Plotly для визуализации
        """)
    
    with col_info2:
        st.subheader("Инструкция по использованию")
        st.markdown("""
        1. **Создайте модель:** запустите `python create_model.py`
        2. **Запустите Flask API:** `python src/flask_api.py`
        3. **Запустите Streamlit:** `streamlit run src/streamlit_app.py`
        4. **Запустите Gradio:** `python src/gradio_app.py`
        
        **API endpoints:**
        - `GET /` - документация
        - `GET /health` - проверка здоровья
        - `POST /predict` - предсказание для одного образца
        - `POST /batch_predict` - предсказание для нескольких образцов
        """)
    
    st.markdown("---")
    st.subheader("Примеры запросов к API")
    
    code_examples = '''
# Пример запроса к Flask API
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [1.2, -0.5, 0.3, 2.1]}'

# Пример запроса с несколькими образцами
curl -X POST http://localhost:5000/batch_predict \\
  -H "Content-Type: application/json" \\
  -d '{"samples": [[1.2, -0.5, 0.3, 2.1], [-0.8, 1.5, -1.2, 0.7]]}'
'''
    
    st.code(code_examples, language='bash')

# Футер
st.markdown("---")
st.caption("Разработано для учебного проекта | Все системы работают нормально ✅")