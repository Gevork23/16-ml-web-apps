# Makefile для ML Web App проекта

.PHONY: help setup train run test clean compare

help:
	@echo "Доступные команды:"
	@echo "  make setup    - Установить зависимости Julia"
	@echo "  make train    - Обучить ML модель"
	@echo "  make run      - Запустить API сервер"
	@echo "  make test     - Протестировать API"
	@echo "  make compare  - Запустить Python аналог для сравнения"
	@echo "  make clean    - Очистить временные файлы"
	@echo "  make all      - Выполнить все шаги (setup → train → run)"

setup:
	@echo "��� Установка зависимостей Julia..."
	julia --project=. setup.jl

train:
	@echo "��� Обучение ML модели..."
	julia --project=. simple_model_fixed.jl

run:
	@echo "��� Запуск API сервера..."
	@echo "   Откройте: http://localhost:8080"
	@echo "   Остановка: Ctrl+C"
	julia --project=. working_final_api.jl

test:
	@echo "��� Тестирование API..."
	@if ! curl -s http://localhost:8080/health > /dev/null; then \
		echo "❌ Сервер не запущен. Сначала выполните: make run"; \
	else \
		echo "✅ Health check:"; \
		curl -s http://localhost:8080/health | python -m json.tool; \
		echo ""; \
		echo "✅ Тест предсказания:"; \
		curl -s -X POST http://localhost:8080/predict \
			-H "Content-Type: application/json" \
			-d '{"features": [0.5, 0.3, 0.8]}' | python -m json.tool; \
	fi

compare:
	@echo "��� Запуск Python аналога..."
	@echo "   Откройте: http://localhost:5000"
	@echo "   Остановка: Ctrl+C"
	python app.py

clean:
	@echo "��� Очистка временных файлов..."
	rm -f model.pkl
	rm -f __pycache__/*.pyc 2>/dev/null || true
	rm -rf *.jl.cov *.jl.mem *.log 2>/dev/null || true

all: setup train run

