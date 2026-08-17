# 1. Берем легкий официальный образ Python 3.11
FROM python:3.11-slim

# 2. Устанавливаем рабочую папку внутри контейнера
WORKDIR /app

# 3. Отключаем генерацию .pyc файлов и включаем мгновенный вывод логов в консоль
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Сначала копируем только requirements.txt (чтобы Docker закэшировал установку библиотек)
COPY requirements.txt .

# 5. Устанавливаем зависимости
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt
# 6. Копируем остальной код нашего приложения
COPY . .

# 7. Открываем порт 8000
EXPOSE 8000

# 8. Команда запуска сервера Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]  