# AI Market Intelligence & Lead Processing Microservices

Набор асинхронных AI-микросервисов на базе **FastAPI**, **Pydantic** и **Mistral AI** для автоматизации бизнес-процессов.

## Возможности сервисов

1. **Market Intelligence Normalizer (`/analyze`)**
   * Анализирует сырые тексты новостей/постов конкурентов.
   * Вытаскивает сущности: название товара, категория, цена, валюта, тип события и оценка важности (1-10).
2. **Lead Qualifier & Scorer (`/classify`)**
   * Обрабатывает входящие обращения клиентов.
   * Распознает имя, контакты, намерение (PURCHASE, QUESTION, SPAM, COMPLAINT), бюджет и рассчитывает `lead_score`.

## Технологический стек

* **Python 3.11+**
* **FastAPI** — асинхронный веб-фреймворк.
* **Pydantic v2** — строгая валидация входящих и исходящих JSON-структур.
* **Mistral AI SDK** — работа с LLM.
* **python-dotenv** — безопасное управление ключами через `.env`.

## 🚀 Быстрый запуск

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/Aleksandr-V-Dmitrichenko/ai-market-intelligence.git
   cd ai-market-intelligence
2. **Установите зависимости:**
   ```bash
   pip install fastapi uvicorn mistralai python-dotenv pydantic
3. **Настройте переменные окружения:**

   Создайте файл .env в корне проекта:
   
   MISTRAL_API_KEY=your_mistral_api_key_here
5. **Запустите сервер:**
   ```bash
   uvicorn main:app --reload
6. **Откройте интерактивную документацию Swagger UI:**
   
   http://127.0.0.1:8000/docs