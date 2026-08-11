import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mistralai import Mistral

#1. Инициализируем сервер FastAPI
app = FastAPI(title="Market Intelligence AI Microservice")

load_dotenv(find_dotenv())

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

# 2. Создаем Pydantic-схему (ВЫХОДНЫЕ ДАННЫЕ ОТ ИИ)
# Это трафарет, которому ИИ ОБЯЗАН следовать

class MarketItemAnalysis(BaseModel):
    product_name: str = Field(description="Название товара, услуги или криптовалютного актива")
    category: str = Field(description="Категория: например, E-commerce, B2B, Crypto, Services")
    price: float | None = Field(default=None, description="Найденная цена или курс, если есть")
    currency: str | None = Field(default="USD", description="Валюта цены (USD, EUR, RUB, KZT и т.д.)")
    event_type: str = Field(
        description="Тип события: PRICE_DROP (скидка), NEW_ITEM (новинка), ANOMALY (резкий скачок), INFO (информация)"
    )
    criticality_score: int = Field(
        description="Индекс важности события для бизнеса от 1 (рутина) до 10 (критично)", ge=1, le=10
    )
    summary: str = Field(description="Краткий вывод для руководителя в 1 предложение")

# 3. ВХОДНАЯ СХЕМА (Что присылает n8n или парсер)

class RawDataInput(BaseModel):
    raw_text: str = Field(description="Сырой текст статьи, поста или спаршенной страницы")

# 4. Создаем "Ручку" (Endpoint)
@app.post("/analyze", response_model=MarketItemAnalysis)
async def analyze_market_data(data: RawDataInput):
    """Принимает сырой текст, отправляет его в LLM и возвращает строго валидированный JSON"""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — AI-аналитик рынка. Проанализируй текст и верни результат "
                    "СТРОГО в формате JSON по ключам: product_name, category, price, "
                    "currency, event_type, criticality_score, summary. "
                    "Не пиши никакой вводный текст и не используй markdown-блоки (без ```json)."
                ),
            },
            {"role": "user", "content": data.raw_text},
        ]

        # Правильный вызов API для MistralClient
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            response_format={"type": "json_object"},
        )

        # 1. Достаем JSON-строку из ответа Mistral
        raw_json_string = response.choices[0].message.content

        # 2. Превращаем JSON-строку в словарь Python
        parsed_dict = json.loads(raw_json_string)

        # 3. Валидируем через Pydantic и отдаем клиентский ответ
        return MarketItemAnalysis(**parsed_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))