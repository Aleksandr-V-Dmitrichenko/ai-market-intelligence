import os, json
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mistralai import Mistral

app = FastAPI(title="AI Lead Classifier & Scorer")

load_dotenv(find_dotenv())

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)


class LeadInput(BaseModel):
    client_msg: str | None = Field(description="текст сообщения от клиента")

class LeadAnalysis(BaseModel):
    client_name: str | None = Field(default=None, description="имя клиента, если он представился")
    contact_info: str | None = Field(default=None, description="телефон/Telegram/email, если клиент их указал")
    intent: str = Field(
        description="намерения клиента (выберите одно: PURCHASE — хочет купить, QUESTION — просто спрашивает, SPAM — спам/реклама, COMPLAINT — жалоба)"
    )
    budget_mentioned: float | None = Field(default=None, description="укузанный бюджет (числом), если клиент его назвал")
    lead_score: int = Field(description="оценка горячности лида от 1 до 10 (где 10 — готов купить прямо сейчас)", ge=1, le=10)
    recommended_action: str = Field(description="рекомендация для менеджера в 1 предложение (например: Срочно перезвонить, клиент горячий!)")

@app.post("/classify", response_model=LeadAnalysis)
async def post_classify(data: LeadInput):
    try:
        messages = [
            {
                "role": "system",
                "content": "Ты — AI-квалификатор отделов продаж. Проанализируй сообщение лида и верни JSON"
                "строго в формате client_name, contact_info, intent, budget_mentioned, lead_score, recommended_action"                
                "НЕ создавай вложенные объекты или словари!"
                "lead_score оценка всегда в int от 1 до 10"
                "Не пиши никакой вводный текст и не используй markdown-блоки (без ```json)."
            },
            {"role": "user", "content": data.client_msg},
        ]

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            response_format={"type": "json_object"},
        )

        raw_json_string = response.choices[0].message.content
        parsed_dict = json.loads(raw_json_string)

        return LeadAnalysis(**parsed_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    