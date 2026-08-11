import os
import chromadb
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from mistralai import Mistral
from pydantic import BaseModel, Field

# 1. Загружаем переменные окружения (.env)
load_dotenv(find_dotenv())

app = FastAPI(title="AI Knowledge Base (RAG) Microservice")

# 2. Инициализируем клиенты Mistral и ChromaDB
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)

# ChromaDB будет сохранять данные локально в папку ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="company_rules"
)


# 3. Pydantic-схемы
class DocumentInput(BaseModel):
    documents: list[str] = Field(
        description="Список текстов/регламентов компании для загрузки в базу знаний",
        examples=[
            [
                "График работы офиса: с 9:00 до 18:00 с понедельника по пятницу.",
                "Оплата задержек: сверхурочные оплачиваются по коэффициенту 1.5.",
            ]
        ],
    )


class QueryInput(BaseModel):
    question: str = Field(
        description="Вопрос пользователя",
        examples=["Как оплачиваются переработки в компании?"],
    )


class QueryResponse(BaseModel):
    answer: str = Field(
        description="Сгенерированный ответ нейросети на основе найденного контекста"
    )
    found_context: list[str] = Field(
        description="Найденные релевантные фрагменты из векторной базы знаний"
    )


# 4. Эндпоинт загрузки документов в векторную БД
@app.post("/add_documents")
async def add_documents(data: DocumentInput):
    try:
        if not data.documents:
            raise HTTPException(
                status_code=400, detail="Список документов пуст"
            )

        # Генерируем уникальные ID для каждого фрагмента
        existing_count = collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(data.documents))]

        # ChromaDB автоматически превратит тексты в векторы (эмбеддинги) и сохранит их
        collection.add(documents=data.documents, ids=ids)

        return {
            "status": "success",
            "message": f"Успешно добавлено документов: {len(data.documents)}",
            "total_documents_in_db": collection.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 5. Эндпоинт RAG-поиска и ответа на вопрос
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(data: QueryInput):
    try:
        # Шаг 1: Ищем 2 самых похожих по смыслу куска текста в ChromaDB
        results = collection.query(query_texts=[data.question], n_results=2)

        retrieved_docs = (
            results["documents"][0] if results.get("documents") else []
        )

        if not retrieved_docs:
            context_str = "Контекст отсутствует."
        else:
            context_str = "\n---\n".join(retrieved_docs)

        # Шаг 2: Формируем строгий системный промпт с найденным контекстом
        prompt = f"""
Ты — AI-ассистент базы знаний компании.
Твоя задача — ответить на вопрос пользователя, опираясь НА КОНТЕКСТ ниже.

ПРАВИЛА:
1. Используй информацию из контекста и делай из нее прямые логические выводы (например: "прием заявок за 14 дней" означает, что подать заявление нужно минимум за 14 дней).
2. Не придумывай факты, которых нет в контексте.
3. Если контекст вообще не содержит информации по теме вопроса, ответь: "К сожалению, в базе знаний нет информации по этому вопросу."

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context_str}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{data.question}
"""

        # Шаг 3: Отправляем контекст + вопрос в Mistral
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = response.choices[0].message.content

        return QueryResponse(
            answer=answer_text, found_context=retrieved_docs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))