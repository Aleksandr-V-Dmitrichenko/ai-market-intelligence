import io
import os
import chromadb
from dotenv import find_dotenv, load_dotenv
from docx import Document as DocxDocument
from fastapi import FastAPI, File, HTTPException, UploadFile
from mistralai import Mistral
from pypdf import PdfReader
from pydantic import BaseModel, Field

# 1. Загружаем переменные окружения (.env)
load_dotenv(find_dotenv())

app = FastAPI(title="AI Knowledge Base (RAG) Microservice")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)

# ChromaDB сохраняет векторы в папку ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_rules")


# Вспомогательные функции для парсинга и чанкинга
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Извлекает сырой текст из PDF, DOCX или TXT файлов."""
    ext = filename.split(".")[-1].lower()
    text = ""

    if ext == "pdf":
        pdf = PdfReader(io.BytesIO(file_bytes))
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    elif ext in ["docx", "doc"]:
        doc = DocxDocument(io.BytesIO(file_bytes))
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"

    elif ext == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(
            f"Неподдерживаемый формат файла: {ext}. Допустимы PDF, DOCX, TXT."
        )

    return text.strip()


def chunk_text(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    """
    Режет большой текст на чанки фиксированного размера с перекрытием (overlap).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# Pydantic-схемы
class QueryInput(BaseModel):
    question: str = Field(
        description="Вопрос пользователя к базе знаний",
        examples=["Как оплачиваются переработки в компании?"],
    )


class QueryResponse(BaseModel):
    answer: str = Field(description="Сгенерированный ответ нейросети")
    found_context: list[str] = Field(
        description="Найденные релевантные фрагменты"
    )


# Эндпоинт 1: Загрузка файлов (PDF, DOCX, TXT)
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # 1. Извлекаем сырой текст из файла
        raw_text = extract_text_from_file(content, file.filename)
        if not raw_text:
            raise HTTPException(
                status_code=400, detail="Файл пуст или не содержит текста"
            )

        # 2. Нарезаем текст на чанки по 500 символов с перекрытием 100
        chunks = chunk_text(raw_text, chunk_size=500, overlap=100)

        # 3. Генерируем уникальные ID и добавляем в ChromaDB
        existing_count = collection.count()
        ids = [f"file_doc_{existing_count + i}" for i in range(len(chunks))]

        collection.add(documents=chunks, ids=ids)

        return {
            "status": "success",
            "filename": file.filename,
            "extracted_text_length": len(raw_text),
            "created_chunks_count": len(chunks),
            "total_documents_in_db": collection.count(),
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт 2: Вопрос к базе знаний
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(data: QueryInput):
    try:
        results = collection.query(query_texts=[data.question], n_results=2)
        retrieved_docs = (
            results["documents"][0] if results.get("documents") else []
        )

        context_str = (
            "\n---\n".join(retrievedsra_docs)
            if retrieved_docs
            else "Контекст отсутствует."
        )

        prompt = f"""
Ты — AI-ассистент базы знаний компании.
Твоя задача — ответить на вопрос пользователя, опираясь НА КОНТЕКСТ ниже.

ПРАВИЛА:
1. Используй информацию из контекста и делай из нее прямые логические выводы.
2. Не придумывай факты, которых нет в контексте.
3. Если контекст не содержит информации, ответь: "К сожалению, в базе знаний нет информации по этому вопросу."

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context_str}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{data.question}
"""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
        )

        return QueryResponse(
            answer=response.choices[0].message.content,
            found_context=retrieved_docs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))