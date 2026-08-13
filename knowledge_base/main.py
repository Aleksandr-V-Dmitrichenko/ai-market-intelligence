import io
import os
import re
import chromadb
import pymorphy3
from chromadb.utils import embedding_functions
from dotenv import find_dotenv, load_dotenv
from docx import Document as DocxDocument
from fastapi import FastAPI, File, HTTPException, UploadFile
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

# 1. Загружаем переменные окружения (.env)
load_dotenv(find_dotenv())

app = FastAPI(title="Production Hybrid RAG Microservice")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)

# Создаем функцию эмбеддинга на базе мультиязычной модели
multilingual_ef = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
)

# ChromaDB сохраняет векторы в папку ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Передаем функцию в коллекцию
collection = chroma_client.get_or_create_collection(
    name="company_rules_v2",
    embedding_function=multilingual_ef,
)

# Настраиваем уманый рекусривный чанкер
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

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


def chunk_text_recursively(text: str) -> list[str]:
    """Умная каскадная нарезка текста по абзацам, предложениям и словам."""
    return text_splitter.split_text(text)

# Инициализируем морфологический анализатор для русского языка
morph = pymorphy3.MorphAnalyzer()

# Вспомогательные функции гибридного поиска
def tokenize(text: str) -> list[str]:
    # Удаляем все знаки препинания, оставляем только буквы и цифры
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    # Режем по пробелам и отбрасываем пустые токены
    words = [word for word in clean_text.split() if len(word) > 1]
    # Приводим КАЖДОЕ слово к его начальной словарной форме (лемме)
    lemmatized_words = [morph.parse(w)[0].normal_form for w in words]
    return lemmatized_words

def reciprocal_rank_fusion(
    vector_docs: list[str], bm25_docs: list[str], k: int = 60, top_n: int = 4
) -> list[str]:
    """
    Алгоритм RRF: объединяет и переранжирует списки результатов от BM25 и Vector Search.
    """
    doc_scores = {}
    # Начисляем очки за позицию в Векторном поиске
    for rank, doc in enumerate(vector_docs):
        if doc not in doc_scores:
            doc_scores[doc] = 0.0
        doc_scores[doc] += 1.0 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        if doc not in doc_scores:
            doc_scores[doc] = 0.0
        doc_scores[doc] += 1.0 / (k + rank + 1)

    # Сортируем документы по убыванию итогового RRF-балла
    sorted_docs = sorted(
        doc_scores.items(), key=lambda item:item[1], reverse=True
    )
    return [doc for doc, score in sorted_docs[:top_n]]

def search_bm25(query: str, all_documents: list[str], top_k: int = 5) -> list[str]:
    """Выполняет точный ключевой поиск BM25 по всему корпусу чанков."""
    if not all_documents:
        return []
    
    tokenized_corpus = [tokenize(doc) for doc in all_documents]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    doc_scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
    )[:top_k]
    return [
        all_documents[i] for i in top_indices if doc_scores[i] > 0
    ]

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

        # 2. Используем рекурсивный чанкинг
        chunks = chunk_text_recursively(raw_text)

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
        # 1. Достаем ВСЕ чанки из БД для BM25 (получаем полный корпус текстов)
        db_data = collection.get()
        all_documents = db_data.get("documents", []) if db_data else []

        if not all_documents:
            return QueryResponse(
                answer="База знаний пуста. Загрузите документы.",
                found_context=[],
            )

        # 2. Векторный поиск (Dense Search) — берутся Top-5 результатов
        vector_results = collection.query(
            query_texts=[data.question], n_results=25
        )
        vector_docs = (
            vector_results["documents"][0]
            if vector_results.get("documents")
            else []
        )
        # 3. Ключевой поиск BM25 (Sparse Search) — берутся Top-5 результатов
        bm25_docs = search_bm25(data.question, all_documents, top_k=25)

        # 4. Объединение и слияние через RRF (берем Топ-4 итоговых чанка)
        hybrid_docs = reciprocal_rank_fusion(
            vector_docs, bm25_docs, k=60, top_n=5
        )

        # 5. Склеиваем контекст
        context_str = (
            "\n---\n".join(hybrid_docs)
            if hybrid_docs
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
            found_context=hybrid_docs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))