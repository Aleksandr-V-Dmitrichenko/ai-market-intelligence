import os
import io
import re
import json
import csv
from io import StringIO
import chromadb
import pymorphy3
from chromadb.utils import embedding_functions
from dotenv import find_dotenv, load_dotenv
from docx import Document as DocxDocument
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

# 1. Загружаем переменные окружения (.env)
load_dotenv(find_dotenv())

app = FastAPI(title="Production Parent-Child Hybrid RAG Service")

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

# ─── PARENT-CHILD ЧАНКЕРЫ ────────────────────────────────────────────────────────
# 1. Родительский чанкер (крупный контекст для LLM)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# 2. Дочерний чанкер (точный точечный поиск для векторов и BM25)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""],
)
# Хранилище родительских чанков key - value
DOCSTORE_FILE = "./docstore.json"

def load_docstore() -> dict:
    if os.path.exists(DOCSTORE_FILE):
        try:
            with open(DOCSTORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # Если файл пуст или поврежден — безопасно возвращаем пустой словарь
            return {}
    return {}


def save_docstore(data: dict):
    with open(DOCSTORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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

def process_csv_content(content_bytes: bytes, filename: str) -> list[str]:
    # Декодируем байты в строку
    text_content = content_bytes.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(StringIO(text_content))

    parent_chunks = []

    for row_idx, row in enumerate(reader, start=1):
        # Очищаем ячейки от None, пустых строк и пробелов
        clean_row = {k.strip(): v.strip() for k, v in row.items() if k and v and v.strip()}

        if not clean_row:
            continue

        # Формируем семантическую строку "Колонка: Значение"
        row_str = " | ".join([f"{k}: {v}" for k, v in clean_row.items()])
        full_chunk = f"[Источник: {filename} | Строка {row_idx}] {row_str}"

        parent_chunks.append(full_chunk)

    return parent_chunks

# Глобальный реестр статусов фоновых задач
tasks_db = {}


def process_file_in_background(task_id: str, content: bytes, filename: str):
    """Тяжелая функция векторизации, выполняемая в фоновом потоке."""
    try:
        tasks_db[task_id] = {"status": "processing", "filename": filename}

        ext = filename.split(".")[-1].lower()

        # 1. Извлекаем и нарезаем родительские чанки
        if ext == "csv":
            parent_chunks = process_csv_content(content, filename)
        else:
            raw_text = extract_text_from_file(content, filename)
            parent_chunks = parent_splitter.split_text(raw_text)

        docstore = load_docstore()
        child_documents = []
        child_metadatas = []
        child_ids = []

        child_counter = collection.count()

        # 2. Формируем дочерние чанки
        initial_docstore_len = len(docstore)
        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"parent_{filename}_{p_idx}_{initial_docstore_len}"
            docstore[parent_id] = parent_text

            child_chunks = child_splitter.split_text(parent_text)

            for c_text in child_chunks:
                child_id = f"child_{child_counter}"
                child_counter += 1
                child_documents.append(c_text)
                child_metadatas.append({"parent_id": parent_id})
                child_ids.append(child_id)

        # 3. Сохраняем в KV-Docstore и ChromaDB
        save_docstore(docstore)
        if child_documents:
            collection.add(
                documents=child_documents,
                metadatas=child_metadatas,
                ids=child_ids,
            )

        # 4. Обновляем финальный статус задачи
        tasks_db[task_id] = {
            "status": "completed",
            "filename": filename,
            "created_parent_chunks": len(parent_chunks),
            "created_child_chunks": len(child_documents),
            "total_children_in_vector_db": collection.count(),
        }

    except Exception as e:
        tasks_db[task_id] = {"status": "failed", "error": str(e)}


# Инициализируем морфологический анализатор для русского языка
morph = pymorphy3.MorphAnalyzer()

def tokenize(text: str) -> list[str]:
    """Токенизация с защитой типов и лемматизацией."""
    if isinstance(text, dict):
        text = text.get("text", "")

    if not isinstance(text, str):
        text = str(text)

    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in clean_text.split() if len(w) > 1]
    return [morph.parse(w)[0].normal_form for w in words]

def reciprocal_rank_fusion_parent_child(
    vector_child_metas: list[dict], 
    bm25_child_metas: list[dict], 
    k: int = 60, 
    top_n: int = 3
) -> list[str]:
    """RRF, который группирует находки по parent_id и отбирает родительские чанки."""
    parent_scores = {}
    
    # Начисляем очки родителям на основе рангов векторного поиска
    for rank, meta in enumerate(vector_child_metas):
        pid = meta.get("parent_id")
        if pid:
            parent_scores[pid] = parent_scores.get(pid, 0.0) + (1.0 / (k + rank + 1))
        
    # Начисляем очки родителям на основе рангов BM25
    for rank, meta in enumerate(bm25_child_metas):
        pid = meta.get("parent_id")
        if pid:
            parent_scores[pid] = parent_scores.get(pid, 0.0) + (1.0 / (k + rank + 1))

    # Сортируем родительские ID по итоговым баллам
    sorted_parents = sorted(
        parent_scores.items(), key=lambda item: item[1], reverse=True
    )
    top_parent_ids = [pid for pid, score in sorted_parents[:top_n]]

    # Извлекаем тексты родителей из Docstore
    docstore = load_docstore()
    return [docstore[pid] for pid in top_parent_ids if pid in docstore]

def search_bm25(
    query: str, child_docs: list[dict], top_k: int = 25
) -> list[dict]:
    """Поиск по ключевым словам BM25 среди Дочерних чанков."""
    if not child_docs:
        return []

    corpus_texts = [
        doc["text"] if isinstance(doc, dict) else str(doc)
        for doc in child_docs
    ]
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]
    return [child_docs[i] for i in top_indices if scores[i] > 0]

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


# Эндпоинт 1: Загрузка файлов (PDF, DOCX, TXT, CSV)
@app.post("/upload_file", status_code=202)
async def upload_file(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    content = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in ["pdf", "docx", "doc", "txt", "csv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла: {ext}.",
        )

    # Генерируем уникальный ID задачи
    task_id = f"task_{file.filename}_{len(tasks_db) + 1}"

    # Ставим задачу в фоновую очередь FastAPI
    background_tasks.add_task(
        process_file_in_background,
        task_id=task_id,
        content=content,
        filename=file.filename,
    )

    # Возвращаем мгновенный ответ со статусом HTTP 202 Accepted
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": "Файл принят на обработку. Проверяйте статус через GET /tasks/{task_id}",
    }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Эндпоинт для polling-проверки статуса индексации."""
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task

# Эндпоинт 2: Вопрос к базе знаний
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(data: QueryInput):
    try:
        db_data = collection.get(include=["documents", "metadatas"])
        all_child_docs = []

        if db_data and db_data.get("documents"):
            for text, meta in zip(db_data["documents"], db_data["metadatas"]):
                all_child_docs.append({"text": text, "metadata": meta})

        if not all_child_docs:
            return QueryResponse(
                answer="База знаний пуста. Загрузите документы.",
                found_context=[],
            )

        # 1. Векторный поиск по Child-чанкам (Top-25)
        vector_results = collection.query(
            query_texts=[data.question], n_results=25
        )
        vector_child_metas = (
            vector_results["metadatas"][0]
            if vector_results.get("metadatas")
            else []
        )
        # 2. BM25 поиск по Child-чанкам (Top-25)
        bm25_child_results = search_bm25(data.question, all_child_docs, top_k=25)
        bm25_child_metas = [item["metadata"] for item in bm25_child_results]

        # 3. RRF Слияние: ищем точных детей, а вытягиваем их РОДИТЕЛЕЙ (Top-3)
        parent_contexts = reciprocal_rank_fusion_parent_child(
            vector_child_metas, bm25_child_metas, k=60, top_n=3
        )

        # 4. Склеиваем контекст
        context_str = (
            "\n---\n".join(parent_contexts)
            if parent_contexts
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
            found_context=parent_contexts,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))