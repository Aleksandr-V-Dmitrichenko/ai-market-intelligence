import os
import pytest
from fastapi.testclient import TestClient
from main import (
    app,
    tasks_db,
    tokenize,
    parent_splitter,
    child_splitter,
    reciprocal_rank_fusion_parent_child,
)

client = TestClient(app)

def test_tokenize_lemmatization():
    raw_text = "Компания выплачивает ЕЖЕМЕСЯЧНЫЕ премии сотрудникам!"
    tokens = tokenize(raw_text)
    
    # Проверяем, что знаки препинания удалены, регистр снижен,
    # а слова приведены к начальной форме (лемме)
    assert "компания" in tokens
    assert "ежемесячный" in tokens
    assert "премия" in tokens
    assert "сотрудник" in tokens


def test_parent_child_chunking():
    # Создаем тестовый текст длиннее родительского порога (>1000 символов)
    paragraph = "Каждый сотрудник имеет право на ежегодный оплачиваемый отпуск. "
    sample_text = paragraph * 25  # ~1600 символов

    # 1. Проверяем нарезку на Родительские чанки (крупный контекст)
    parent_chunks = parent_splitter.split_text(sample_text)
    assert len(parent_chunks) >= 2
    assert len(parent_chunks[0]) <= 1100  # Допустимый размер с учетом overlap

    # 2. Проверяем нарезку первого родителя на Дочерние чанки (точечный поиск)
    child_chunks = child_splitter.split_text(parent_chunks[0])
    assert len(child_chunks) >= 4
    assert len(child_chunks[0]) <= 250


def test_rrf_parent_child_fusion(monkeypatch):
    # Мокаем (подменяем) функцию загрузки docstore, чтобы не читать реальный диск
    mock_docstore = {
        "parent_1": "Текст первого родителя о правилах отпусков.",
        "parent_2": "Текст второго родителя об оплате переработок.",
        "parent_3": "Текст третьего родителя про дресс-код.",
    }
    
    import main
    monkeypatch.setattr(main, "load_docstore", lambda: mock_docstore)

    # Эмулируем результаты:
    # Vector-поиск нашел детей родителя parent_1 и parent_2
    vector_child_metas = [{"parent_id": "parent_1"}, {"parent_id": "parent_2"}]
    # BM25-поиск нашел детей родителя parent_2 и parent_3
    bm25_child_metas = [{"parent_id": "parent_2"}, {"parent_id": "parent_3"}]

    # Так как parent_2 нашли И векторный поиск, И BM25, он должен занять 1-е место!
    top_parents = reciprocal_rank_fusion_parent_child(
        vector_child_metas=vector_child_metas,
        bm25_child_metas=bm25_child_metas,
        top_n=2
    )

    assert len(top_parents) == 2
    assert top_parents[0] == mock_docstore["parent_2"]  # Победитель RRF

def test_upload_file_async_and_task_polling():
    """Тест фоновой загрузки файла и проверки статуса через /tasks/{task_id}."""
    # 1. Готовим фейковый CSV-файл в памяти
    csv_content = "Title,Price\nБрус обрезной,1500\nДоска сухая,800"
    files = {
        "file": (
            "test_async.csv",
            csv_content.encode("utf-8"),
            "text/csv",
        )
    }

    # 2. Отправляем запрос на загрузку
    response = client.post("/upload_file", files=files)

    # Проверяем мгновенный ответ 202 Accepted
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert "task_id" in data

    task_id = data["task_id"]

    # 3. Проверяем статус задачи через GET /tasks/{task_id}
    task_response = client.get(f"/tasks/{task_id}")
    assert task_response.status_code == 200

    task_data = task_response.json()
    assert task_data["status"] == "completed"
    assert task_data["created_parent_chunks"] == 2


def test_get_non_existent_task():
    """Тест обращения к несуществующей фоновой задаче."""
    response = client.get("/tasks/task_does_not_exist_999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Задача не найдена"