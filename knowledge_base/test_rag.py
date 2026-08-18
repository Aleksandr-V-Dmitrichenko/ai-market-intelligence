import os
import pytest
from main import (
    tokenize,
    parent_splitter,
    child_splitter,
    reciprocal_rank_fusion_parent_child,
)

# ─── 1. ТЕСТ ТОКЕНИЗАЦИИ И ЛЕММАТИЗАЦИИ ───────────────────────────────────────
def test_tokenize_lemmatization():
    raw_text = "Компания выплачивает ЕЖЕМЕСЯЧНЫЕ премии сотрудникам!"
    tokens = tokenize(raw_text)
    
    # Проверяем, что знаки препинания удалены, регистр снижен,
    # а слова приведены к начальной форме (лемме)
    assert "компания" in tokens
    assert "ежемесячный" in tokens
    assert "премия" in tokens
    assert "сотрудник" in tokens


# ─── 2. ТЕСТ PARENT-CHILD ЧАНКИНГА ───────────────────────────────────────────
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


# ─── 3. ТЕСТ АЛГОРИТМА RRF (Reciprocal Rank Fusion) ──────────────────────────
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