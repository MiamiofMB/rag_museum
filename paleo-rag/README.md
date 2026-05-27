# 🦕 Paleo RAG — RAG-система палеонтологического музея

Учебный проект RAG (Retrieval-Augmented Generation) системы для виртуального помощника палеонтологического музея. Система отвечает на вопросы посетителей о динозаврах, окаменелостях, методах датирования и экспонатах музея.

## ✨ Особенности

- **HyDE (Hypothetical Document Embeddings)** — основной метод переформулирования запросов для улучшения поиска
- **Fallback режим** — автоматический откат к стандартному поиску при неудаче HyDE
- **Русский язык** — все интерфейсы, данные и модели оптимизированы для русского языка
- **Полностью локально** — не требует внешних API, работает на CPU
- **Модульная архитектура** — легко расширяемый код с чётким разделением ответственности

## 📁 Структура проекта

```
paleo-rag/
├── .env.example          # Шаблон конфигурации
├── requirements.txt      # Зависимости Python
├── config.py            # Конфигурация через pydantic-settings
├── main.py              # Точка входа
├── data/
│   ├── generate_data.py # Генератор синтетических данных
│   └── raw_synthetic.jsonl # Сгенерированные документы
├── pipeline/
│   ├── chunker.py       # Разбиение текста на чанки
│   ├── embedder.py      # BGE эмбеддинги
│   └── vector_store.py  # FAISS индекс
├── rag/
│   ├── hyde_rewriter.py # HyDE переформулирование
│   ├── retriever.py     # Семантический поиск
│   └── rag_chain.py     # Полный RAG пайплайн
├── eval/
│   └── evaluate.py      # Метрики (HitRate, Relevancy)
├── ui/
│   └── app.py           # Gradio интерфейс
└── README.md            # Этот файл
```

## 🛠 Стек технологий

| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.10+ |
| LangChain | 0.3+ (langchain, langchain-community, langchain-core) |
| Векторное хранилище | FAISS (faiss-cpu) |
| Эмбеддинги | sentence-transformers (BAAI/bge-small-ru-v1.5) |
| LLM | Ollama (qwen2.5:7b или llama3.1:8b) |
| UI | Gradio |
| Конфигурация | pydantic-settings + .env |
| Метрики | ragas (HitRate, AnswerRelevancy) |

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd paleo-rag
pip install -r requirements.txt
```

### 2. Установка и запуск Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
```

**Windows:**
1. Скачайте установщик с [ollama.com](https://ollama.com)
2. Установите и запустите терминал
3. Выполните: `ollama pull qwen2.5:7b`

**Проверка работы:**
```bash
ollama run qwen2.5:7b "Привет!"
```

### 3. Настройка конфигурации

```bash
cp .env.example .env
```

При желании отредактируйте `.env` (можно использовать значения по умолчанию).

### 4. Запуск системы

```bash
python main.py
```

Система автоматически:
1. Сгенерирует 200 синтетических документов
2. Построит FAISS индекс
3. Запустит веб-интерфейс

Откройте http://localhost:7860 в браузере.

## 📝 Использование

### Команды main.py

```bash
# Полный запуск
python main.py

# Пропустить генерацию данных (если уже есть)
python main.py --skip-data

# Пропустить построение индекса
python main.py --skip-index

# Только оценка качества
python main.py --eval-only

# Принудительно перегенерировать всё
python main.py --force

# Без запуска UI (только подготовка)
python main.py --no-ui
```

### Примеры вопросов

- Какой динозавр был самым большим?
- Почему вымерли динозавры?
- Как определяют возраст окаменелостей?
- Где можно найти скелеты тираннозавра?
- Что такое аммониты?
- Какие динозавры имели перья?
- Как работают палеонтологические раскопки?
- Можно ли клонировать динозавра?

## 🔧 Конфигурация (.env)

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `DATA_DIR` | ./data | Директория данных |
| `INDEX_DIR` | ./index | Директория индексов |
| `RAW_DATA_FILE` | ./data/raw_synthetic.jsonl | Файл документов |
| `FAISS_INDEX_PATH` | ./index/faiss_index | Путь к FAISS индексу |
| `EMBEDDING_MODEL` | BAAI/bge-small-ru-v1.5 | Модель эмбеддингов |
| `LLM_MODEL` | qwen2.5:7b | Ollama модель |
| `LLM_BASE_URL` | http://localhost:11434 | URL Ollama сервера |
| `TOP_K` | 5 | Количество результатов поиска |
| `CHUNK_SIZE` | 300 | Размер чанка (токены) |
| `CHUNK_OVERLAP` | 50 | Перекрытие чанков |
| `NUM_DOCUMENTS` | 200 | Количество документов |
| `RANDOM_SEED` | 42 | Seed для воспроизводимости |
| `TEMPERATURE` | 0.7 | Температура LLM |
| `MAX_TOKENS` | 512 | Максимум токенов ответа |

## 📊 Архитектура

### RAG пайплайн с HyDE

```
Вопрос пользователя
        ↓
┌───────────────────────┐
│   HyDE Rewriter       │ ← Генерация гипотетического
│   (Ollama LLM)        │   научного текста
└───────────────────────┘
        ↓
┌───────────────────────┐
│   Embedder            │ ← BGE-small-ru-v1.5
│   (sentence-transformers)
└───────────────────────┘
        ↓
┌───────────────────────┐
│   FAISS Retriever     │ ← Поиск top-k чанков
└───────────────────────┘
        ↓
┌───────────────────────┐
│   Context Assembly    │ ← Сборка контекста
└───────────────────────┘
        ↓
┌───────────────────────┐
│   Answer Generator    │ ← Финальный ответ (Ollama)
│   (Ollama LLM)        │
└───────────────────────┘
        ↓
Ответ + Источники
```

### HyDE fallback механизм

Если HyDE генерация:
- Возвращает пустой ответ
- Содержит менее ~30 токенов
- Содержит ошибки LLM

→ Автоматически используется исходный вопрос пользователя

## 📈 Оценка качества

Запустите оценку:

```bash
python main.py --eval-only
```

Метрики:
- **HitRate@K** — доля запросов, где релевантный документ в топ-K
- **Answer Relevancy** — релевантность ответа (0-1) на основе keyword matching

Результаты сохраняются в `index/evaluation_report.json`.

## 🧪 Модульное тестирование

Каждый модуль можно запустить отдельно:

```bash
# Тест генерации данных
python -m data.generate_data

# Тест чанкинга
python -m pipeline.chunker

# Тест эмбеддингов
python -m pipeline.embedder

# Тест векторного хранилища
python -m pipeline.vector_store

# Тест HyDE
python -m rag.hyde_rewriter

# Тест ретривера (требует индекс)
python -m rag.retriever

# Тест RAG цепочки (требует индекс)
python -m rag.rag_chain

# Тест оценки (требует индекс)
python -m eval.evaluate
```

## 🔄 Переключение моделей

### Смена LLM

В `.env` измените:
```
LLM_MODEL=llama3.1:8b
```

Предварительно скачайте модель:
```bash
ollama pull llama3.1:8b
```

### Смена эмбеддингов

Для других языков или большей точности:
```
EMBEDDING_MODEL=BAAI/bge-large-ru-v1.5
```

⚠️ Потребуется перестроить индекс после смены модели эмбеддингов!

## ⚠️ Требования к ресурсам

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 ядра | 8 ядер |
| Диск | 5 GB | 10 GB |
| GPU | Не требуется | Опционально для LLM |

## 📚 Дополнительные материалы

- [LangChain Documentation](https://python.langchain.com/)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [FAISS Documentation](https://faiss.ai/)
- [Ollama Documentation](https://ollama.com/)
- [BGE Models](https://huggingface.co/BAAI)

## 👨‍💻 Для разработчиков

### Добавление новых типов документов

1. Откройте `data/generate_data.py`
2. Добавьте новый шаблон в соответствующий список (`EXHIBIT_TEMPLATES`, `FAQ_TEMPLATES`, etc.)
3. Реализуйте функцию генерации
4. Обновите распределение в `generate_documents()`

### Расширение промптов

Промпты находятся в:
- `rag/hyde_rewriter.py` — `HYDE_PROMPT_TEMPLATE`
- `rag/rag_chain.py` — `RAG_PROMPT_TEMPLATE`

### Логирование

Все модули используют стандартный `logging`. Уровень настраивается в `main.py`:

```python
logging.basicConfig(level=logging.DEBUG)  # Подробные логи
```

## 📄 Лицензия

Учебный проект. Код распространяется под лицензией MIT.

---

**Автор:** Учебный проект по RAG-системам  
**Версия:** 1.0.0  
**Дата:** 2024
