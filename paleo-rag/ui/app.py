"""
Gradio UI for Paleo RAG system.

Provides a simple web interface for querying the RAG system
with answer display and source citations.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from config import config
from rag.rag_chain import RAGChain, RAGResponse

logger = logging.getLogger(__name__)


# Global RAG chain instance (lazy-loaded)
_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get or create the global RAG chain instance."""
    global _rag_chain
    
    if _rag_chain is None:
        logger.info("Initializing RAG chain for UI...")
        _rag_chain = RAGChain()
    
    return _rag_chain


def format_sources_html(sources: list[dict[str, Any]]) -> str:
    """Format sources as HTML for display."""
    if not sources:
        return "<p><em>Источники не найдены</em></p>"
    
    html_parts = []
    
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Неизвестно")
        epoch = source.get("epoch", "Н/Д")
        hall = source.get("hall", "Н/Д")
        doc_type = source.get("doc_type", "unknown")
        score = source.get("score", 0)
        
        html_parts.append(f"""
        <div style="border-left: 3px solid #4CAF50; padding-left: 12px; margin: 8px 0;">
            <strong>Источник {i}:</strong> {title}<br>
            <span style="color: #666; font-size: 0.9em;">
                Эпоха: {epoch} | Зал: {hall} | Тип: {doc_type}<br>
                Релевантность: {score:.4f}
            </span>
        </div>
        """)
    
    return "".join(html_parts)


def query_rag(
    question: str,
    use_hyde: bool = True,
    top_k: int = 5,
) -> tuple[str, str, str]:
    """
    Process a question through the RAG chain.
    
    Args:
        question: User's question.
        use_hyde: Whether to use HyDE query rewriting.
        top_k: Number of documents to retrieve.
    
    Returns:
        Tuple of (answer, sources_html, metadata).
    """
    if not question.strip():
        return (
            "Пожалуйста, введите вопрос.",
            "",
            "Статус: Ошибка ввода",
        )
    
    try:
        chain = get_rag_chain()
        
        # Get response
        response = chain.invoke(
            query=question,
            top_k=top_k,
            use_hyde=use_hyde,
        )
        
        # Format sources
        sources_list = [s.to_dict() for s in response.sources]
        sources_html = format_sources_html(sources_list)
        
        # Format metadata
        hyde_status = "HyDE включен" if use_hyde else "HyDE выключен"
        metadata = (
            f"⏱ Время обработки: {response.processing_time_sec:.2f}s\n"
            f"📚 Источников: {response.num_sources}\n"
            f"🔄 {hyde_status}"
        )
        
        return (response.answer, sources_html, metadata)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        error_msg = f"Произошла ошибка при обработке запроса: {str(e)}"
        return (error_msg, "", f"Статус: Ошибка\n{str(e)}")


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    
    with gr.Blocks(
        title="Paleo RAG — Палеонтологический Музей",
        theme=gr.themes.Soft(),
    ) as ui:
        
        gr.Markdown("""
        # 🦕 Paleo RAG — Виртуальный помощник палеонтологического музея
        
        Задавайте вопросы о динозаврах, окаменелостях, методах датирования и экспонатах музея.
        Система использует RAG (Retrieval-Augmented Generation) с HyDE для поиска релевантной информации.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Ваш вопрос",
                    placeholder="Например: Какой динозавр был самым большим?",
                    lines=3,
                )
                
                with gr.Row():
                    hyde_checkbox = gr.Checkbox(
                        label="Использовать HyDE",
                        value=True,
                        info="Улучшает поиск за счёт генерации гипотетического документа",
                    )
                    
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Количество источников",
                    )
                
                submit_btn = gr.Button("🔍 Ответить", variant="primary")
            
            with gr.Column(scale=2):
                metadata_output = gr.Textbox(
                    label="Метаданные",
                    lines=4,
                    interactive=False,
                )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=3):
                answer_output = gr.Textbox(
                    label="Ответ эксперта",
                    lines=6,
                    show_copy_button=True,
                )
            
            with gr.Column(scale=2):
                sources_output = gr.HTML(label="Источники")
        
        # Examples
        gr.Examples(
            examples=[
                "Какой динозавр был самым большим?",
                "Почему вымерли динозавры?",
                "Как определяют возраст окаменелостей?",
                "Где можно найти скелеты тираннозавра?",
                "Что такое аммониты?",
                "Какие динозавры имели перья?",
                "Как работают палеонтологические раскопки?",
                "Можно ли клонировать динозавра?",
            ],
            inputs=[question_input],
            label="Примеры вопросов",
        )
        
        # Wire up the event
        submit_btn.click(
            fn=query_rag,
            inputs=[question_input, hyde_checkbox, top_k_slider],
            outputs=[answer_output, sources_output, metadata_output],
        )
        
        # Also trigger on Enter key
        question_input.submit(
            fn=query_rag,
            inputs=[question_input, hyde_checkbox, top_k_slider],
            outputs=[answer_output, sources_output, metadata_output],
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ О системе
        
        **RAG** (Retrieval-Augmented Generation) — метод, сочетающий поиск по базе знаний 
        с генерацией ответов на основе языковой модели.
        
        **HyDE** (Hypothetical Document Embeddings) — техника переформулирования запроса,
        при которой сначала генерируется гипотетический документ-ответ, а затем по нему
        ищутся релевантные документы в базе.
        
        **Модели:**
        - Эмбеддинги: BAAI/bge-small-ru-v1.5 (русский язык)
        - LLM: Qwen 2.5 7B через Ollama
        
        **База знаний:** Синтетические данные палеонтологического музея (200 документов)
        """)
    
    return ui


def launch_ui(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """
    Launch the Gradio UI.
    
    Args:
        server_name: Server hostname.
        server_port: Server port.
        share: Whether to create a public shareable link.
    """
    logger.info(f"Starting UI on {server_name}:{server_port}...")
    
    ui = create_ui()
    ui.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


def main() -> None:
    """Main entry point for UI."""
    logging.basicConfig(level=logging.INFO)
    
    # Check if index exists
    if not config.FAISS_INDEX_PATH.exists():
        print(f"Vector store not found at {config.FAISS_INDEX_PATH}")
        print("Please run main.py first to build the index.")
        return
    
    print("Starting Paleo RAG UI...")
    print(f"Open http://localhost:7860 in your browser")
    
    launch_ui()


if __name__ == "__main__":
    main()
