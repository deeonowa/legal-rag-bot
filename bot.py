import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from rag_pipeline import RAGPipeline

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальный объект RAG-пайплайна
rag_pipeline = None

# ✅ ВЫДЕЛЕННЫЕ ПУЛЫ ПОТОКОВ (Гарантируют, что задачи не блокируют друг друга)
rag_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="RAG_Worker")
db_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="DB_Worker")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Получена команда /start")
    welcome_text = (
        "👋 Привет! Я — юридический ассистент на базе локальной RAG-системы.\n\n"
        "Я могу отвечать на вопросы по загруженным документам.\n\n"
        "📌 Доступные команды:\n"
        "/help — подробная справка\n"
        "/stats — статистика базы знаний\n"
        "/rebuild — пересоздать векторное хранилище\n\n"
        "💬 Просто напишите ваш вопрос, и я найду ответ в документах!"
    )
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Получена команда /help")
    help_text = (
        "📚 Справка по боту\n\n"
        "Как пользоваться:\n"
        "Просто напишите вопрос текстом, например:\n"
        "• «Какой срок исковой давности?»\n"
        "• «Что будет, если не заплатить налог вовремя?»\n\n"
        "Команды:\n"
        "/start — приветствие\n"
        "/help — эта справка\n"
        "/stats — сколько документов в базе\n"
        "/rebuild — пересоздать хранилище\n\n"
        "Важно: Ответы основаны ТОЛЬКО на загруженных документах."
    )
    # Убран parse_mode, чтобы избежать любых ошибок парсинга
    await update.message.reply_text(help_text)


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Получена команда /stats")
    await update.message.reply_text("📊 Собираю статистику...")
    
    try:
        # ✅ ИСПРАВЛЕНО: get_running_loop() и явное использование db_executor
        loop = asyncio.get_running_loop()
        doc_count = await loop.run_in_executor(db_executor, rag_pipeline.db.get_document_count)
        docs = await loop.run_in_executor(db_executor, rag_pipeline.db.get_all_documents)
        
        stats_text = f"📊 Статистика базы знаний\n\n"
        stats_text += f"📁 Всего документов: {doc_count}\n"
        
        if rag_pipeline.vectorstore is not None:
            chunks_count = await loop.run_in_executor(db_executor, rag_pipeline.vectorstore._collection.count)
            stats_text += f"🔪 Всего чанков: {chunks_count}\n\n"
        else:
            stats_text += f"⚠️ Векторное хранилище не загружено\n\n"
        
        stats_text += "Список документов:\n"
        for doc in docs:
            size_kb = len(doc['content']) / 1024
            stats_text += f"• {doc['title']} ({size_kb:.1f} КБ)\n"
        
        if len(stats_text) > 4000:
            stats_text = stats_text[:4000] + "\n... (список обрезан)"
        
        await update.message.reply_text(stats_text)
    except Exception as e:
        logger.error(f"Ошибка в /stats: {e}")
        await update.message.reply_text("❌ Не удалось получить статистику.")


async def rebuild(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Получена команда /rebuild")
    await update.message.reply_text(
        "🔄 Начинаю пересоздание векторного хранилища...\n"
        "Это может занять 1-2 минуты. Пожалуйста, подождите."
    )
    
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(rag_executor, rag_pipeline.force_rebuild)
        
        await update.message.reply_text(
            "✅ Векторное хранилище успешно пересоздано!\n"
            "Теперь можете задавать вопросы по обновлённым документам."
        )
    except Exception as e:
        logger.error(f"Ошибка в /rebuild: {e}")
        await update.message.reply_text(f"❌ Ошибка при пересоздании: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Получено сообщение: {update.message.text[:50]}...")
    user_question = update.message.text.strip()
    
    if not user_question:
        await update.message.reply_text("Пожалуйста, задайте вопрос текстом.")
        return
    
    # ✅ ВОЗВРАЩЕНО: текстовое сообщение "🤔 Думаю..." вместо send_chat_action
    thinking_msg = await update.message.reply_text("🤔 Думаю...")
    
    try:
        loop = asyncio.get_running_loop()
        # ✅ ИСПРАВЛЕНО: явное использование rag_executor вместо None
        result = await loop.run_in_executor(
            rag_executor, 
            rag_pipeline.ask_with_sources, 
            user_question
        )
        
        answer = result['answer']
        response = f"💡 {answer}\n"
        
        if result['sources_count'] > 0:
            response += f"\n📚 Источники ({result['sources_count']}):\n"
            for i, source in enumerate(result['sources'], 1):
                response += f"{i}. {source['title']}\n"
        else:
            response += "\n⚠️ Подходящих источников не найдено."
        
        if len(response) > 4000:
            response = response[:4000] + "\n... (ответ обрезан)"
        
        # ✅ Удаляем сообщение "Думаю..." и отправляем ответ
        await thinking_msg.delete()
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        # ✅ Удаляем сообщение "Думаю..." даже при ошибке
        await thinking_msg.delete()
        await update.message.reply_text(
            f"❌ Произошла ошибка при обработке вопроса.\n"
            f"Попробуйте переформулировать или используйте /rebuild."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.warning(f'Update {update} caused error: {context.error}')
    if update and update.message:
        await update.message.reply_text("❌ Произошла внутренняя ошибка. Попробуйте позже.")


def main():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not TOKEN:
        print(" Ошибка: токен бота не найден в файле .env!")
        return
    
    print("🚀 Инициализация RAG-системы...")
    global rag_pipeline
    rag_pipeline = RAGPipeline(
        db_path=os.getenv("DB_PATH", "documents.db"),
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    rag_pipeline.build_rag_chain()
    print("✅ RAG-система готова")
    
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("rebuild", rebuild))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    print("\n🤖 Бот запущен! Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()