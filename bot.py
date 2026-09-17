import asyncio
import logging
import os
from pathlib import Path
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


async def initialize_rag():
    """Инициализация RAG-системы при старте бота"""
    global rag_pipeline
    logger.info(" Инициализация RAG-системы...")
    rag_pipeline = RAGPipeline(
        db_path=os.getenv("DB_PATH", "documents.db"),
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    rag_pipeline.build_rag_chain()
    logger.info("✅ RAG-система готова к работе")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start — приветствие"""
    welcome_text = (
        "👋 Привет! Я — юридический ассистент на базе локальной RAG-системы.\n\n"
        "Я могу отвечать на вопросы по загруженным документам (149-ФЗ, 152-ФЗ, 187-ФЗ и др.).\n\n"
        "📌 Доступные команды:\n"
        "/help — подробная справка\n"
        "/stats — статистика базы знаний\n"
        "/rebuild — пересоздать векторное хранилище\n\n"
        " Просто напишите ваш вопрос, и я найду ответ в документах!"
    )
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help — справка"""
    help_text = (
        "📚 **Справка по боту**\n\n"
        "**Как пользоваться:**\n"
        "Просто напишите вопрос текстом, например:\n"
        "• «Какой срок исковой давности?»\n"
        "• «Что будет, если не заплатить налог вовремя?»\n"
        "• «Какие случаи освобождают от согласия на обработку ПДн?»\n\n"
        "**Команды:**\n"
        "/start — приветствие\n"
        "/help — эта справка\n"
        "/stats — сколько документов в базе\n"
        "/rebuild — пересоздать хранилище (после добавления новых документов)\n\n"
        "**Важно:**\n"
        "• Ответы основаны ТОЛЬКО на загруженных документах\n"
        "• Если информации нет в базе, я так и скажу\n"
        "• В ответе будут указаны номера статей и пунктов"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stats — статистика базы знаний"""
    await update.message.reply_text("📊 Собираю статистику...")
    
    loop = asyncio.get_event_loop()
    doc_count = await loop.run_in_executor(None, rag_pipeline.db.get_document_count)
    docs = rag_pipeline.db.get_all_documents()
    
    stats_text = f"📊 **Статистика базы знаний**\n\n"
    stats_text += f"📁 Всего документов: **{doc_count}**\n"
    
    if rag_pipeline.vectorstore is not None:
        chunks_count = rag_pipeline.vectorstore._collection.count()
        stats_text += f"🔪 Всего чанков: **{chunks_count}**\n\n"
    else:
        stats_text += f"️ Векторное хранилище не загружено\n\n"
    
    stats_text += "**Список документов:**\n"
    for doc in docs:
        size_kb = len(doc['content']) / 1024
        stats_text += f"• {doc['title']} ({size_kb:.1f} КБ)\n"
    
    if len(stats_text) > 4000:
        stats_text = stats_text[:4000] + "\n... (список обрезан)"
    
    await update.message.reply_text(stats_text, parse_mode='Markdown')


async def rebuild(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /rebuild — пересоздание векторного хранилища"""
    await update.message.reply_text(
        "🔄 Начинаю пересоздание векторного хранилища...\n"
        "Это может занять 1-2 минуты. Пожалуйста, подождите."
    )
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, rag_pipeline.force_rebuild)
        
        await update.message.reply_text(
            "✅ Векторное хранилище успешно пересоздано!\n"
            "Теперь можете задавать вопросы по обновлённым документам."
        )
    except Exception as e:
        await update.message.reply_text(f" Ошибка при пересоздании: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка обычных текстовых сообщений через RAG"""
    user_question = update.message.text.strip()
    
    if not user_question:
        await update.message.reply_text("Пожалуйста, задайте вопрос текстом.")
        return
    
    await update.message.reply_text("🤔 Ищу ответ в документах...")
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            rag_pipeline.ask_with_sources, 
            user_question
        )
        
        answer = result['answer']
        response = f"💡 {answer}\n"
        
        if result['sources_count'] > 0:
            response += f"\n📚 **Источники ({result['sources_count']}):**\n"
            for i, source in enumerate(result['sources'], 1):
                response += f"{i}. {source['title']}\n"
        else:
            response += "\n⚠️ Подходящих источников не найдено."
        
        if len(response) > 4000:
            response = response[:4000] + "\n... (ответ обрезан)"
        
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        await update.message.reply_text(
            f"❌ Произошла ошибка при обработке вопроса.\n"
            f"Попробуйте переформулировать или используйте /rebuild."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ошибок"""
    logger.warning(f'Update {update} caused error: {context.error}')
    if update and update.message:
        await update.message.reply_text(
            "❌ Произошла внутренняя ошибка. Попробуйте позже."
        )


def main():
    """Запуск бота"""
    # Получаем токен из переменных окружения
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not TOKEN:
        print("❌ Ошибка: токен бота не найден!")
        print("1. Создайте файл .env в корне проекта")
        print("2. Добавьте в него строку: TELEGRAM_BOT_TOKEN=ваш_токен")
        print("3. Получите токен у @BotFather в Telegram")
        print("\nПример .env файла:")
        print("TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ")
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