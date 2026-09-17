from rag_pipeline import RAGPipeline

def main():
    print("🚀 Инициализация RAG-системы...")

    rag = RAGPipeline(
        db_path="documents.db",
        persist_directory="./chroma_db"
    )

    rag.build_rag_chain()

    print("\n💬 RAG-система готова! Задавайте вопросы (для выхода введите 'exit')")
    print("💡 Команды: 'rebuild' — пересоздать хранилище, 'exit' — выход")
    print("-" * 60)

    while True:
        try:
            question = input("\n❓ Ваш вопрос: ").strip()

            if question.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break
            
            if question.lower() == 'rebuild':
                print("\n🔄 Пересоздание хранилища...")
                rag.force_rebuild()
                print("✅ Готово! Можете задавать вопросы.")
                continue

            if not question:
                continue

            print("\n🤔 Думаю...")
            result = rag.ask_with_sources(question)

            print(f"\n💡 {result['answer']}")

            if result['sources_count'] > 0:
                print(f"\n📚 Источники ({result['sources_count']}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['title']}")
            else:
                print("\n⚠️  Подходящих источников не найдено")

        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

if __name__ == "__main__":
    main()