from database import DocumentDatabase
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_documents_from_folder(folder_path: str = "./my_docs"):
    """Загрузка документов из папки с проверкой дубликатов"""
    db = DocumentDatabase()
    
    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"❌ Папка {folder_path} не найдена")
        return
    
    # Поддерживаемые расширения
    supported_extensions = {'.txt', '.md', '.pdf'}
    
    # Получаем список уже загруженных файлов
    existing_docs = db.get_all_documents()
    existing_sources = {doc['source'] for doc in existing_docs}
    
    print(f"📂 Сканирование папки: {folder_path}")
    print(f"📊 Уже загружено документов: {len(existing_sources)}")
    
    added_count = 0
    skipped_count = 0
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Пропускаем папки
        if not os.path.isfile(filepath):
            continue
        
        # Получаем расширение
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # Проверяем, поддерживается ли расширение
        if ext not in supported_extensions:
            print(f"⏭️  Пропущен (неподдерживаемый формат): {filename}")
            continue
        
        # Проверяем, загружен ли уже этот файл
        if filename in existing_sources:
            print(f"⏭️  Уже загружен: {filename}")
            skipped_count += 1
            continue
        
        # Извлекаем название без расширения
        title = os.path.splitext(filename)[0]
        
        # Загружаем содержимое в зависимости от типа файла
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                content = "\n\n".join([page.page_content for page in pages])
            else:  # .txt или .md
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            
            # Добавляем в БД
            doc_id = db.add_document(
                title=title,
                content=content,
                source=filename
            )
            
            print(f"✅ Добавлен: {filename} → ID: {doc_id}")
            added_count += 1
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке {filename}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📈 Итого:")
    print(f"   ✅ Добавлено: {added_count}")
    print(f"   ⏭️  Пропущено: {skipped_count}")
    print(f"   📚 Всего в БД: {db.get_document_count()}")
    print("=" * 60)

if __name__ == "__main__":
    # Можно указать путь к папке или использовать ./my_docs по умолчанию
    folder = "./my_docs"
    
    load_documents_from_folder(folder)