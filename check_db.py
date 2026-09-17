from database import DocumentDatabase

db = DocumentDatabase()
docs = db.get_all_documents()

print(f"📊 Всего документов в SQLite БД: {len(docs)}\n")
for doc in docs:
    print(f"  • ID: {doc['id']}")
    print(f"    Title: {doc['title']}")
    print(f"    Source: {doc['source']}")
    print(f"    Размер: {len(doc['content'])} символов")
    print(f"    Добавлен: {doc['created_at']}\n")