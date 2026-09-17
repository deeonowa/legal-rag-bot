import sqlite3
from typing import List, Dict, Optional
from datetime import datetime

class DocumentDatabase:
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Создание таблиц БД"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def add_document(self, title: str, content: str, source: str = None, metadata: str = None) -> int:
        """Добавление документа в БД. Возвращает ID документа."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO documents (title, content, source, metadata) VALUES (?, ?, ?, ?)",
                (title, content, source, metadata)
            )
            conn.commit()
            return cursor.lastrowid
    
    def add_documents_batch(self, documents: List[Dict]) -> List[int]:
        """Пакетное добавление документов"""
        ids = []
        with sqlite3.connect(self.db_path) as conn:
            for doc in documents:
                cursor = conn.execute(
                    "INSERT INTO documents (title, content, source, metadata) VALUES (?, ?, ?, ?)",
                    (doc['title'], doc['content'], doc.get('source'), doc.get('metadata'))
                )
                ids.append(cursor.lastrowid)
            conn.commit()
        return ids
    
    def get_all_documents(self) -> List[Dict]:
        """Получение всех документов из БД"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Получение документа по ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_document_by_source(self, source: str) -> Optional[Dict]:
        """Получение документа по источнику (имени файла)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM documents WHERE source = ?", (source,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def search_documents(self, query: str) -> List[Dict]:
        """Простой поиск по содержимому (без векторного поиска)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE content LIKE ? OR title LIKE ?",
                (f"%{query}%", f"%{query}%")
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_document_count(self) -> int:
        """Количество документов в БД"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]
    
    def delete_document(self, doc_id: int) -> bool:
        """Удаление документа по ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0