import os
import shutil
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from database import DocumentDatabase
from typing import List, Any


class HybridRetriever(BaseRetriever):
    """Гибридный retriever: BM25 (точные слова) + Векторы (смысл)"""
    vector_retriever: Any
    bm25_retriever: Any
    k: int = 10
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        seen = set()
        combined = []
        for doc in vector_docs + bm25_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                combined.append(doc)
        
        return combined[:self.k]


class RAGPipeline:
    def __init__(self, db_path: str = "documents.db", persist_directory: str = "./chroma_db"):
        self.db = DocumentDatabase(db_path)
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model="bge-m3")
        self.llm = ChatOllama(model="qwen3.5:9b", temperature=0)
        self.vectorstore = None
        self.rag_chain = None
        self.last_doc_count = 0
        self.all_splits = []
        
        # При старте сразу пытаемся загрузить существующее хранилище
        self._try_load_existing_vectorstore()
    
    def _try_load_existing_vectorstore(self):
        """Загружает существующее хранилище с диска, если оно есть и актуально"""
        if not os.path.exists(self.persist_directory):
            print("ℹ️  Векторное хранилище не найдено на диске. Будет создано при первом запросе.")
            return
        
        try:
            self.vectorstore = Chroma(
                collection_name="legal_rag_collection",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            stored_count = self.vectorstore._collection.count()
            
            # Считаем, сколько чанков должно быть в БД
            docs = self.load_documents_from_db()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            self.all_splits = text_splitter.split_documents(docs)
            expected_chunks = len(self.all_splits)
            
            if stored_count == expected_chunks and stored_count > 0:
                # ✅ КРИТИЧЕСКИ ВАЖНО: устанавливаем last_doc_count, 
                # чтобы система не пересоздавала хранилище при каждом вопросе
                self.last_doc_count = len(docs)
                print(f"✅ Загружено существующее векторное хранилище ({stored_count} чанков)")
                print("⚡ Пересоздание не требуется — используем кэш с диска")
            else:
                print(f"🔄 Хранилище устарело (в БД: {stored_count} чанков, нужно: {expected_chunks})")
                print("   Будет пересоздано при первом запросе.")
                self.vectorstore = None
                self.all_splits = []
                self.last_doc_count = 0
                
        except Exception as e:
            print(f"⚠️  Не удалось загрузить хранилище: {e}")
            self.vectorstore = None
            self.all_splits = []
            self.last_doc_count = 0

    def load_documents_from_db(self) -> list[Document]:
        db_docs = self.db.get_all_documents()
        langchain_docs = []
        for doc in db_docs:
            langchain_docs.append(
                Document(
                    page_content=doc['content'],
                    metadata={
                        "title": doc['title'],
                        "source": doc['source'],
                        "doc_id": doc['id'],
                        "created_at": doc['created_at']
                    }
                )
            )
        return langchain_docs

    def build_vectorstore(self, force_rebuild: bool = False):
        """Создание или обновление векторного хранилища"""
        if not force_rebuild and self.vectorstore is not None:
            return

        docs = self.load_documents_from_db()
        if not docs:
            raise ValueError("В БД нет документов. Сначала запустите populate_db.py")

        print(f"📚 Загружено {len(docs)} документов из БД")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ""]
        )
        self.all_splits = text_splitter.split_documents(docs)
        print(f"🔪 Текст разбит на {len(self.all_splits)} чанков")
        print("🔄 Создание векторного хранилища (это может занять 1-2 минуты)...")

        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        self.vectorstore = Chroma(
            collection_name="legal_rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        batch_size = 50
        total_splits = len(self.all_splits)
        
        for i in range(0, total_splits, batch_size):
            batch = self.all_splits[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            print(f"   Обработано {min(i + batch_size, total_splits)} из {total_splits} чанков...")

        # ✅ Устанавливаем last_doc_count, чтобы не пересоздавать при каждом вопросе
        self.last_doc_count = len(docs)
        print("✅ Векторное хранилище успешно создано/обновлено")

    def _get_retriever(self, k: int = 5):  # Уменьшено с 10 до 5 для скорости
        if self.vectorstore is None:
            self.build_vectorstore()

        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        bm25_retriever = BM25Retriever.from_documents(self.all_splits)
        bm25_retriever.k = k

        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            k=k
        )

    def build_rag_chain(self):
        if self.vectorstore is None:
            self.build_vectorstore()

        retriever = self._get_retriever(k=5)

        template = """Ты — профессиональный юридический помощник. Отвечай строго на основе предоставленных ниже текстов нормативных актов.

Правила:
1. Давай точный, структурированный и понятный ответ.
2. Если в тексте есть номера статей, пунктов или частей, ОБЯЗАТЕЛЬНО указывай их в ответе (например: "Согласно пункту 2 статьи 5...").
3. Если есть исключения из правила, упомяни их.
4. НИКОГДА не используй фразы: "в предоставленном контексте", "согласно документу", "в тексте сказано". Отвечай утвердительно, как эксперт.
5. Если информации для ответа в предоставленных фактах нет, так и скажи: "В предоставленных документах нет информации по этому вопросу".

Факты:
{context}

Вопрос: {question}

Ответ:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            if not docs:
                return "Информация не найдена."
            formatted = []
            for doc in docs:
                title = doc.metadata.get('title', 'Неизвестный документ')
                formatted.append(f"[{title}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("✅ RAG-цепочка собрана")

    def ask(self, question: str) -> str:
        # ✅ Убрана проверка обновлений перед каждым вопросом
        return self.rag_chain.invoke(question)

    def ask_with_sources(self, question: str) -> dict:
        # ✅ Убрана проверка обновлений перед каждым вопросом
        retriever = self._get_retriever(k=5)
        docs = retriever.invoke(question)
        
        answer = self.rag_chain.invoke(question)

        unique_sources = {}
        for doc in docs:
            title = doc.metadata.get('title', 'Без названия')
            source = doc.metadata.get('source', 'неизвестно')
            key = f"{title}_{source}"
            if key not in unique_sources:
                unique_sources[key] = {
                    "title": title,
                    "source": source,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }

        sources = list(unique_sources.values())
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "sources_count": len(sources)
        }
    
    def force_rebuild(self):
        """Принудительная перестройка хранилища (вызывать вручную при добавлении документов)"""
        print("🔄 Принудительная перестройка векторного хранилища...")
        self.vectorstore = None
        self.rag_chain = None
        self.last_doc_count = 0
        self.all_splits = []
        self.build_vectorstore(force_rebuild=True)
        self.build_rag_chain()