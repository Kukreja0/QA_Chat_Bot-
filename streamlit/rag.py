# =============================================================================
#  RAG PIPELINE
#  Three classes with single responsibilities:
#    DocumentProcessor  — ingest, chunk, embed, index
#    SessionMemory      — sliding-window conversation history
#    RAGEngine          — retrieval + prompt assembly + LLM call
# =============================================================================

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,   # Gemini embedding model
)
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------------------------------------------------------
#  1. DocumentProcessor
# -----------------------------------------------------------------------------
class DocumentProcessor:
    """
    Handles the ingestion half of the RAG pipeline.

    Steps:
        load_file() / load_directory()  →  RecursiveCharacterTextSplitter
        →  GoogleGenerativeAIEmbeddings  →  FAISS index

    Each document chunk is tagged with its source filename in metadata so the
    retriever can surface provenance alongside the answer.
    """

    LOADER_MAP = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
        ".md":   TextLoader,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retriever_k: int = 4,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k   = retriever_k

        self.documents:    list[Document] = []
        self.chunks:       list[Document] = []
        self.embeddings:   Optional[GoogleGenerativeAIEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.loaded_files: list[str] = []

    # -- Loading -------------------------------------------------------------- #

    def load_file(self, file_path: str) -> list[Document]:
        """Load a single supported file and append its pages/sections."""
        path   = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.LOADER_MAP:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {list(self.LOADER_MAP.keys())}"
            )

        loader = self.LOADER_MAP[suffix](str(path))
        docs   = loader.load()

        # Tag each page/section with the source filename
        for doc in docs:
            doc.metadata["source_file"] = path.name

        self.documents.extend(docs)
        if path.name not in self.loaded_files:
            self.loaded_files.append(path.name)

        return docs

    def load_directory(self, dir_path: str) -> list[Document]:
        """Recursively load all supported files from a folder."""
        all_docs = []
        for file in Path(dir_path).rglob("*"):
            if file.is_file() and file.suffix.lower() in self.LOADER_MAP:
                try:
                    all_docs.extend(self.load_file(str(file)))
                    print(f"  Loaded: {file.name}")
                except Exception as exc:
                    print(f"  Error loading {file.name}: {exc}")
        return all_docs

    # -- Chunking ------------------------------------------------------------- #

    def _split(self) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators    = ["\n\n", "\n", " ", ""],
        )
        self.chunks = splitter.split_documents(self.documents)
        return self.chunks

    # -- Embeddings ---------------------------------------------------------- #

    # def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
    #     """
    #     Returns a cached GoogleGenerativeAIEmbeddings instance.
    #     Model: gemini-embedding-001  (1536-dim, best accuracy in Gemini family).
    #     """
    #     if self.embeddings is None:
    #         api_key = os.getenv("GEMINI_API_KEY")
    #         if not api_key:
    #             raise EnvironmentError("GEMINI_API_KEY not set. Run Cell 2 first.")
    #         self.embeddings = GoogleGenerativeAIEmbeddings(
    #             model = "gemini-embedding-001",
    #             task_type="retrieval_document" ,
    #             google_api_key = api_key,
    #         )
    #     return self.embeddings

    def _get_embeddings(self):
        """
        Returns a cached HuggingFace embedding model.
        """
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
        return self.embeddings

    # -- Vector Store --------------------------------------------------------- #


    def build_vector_store(self) -> FAISS:
        """
        Split loaded documents into chunks, embed them with Gemini, and
        build a FAISS index for fast similarity search.
        """
        if not self.documents:
            raise ValueError(
                "No documents loaded. Call load_file() or load_directory() first."
            )
        chunks     = self._split()
        embeddings = self._get_embeddings()
        print(f"  Indexing {len(chunks)} chunks...")
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        print("  Vector store ready.")
        return self.vector_store

    def add_documents(self, new_docs: list[Document]) -> None:
        """Incrementally add documents to an existing FAISS index."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        new_chunks = splitter.split_documents(new_docs)
        self.chunks.extend(new_chunks)
        embeddings = self._get_embeddings()

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(new_chunks, embeddings)
        else:
            self.vector_store.add_documents(new_chunks)

    def get_retriever(self):
        """Return a LangChain retriever ready for use in the RAG chain."""
        if self.vector_store is None:
            raise ValueError("Vector store not built yet. Call build_vector_store().")
        return self.vector_store.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": self.retriever_k},
        )


# -----------------------------------------------------------------------------
#  2. SessionMemory
# -----------------------------------------------------------------------------
class SessionMemory:
    """
    Sliding-window conversation history.

    Stores the full message list in an InMemoryChatMessageHistory (LangChain
    Core — no deprecated langchain.memory dependency). Only the last `window_k`
    human/AI exchange pairs are injected into the prompt to keep token counts
    predictable while still supporting follow-up questions.

    history  — raw list of dicts used by the UI renderer
    """

    def __init__(self, window_k: int = 5):
        self.window_k = window_k
        self._store   = InMemoryChatMessageHistory()
        self.history: list[dict] = []

    def add(self, question: str, answer: str) -> None:
        """Append a human/AI exchange to the history."""
        self._store.add_user_message(question)
        self._store.add_ai_message(answer)
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": answer})

    def get_window_string(self):
        messages = self._store.messages[-self.window_k * 2:]

        history = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                history.append(f"User: {msg.content}")

            elif isinstance(msg, AIMessage):
                history.append(f"Assistant: {msg.content}")

        return "\n".join(history)

    def clear(self) -> None:
        """Reset history for a new chat session."""
        self._store.clear()
        self.history.clear()


# -----------------------------------------------------------------------------
#  3. RAGEngine
# -----------------------------------------------------------------------------
class RAGEngine:
    """
    Orchestrates the full retrieval-augmented generation cycle.

    On each call to ask():
        1. Retrieve top-K relevant chunks from FAISS.
        2. Assemble a prompt: conversation window + retrieved context + question.
        3. Call Gemini 1.5 Flash and return the answer with source provenance.

    The LLM is instantiated once at construction time (not on every call) to
    avoid repeated authentication overhead.
    """

    PROMPT = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
You are a knowledgeable assistant with access to a private document library.

RULES:
1. Answer using ONLY the information in the CONTEXT section.
2. If the context does not contain enough information to answer, say:
   "The uploaded documents do not cover this topic."
   Then answer from your general knowledge, clearly prefixed with:
   "Based on general knowledge:"
3. Keep answers clear, well-structured, and factual.
4. When helpful, mention which document the information comes from.

CONVERSATION HISTORY:
{chat_history}

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:


""",
    )

    def __init__(
        self,
        retriever,
        memory:      SessionMemory,
        model:       str   = "gemini-3.5-flash",
        temperature: float = 0.3,
    ):
        self.retriever = retriever
        self.memory    = memory

        from dotenv import load_dotenv

        # Priority 1: .env file in the project root
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set. Run Cell 2 first.")

        self.llm = ChatGoogleGenerativeAI(
            model          = model,
            google_api_key = api_key,
            temperature    = temperature,
        )

    def ask(self, question: str) -> dict:
        """
        Run the full RAG cycle and return a dict with keys:
            answer  — the LLM's response string
            sources — list of unique source filenames cited
            context — raw retrieved context (for debugging)
        """
        # Step 1: Retrieve
        retrieved = self.retriever.invoke(question)

        context_parts = []
        source_set = set()
        for doc in retrieved:
            source_file = doc.metadata.get('source_file', 'unknown')
            page_number = doc.metadata.get('page')

            if page_number is not None:
                context_parts.append(f"[Source: {source_file}, Page: {page_number}]\n{doc.page_content}")
                source_set.add(f"{source_file} (Page {page_number})")
            else:
                context_parts.append(f"[Source: {source_file}]\n{doc.page_content}")
                source_set.add(source_file)

        context = "\n\n".join(context_parts)
        sources = list(source_set)

        # Step 2: Assemble prompt
        filled = self.PROMPT.invoke({
            "chat_history": self.memory.get_window_string(),
            "context":      context,
            "question":     question,
        })

        # Step 3: Generate
        response = self.llm.invoke(filled)
        answer   = response.content[0]['text']

        if "The uploaded documents do not cover this topic." in answer:
          sources = None

        # Step 4: Persist to memory
        self.memory.add(question, answer)

        return {"answer": answer, "sources": sources, "context": context}

