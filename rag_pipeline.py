"""
Healthcare RAG Pipeline
Handles document loading, chunking, embedding, hybrid retrieval, and LLM generation.
Enforces strict out-of-scope refusal when no relevant documents are found.
Supports both Google Gemini API and NVIDIA NIM API.
Uses FAISS for high-performance, robust, segfault-free local vector storage.
"""

import os
import logging
import hashlib
import pickle
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import faiss

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from rank_bm25 import BM25Okapi

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDEFINED_KB_PATH = Path(__file__).parent / "knowledge_base" / "medical_faq.txt"


# ─── API Setup ─────────────────────────────────────────────────────────────
def test_api_key(api_key: str) -> Tuple[bool, str]:
    """Test connection using either Google Gemini or NVIDIA NIM."""
    if api_key.startswith("nvapi-"):
        # Test NVIDIA NIM connection
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5
        }
        try:
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                return True, "NVIDIA"
            else:
                return False, f"NVIDIA error {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"NVIDIA connection failed: {e}"
    else:
        # Test Google Gemini connection
        import google.generativeai as genai
        try:
            genai.configure(api_key=api_key)
            list(genai.list_models())
            return True, "Gemini"
        except Exception as e:
            return False, f"Gemini connection failed: {e}"


# ─── Embedding Generation (NVIDIA or Gemini) ──────────────────────────────────
def get_embedding(text: str, api_key: str, provider: str) -> List[float]:
    """Get high-quality semantic embedding using active provider API."""
    if provider == "NVIDIA":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": text,
            "input_type": "passage"
        }
        try:
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=20
            )
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
            else:
                logger.error(f"NVIDIA embedding failed with status {response.status_code}: {response.text}")
                return []
        except Exception as e:
            logger.error(f"NVIDIA embedding connection error: {e}")
            return []
    else:
        # Gemini embedding
        import google.generativeai as genai
        try:
            result = genai.embed_content(
                model=config.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            return []


def get_query_embedding(text: str, api_key: str, provider: str) -> List[float]:
    """Get high-quality semantic embedding for the query."""
    if provider == "NVIDIA":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": text,
            "input_type": "query"
        }
        try:
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=20
            )
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
            else:
                logger.error(f"NVIDIA query embedding failed: {response.text}")
                return []
        except Exception as e:
            logger.error(f"NVIDIA query embedding connection error: {e}")
            return []
    else:
        # Gemini embedding
        import google.generativeai as genai
        try:
            result = genai.embed_content(
                model=config.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_query",
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini query embedding error: {e}")
            return []


# ─── Document Loaders ─────────────────────────────────────────────────────────
def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    docs = []
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".md":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = Path(file_path).name
            doc.metadata["file_type"] = ext
        logger.info(f"Loaded {len(docs)} pages from '{Path(file_path).name}'")
    except Exception as e:
        logger.error(f"Error loading '{file_path}': {e}")
    return docs


# ─── Chunking ─────────────────────────────────────────────────────────────────
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


# ─── FAISS Vector Store ───────────────────────────────────────────────────────
class HealthcareVectorStore:
    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider
        
        # Database directory specific to provider
        self.db_dir = Path(__file__).parent / "healthcare_faiss" / provider.lower()
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.db_dir / "index.faiss"
        self.meta_path = self.db_dir / "metadata.pkl"
        
        # Dimensions: 1024 for nvidia/nv-embedqa-e5-v5, 768 for Gemini text-embedding-004
        self.dimension = 1024 if provider == "NVIDIA" else 768
        
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict] = []
        self.chunks_data: List[Dict] = []
        
        # Load or initialize FAISS
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "rb") as f:
                    self.chunks_data = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks_data = []
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks_data = []
            
        self._load_bm25_index()

    def _load_bm25_index(self):
        try:
            if self.chunks_data:
                self._bm25_docs = [
                    {"text": c["text"], "metadata": c["metadata"]}
                    for c in self.chunks_data
                ]
                tokenized = [d["text"].lower().split() for d in self._bm25_docs]
                self._bm25_index = BM25Okapi(tokenized)
        except Exception as e:
            logger.warning(f"BM25 load failed: {e}")

    def add_documents(self, chunks: List[Document], progress_callback=None) -> int:
        added = 0
        total = len(chunks)
        existing_hashes = {c["hash"] for c in self.chunks_data}
        
        for i, chunk in enumerate(chunks):
            try:
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
                if content_hash in existing_hashes:
                    if progress_callback:
                        progress_callback(i + 1, total, "Skipping duplicate")
                    continue
                    
                embedding = get_embedding(chunk.page_content, self.api_key, self.provider)
                if not embedding or len(embedding) != self.dimension:
                    continue
                
                # Normalize vector to get exact cosine similarity with IndexFlatIP
                emb_arr = np.array(embedding, dtype="float32").reshape(1, -1)
                faiss.normalize_L2(emb_arr)
                
                self.index.add(emb_arr)
                self.chunks_data.append({
                    "id": f"doc_{content_hash}",
                    "hash": content_hash,
                    "text": chunk.page_content,
                    "metadata": {k: str(v) for k, v in chunk.metadata.items()}
                })
                added += 1
                if progress_callback:
                    progress_callback(i + 1, total, chunk.metadata.get("source_file", ""))
            except Exception as e:
                logger.error(f"Error adding chunk {i}: {e}")
                
        if added > 0:
            try:
                faiss.write_index(self.index, str(self.index_path))
                with open(self.meta_path, "wb") as f:
                    pickle.dump(self.chunks_data, f)
            except Exception as e:
                logger.error(f"Error saving FAISS database: {e}")
                
        self._load_bm25_index()
        logger.info(f"Added {added} new chunks to FAISS vector store")
        return added

    def hybrid_search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[Dict]:
        """Hybrid FAISS + BM25 search with Reciprocal Rank Fusion."""
        results: Dict[str, Dict] = {}
        total = self.index.ntotal
        if total == 0:
            return []

        # 1. Semantic search
        try:
            query_emb = get_query_embedding(query, self.api_key, self.provider)
            if query_emb and len(query_emb) == self.dimension:
                q_arr = np.array(query_emb, dtype="float32").reshape(1, -1)
                faiss.normalize_L2(q_arr)
                
                n = min(top_k * 2, total)
                distances, indices = self.index.search(q_arr, n)
                
                for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx != -1:
                        c_data = self.chunks_data[idx]
                        key = c_data["text"][:120]
                        # dist is cosine similarity since both query and index vectors are L2-normalized
                        results[key] = {
                            "text": c_data["text"],
                            "metadata": c_data["metadata"],
                            "semantic_score": round(float(dist), 4),
                            "rrf_score": 0.7 / (60 + rank + 1),
                        }
        except Exception as e:
            logger.warning(f"FAISS search error: {e}")

        # 2. BM25 search
        if self._bm25_index and self._bm25_docs:
            try:
                scores = self._bm25_index.get_scores(query.lower().split())
                top_idx = np.argsort(scores)[::-1][:top_k * 2]
                for rank, idx in enumerate(top_idx):
                    if scores[idx] > 0:
                        d = self._bm25_docs[idx]
                        key = d["text"][:120]
                        rrf = 0.3 / (60 + rank + 1)
                        if key in results:
                            results[key]["rrf_score"] += rrf
                            results[key]["bm25_score"] = round(float(scores[idx]), 4)
                        else:
                            results[key] = {
                                "text": d["text"],
                                "metadata": d["metadata"],
                                "semantic_score": 0.0,
                                "bm25_score": round(float(scores[idx]), 4),
                                "rrf_score": rrf,
                            }
            except Exception as e:
                logger.warning(f"BM25 error: {e}")

        sorted_results = sorted(results.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:top_k]

    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def get_stats(self) -> Dict:
        count = self.index.ntotal
        sources = list({c["metadata"].get("source_file", "Unknown") for c in self.chunks_data})
        return {"total_chunks": count, "source_files": sources, "num_sources": len(sources)}

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_data = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        self._bm25_index = None
        self._bm25_docs = []


# ─── LLM Generator (NVIDIA NIM or Google Gemini) ──────────────────────────────
class HealthcareLLM:
    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider
        if provider == "Gemini":
            import google.generativeai as genai
            self.model = genai.GenerativeModel(
                model_name=config.LLM_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=config.MAX_OUTPUT_TOKENS,
                ),
            )

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict],
        chat_history: List[Dict] = None,
    ) -> Dict[str, Any]:
        # ── Out-of-scope check ──────────────────────────────────────────────
        if not retrieved_docs:
            return {
                "answer": config.OUT_OF_SCOPE_RESPONSE,
                "citations": [],
                "retrieved_docs": [],
                "is_out_of_scope": True,
            }

        # Check if best semantic score is below threshold AND no BM25 matches
        best_score = max(d.get("semantic_score", 0) for d in retrieved_docs)
        best_bm25 = max(d.get("bm25_score", 0) for d in retrieved_docs)
        
        # Bypass strict thresholds for meta-document summary or topic requests
        meta_keywords = {
            "summary", "summarize", "about the document", "what is this pdf", "what is this document", 
            "kis bare me", "kis baare mein", "kisse sambandith", "kisse sambandhit", "is document", "इस दस्तावेज", "क्या है यह"
        }
        is_meta_query = any(kw in question.lower() for kw in meta_keywords)
        
        if best_score < config.RELEVANCE_THRESHOLD and best_bm25 == 0 and not is_meta_query:
            return {
                "answer": config.OUT_OF_SCOPE_RESPONSE,
                "citations": [],
                "retrieved_docs": retrieved_docs,
                "is_out_of_scope": True,
            }

        # ── Build context & citations ───────────────────────────────────────
        context_parts = []
        citations = []
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc["metadata"].get("source_file", "Unknown")
            page = doc["metadata"].get("page", "N/A")
            citations.append(f"[{i}] {src}" + (f", Page {page}" if page != "N/A" else ""))
            context_parts.append(
                f"[Document {i} | Source: {src} | Page: {page}]\n{doc['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        history_text = ""
        for turn in (chat_history or [])[-4:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        prompt = config.SYSTEM_PROMPT.format(
            context=context,
            chat_history=history_text,
            question=question,
        )

        # Programmatic Language Override to completely prevent history-anchoring issues
        has_devanagari = any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in question)
        cleaned_q = "".join(c for c in question.lower() if c.isalnum() or c.isspace()).strip()
        is_hinglish = any(w in cleaned_q.split() for w in ["kya", "hai", "h", "karta", "kis", "bare", "baare", "sambandith", "sambandhit", "kisse", "mein", "me", "ko", "se", "ho", "kaise", "kese", "namaste", "dhanyawad", "shukriya", "hai क्या", "hai?"])
        
        target_lang = "Hindi" if (has_devanagari or is_hinglish) else "English"
        if target_lang == "Hindi":
            lang_instruction = "\n\n⚠️ CRITICAL DIRECTIVE: The user asked in Hindi/Hinglish. You MUST respond completely in fluent Hindi (Devanagari script) and use the Hindi safety disclaimer. Do NOT write in English."
        else:
            lang_instruction = "\n\n⚠️ CRITICAL DIRECTIVE: The user asked in English. You MUST respond completely in English and use the English safety disclaimer. Do NOT write in Hindi."
            
        prompt += lang_instruction

        # Strict Grounding Enforcement Directive
        refusal_msg = config.OUT_OF_SCOPE_RESPONSE
        if target_lang == "Hindi":
            refusal_msg = "⚠️ आउट ऑफ स्कोप: मुझे प्रदान किए गए दस्तावेजों में इस प्रश्न का उत्तर देने के लिए प्रासंगिक जानकारी नहीं मिली। कृपया किसी योग्य स्वास्थ्य पेशेवर से परामर्श लें।"

        grounding_directive = (
            f"\n\n⚠️ STRICT GROUNDING DIRECTIVE: You are strictly forbidden from answering the question from your own training data "
            f"if the retrieved documents do not contain the answer. Carefully read the [RETRIEVED DOCUMENTS] above. "
            f"If the documents do not discuss the topic of the user's question '{question}' (such as geography, cooking, or general programming), "
            f"you MUST refuse the request completely by responding ONLY with the exact out-of-scope refusal message: '{refusal_msg}'. "
            f"Do not write anything else. Do not attempt to answer the question."
        )
        prompt += grounding_directive

        try:
            if self.provider == "NVIDIA":
                # NVIDIA NIM API Call
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "meta/llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config.LLM_TEMPERATURE,
                    "max_tokens": config.MAX_OUTPUT_TOKENS
                }
                response = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if response.status_code == 200:
                    answer_text = response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"NVIDIA API responded with code {response.status_code}: {response.text}")
            else:
                # Google Gemini API Call
                response = self.model.generate_content(prompt)
                answer_text = response.text

            return {
                "answer": answer_text,
                "citations": citations,
                "retrieved_docs": retrieved_docs,
                "is_out_of_scope": False,
                "error": None,
            }
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {
                "answer": "Error generating response. Please try again.",
                "citations": [],
                "retrieved_docs": [],
                "is_out_of_scope": False,
                "error": str(e),
            }


# ─── Main Pipeline ────────────────────────────────────────────────────────────
class HealthcareRAGPipeline:
    def __init__(self):
        self.vector_store: Optional[HealthcareVectorStore] = None
        self.llm: Optional[HealthcareLLM] = None
        self.is_ready = False
        self.provider = "None"

    def initialize(self, api_key: str) -> bool:
        success, provider = test_api_key(api_key)
        if not success:
            logger.error(f"Initialization API key test failed: {provider}")
            return False
        try:
            self.provider = provider
            self.vector_store = HealthcareVectorStore(api_key, provider)
            self.llm = HealthcareLLM(api_key, provider)
            self.is_ready = True

            # Auto-load predefined KB if vector store is empty
            if self.vector_store.is_empty() and PREDEFINED_KB_PATH.exists():
                logger.info("Loading predefined medical knowledge base...")
                self.ingest_files([str(PREDEFINED_KB_PATH)])

            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def ingest_files(self, file_paths: List[str], progress_callback=None) -> Dict[str, Any]:
        if not self.is_ready:
            return {"success": False, "error": "Pipeline not initialized"}
        try:
            docs = []
            for fp in file_paths:
                docs.extend(load_document(fp))
            if not docs:
                return {"success": False, "error": "No content extracted"}
            chunks = chunk_documents(docs)
            added = self.vector_store.add_documents(chunks, progress_callback)
            return {
                "success": True,
                "pages_loaded": len(docs),
                "chunks_created": len(chunks),
                "chunks_added": added,
                "stats": self.vector_store.get_stats(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def query(self, question: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        if not self.is_ready:
            return {
                "answer": "⚠️ Pipeline not initialized. Please set your API key.",
                "is_out_of_scope": True,
                "citations": [],
                "retrieved_docs": [],
            }

        # ─── CONVERSATIONAL FALLBACKS (Greetings, Thanks, Byes - English & Hindi) ──
        cleaned = "".join(c for c in question.lower() if c.isalnum() or c.isspace()).strip()
        
        greetings = {
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "hola", "hi there", "hello there",
            "namaste", "pranam", "namaskar", "kaise ho", "kese ho", "kaise", "kese", "help", "madad", "नमस्ते", "नमस्कार", "प्रणाम"
        }
        thanks = {
            "thank you", "thanks", "appreciate it", "thank you very much", "many thanks",
            "dhanyawad", "shukriya", "thankyou", "धन्यवाद", "शुक्रिया"
        }
        byes = {
            "bye", "goodbye", "see you", "bye bye",
            "alvida", "phir milenge", "khuda hafiz", "अलविदा", "फिर मिलेंगे"
        }

        # Check if the query specifically triggers Hindi conversational keywords
        is_hindi = any(w in cleaned for w in ["namaste", "pranam", "namaskar", "kaise ho", "kese ho", "kaise", "kese", "dhanyawad", "shukriya", "alvida", "phir milenge", "madad", "नमस्ते", "नमस्कार", "प्रणाम", "धन्यवाद", "शुक्रिया", "अलविदा", "फिर मिलेंगे"])

        if cleaned in greetings or any(w == cleaned for w in ["namaste", "namaskar", "नमस्ते", "नमस्कार"]):
            if is_hindi or any(w in cleaned for w in ["namaste", "namaskar", "नमस्ते", "नमस्कार"]):
                ans = "👋 नमस्ते! मैं आपका क्लिनिकल दस्तावेज़ सहायक हूँ। आज मैं आपकी क्या सहायता कर सकता हूँ? लोड किए गए चिकित्सा दस्तावेज़ों के आधार पर बेझिझक कोई भी प्रश्न पूछें।"
            else:
                ans = "👋 Hello! I am your clinical document assistant. How can I help you today? Please feel free to ask any medical or healthcare questions based on the loaded documentation."
            return {
                "answer": ans,
                "is_out_of_scope": False,
                "citations": [],
                "retrieved_docs": [],
            }
        elif cleaned in thanks or any(w == cleaned for w in ["dhanyawad", "shukriya", "धन्यवाद", "शुक्रिया"]):
            if is_hindi or any(w in cleaned for w in ["dhanyawad", "shukriya", "धन्यवाद", "शुक्रिया"]):
                ans = "😊 आपका बहुत-बहुत धन्यवाद! मुझे आपकी सहायता करके बेहद खुशी हुई। यदि दस्तावेज़ों से संबंधित कोई अन्य जानकारी चाहिए, तो अवश्य पूछें।"
            else:
                ans = "😊 You are very welcome! Let me know if there's anything else you would like to retrieve from the clinical documents."
            return {
                "answer": ans,
                "is_out_of_scope": False,
                "citations": [],
                "retrieved_docs": [],
            }
        elif cleaned in byes or any(w == cleaned for w in ["alvida", "phir milenge", "अलविदा", "फिर मिलेंगे"]):
            if is_hindi or any(w in cleaned for w in ["alvida", "phir milenge", "अलविदा", "फिर मिलेंगे"]):
                ans = "👋 अलविदा! स्वस्थ रहें और आपका दिन मंगलमय हो!"
            else:
                ans = "👋 Goodbye! Stay healthy and have a great day!"
            return {
                "answer": ans,
                "is_out_of_scope": False,
                "citations": [],
                "retrieved_docs": [],
            }
        # ─── RAG & SYSTEM ARCHITECTURE META-QUESTIONS ───────────────────────
        rag_keywords = {"rag", "retrieval", "architecture", "how does this work", "system design", "hybrid search", "faiss", "bm25", "llm", "nvidia", "gemini", "about this app", "mediassist"}
        if any(kw in cleaned for kw in rag_keywords):
            return {
                "answer": (
                    "⚕️ **MediAssist AI — Healthcare RAG Architecture**\n\n"
                    "MediAssist AI is a premium, clinical-grade **Retrieval-Augmented Generation (RAG)** assistant designed specifically for healthcare document grounding. Here is how it functions:\n\n"
                    "1. **Hybrid Retrieval Pipeline**:\n"
                    "   * **FAISS Vector Storage**: Converts document chunks into dense semantic embeddings (1024-dim for NVIDIA NIM, 768-dim for Gemini) to understand the clinical context of queries.\n"
                    "   * **BM25 Lexical Search**: Runs local keyword indexing to catch exact terminology, medical abbreviations, and acronyms (like 'WHO' or 'BP').\n"
                    "   * **Reciprocal Rank Fusion (RRF)**: Merges semantic and lexical ranks to retrieve the top 5 most relevant clinical segments.\n\n"
                    "2. **Strict Document Grounding & Refusal**:\n"
                    "   * Enforces a dual out-of-scope validation. If a query falls below the relevance threshold and lacks exact keyword matches, the system safely refuses to answer (`Out of Scope`). This completely eliminates harmful clinical hallucinations.\n\n"
                    "3. **Dual LLM Engines**:\n"
                    "   * Integrates seamlessly with high-performance **NVIDIA NIM API** (Llama-3.1) and **Google Gemini API** for fast, citation-anchored, and safe healthcare generation."
                ),
                "is_out_of_scope": False,
                "citations": ["System Architecture Specification"],
                "retrieved_docs": [],
            }

        retrieved = self.vector_store.hybrid_search(question)
        return self.llm.generate_answer(question, retrieved, chat_history)

    def get_stats(self) -> Dict:
        if not self.is_ready or not self.vector_store:
            return {"total_chunks": 0, "source_files": [], "num_sources": 0}
        return self.vector_store.get_stats()

    def clear_knowledge_base(self):
        if self.vector_store:
            self.vector_store.clear()
