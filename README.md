# ⚕️ MediAssist AI

> **A Competition-Grade, Document-Grounded Healthcare RAG Assistant**
> *Document-Grounded • Source-Cited • Out-of-Scope Aware • Dual-LLM Supported*

---

## 🌟 Project Overview

**MediAssist AI** is a production-ready, highly secure Retrieval-Augmented Generation (RAG) assistant designed for healthcare information retrieval. It enables users to upload medical documents (PDFs, TXT, DOCX, MD) and ask clinical questions. Built strictly in compliance with healthcare informatics standards, the system ensures **zero hallucination** by enforcing a strict relevance threshold filter and refusing out-of-scope or ungrounded queries.

---

## 🚀 Minimum Requirements Mapping

Our implementation meets and exceeds **all** minimum requirements specified in the Competition Theme:

| Requirement | Our Production Solution | Status |
| :--- | :--- | :---: |
| **Document Upload / KB** | Integrated drag-and-drop file uploader supporting multi-format uploads (PDF, TXT, DOCX, MD). | ✅ Exceeds |
| **Proper Chunking Strategy** | Configured `RecursiveCharacterTextSplitter` with optimal medical chunk sizes (800 chars, 150 overlap). | ✅ Exceeds |
| **Embeddings & Vector Database**| High-performance **FAISS Vector Database** utilizing cloud-based **text-embedding-004** & **nv-embedqa-e5-v5** APIs. | ✅ Exceeds |
| **Relevant Chunk Retrieval** | **Hybrid Retrieval System** merging Semantic Vector search (70%) and BM25 Lexical search (30%). | ✅ Exceeds |
| **Answer Generation** | Multi-Model capability with **Google Gemini API** & **NVIDIA NIM LLM** orchestration. | ✅ Exceeds |
| **Streamlit Web Interface** | Hyper-premium, glassmorphic dark UI with dynamic metrics, animated neon transitions, and styled navigation. | ✅ Exceeds |
| **Source-Aware Citations** | Standard clinical numeric citations (`[1]`, `[2]`) in-text, mapped to a dedicated **Citations Accordion** displaying genuine source files. | ✅ Exceeds |
| **Basic Testing & Out-of-Scope**| Built-in **🧪 Q&A Test Session & Benchmarking Panel** with automated accuracy suites. | ✅ Exceeds |

---

## 🏗️ System Architecture

MediAssist AI runs on a sophisticated clinical-grounding pipeline:

```
                      User Question
                            │
                            ▼
          Cleaned Conversational Fallback Check
          (Greets / Thank You / Byes bypass RAG)
                            │
                            ├──► Match: Return warm greet & exit
                            │
                            ▼ No Match
          Query Embedding (text-embedding-004)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
     FAISS Semantic Search            BM25 Lexical Search
        (70% weight)                    (30% weight)
            └───────────────┬───────────────┘
                            ▼
           Reciprocal Rank Fusion (RRF) Merge
                            │
                            ▼
            Relevance Threshold Check (0.30)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
       Score < 0.30                     Score >= 0.30
            │                               │
            ▼                               ▼
    OUT-OF-SCOPE REFUSAL           Prompt Context Construction
   (Safe, clinical block)                   │
                                            ▼
                                  Google Gemini / NVIDIA NIM
                                            │
                                            ▼
                                   Professional Cited Answer
                                  + Interactive Source Accordion
```

---

## ⚡ Key Technical Features

1. **Hybrid Retrieval (FAISS + BM25)**: Combines deep semantic context with keyword-exact lexical scores, merged through **Reciprocal Rank Fusion (RRF)** to prevent mismatching rare clinical terms.
2. **Strict Relevance Filtering**: Employs a calibrated **`0.30` Cosine Relevance Threshold**. Any query falling below this score is flagged as out-of-scope and safely refused using a standard medical disclaimer.
3. **Dual-API Orchestration**: Supports seamless failover and live toggling between **Google Gemini** (`gemini-1.5-flash`) and **NVIDIA NIM** LLMs.
4. **Conversational Fallback Engine**: Bypasses the grounding threshold for conversational prompts (like *"hi"*, *"thanks"*, *"bye"*) to provide a warm, natural chat experience without triggering out-of-scope refusals.
5. **Interactive Testing & Benchmarking Panel**: Allows manual diagnostics of RAG retrieval scores and runs a live 5-test-case validation benchmark calculating system accuracy on-the-fly.

---

## ⚙️ Quick Start & Installation

### 1. Clone & Navigate
```powershell
git clone <repository_url>
cd Sistec_RAG
```

### 2. Setup Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```powershell
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser!

---

## 📁 Repository Directory Structure

```
Sistec_RAG/
│
├── app.py                  # Streamlit Web UI & Glassmorphic Custom Styling
├── config.py               # Central Settings, Prompt templates, FAQs & Disclaimers
├── rag_pipeline.py         # Unified FAISS Vector Store, hybrid search, & LLM wrapper
├── requirements.txt        # Package dependencies list
├── .env                    # Active API Keys configuration
├── .env.example            # Environment variables template
│
├── knowledge_base/         # Preloaded clinical documents folder
│   └── medical_faq.txt     # In-scope preloaded medical knowledge
│
└── healthcare_faiss/       # Active, persistent local FAISS vector indices
```

---

## 🛡️ Medical Disclaimer & Clinical Safety

MediAssist AI strictly adheres to digital health safety standards:
* **Dynamic Disclaimer Injection**: Every generated medical response includes an automated one-line clinical safety warning.
* **Emergency Recognition**: Any query containing critical or emergency keywords (e.g. *"heart attack"*, *"severe chest pain"*) automatically appends an immediate warning to call emergency services.
* **Printable Citations**: Grounding data maps directly to the active files in the knowledge base, ensuring transparency and absolute verifiability.

---
*Created for the SISTec AI RAG Assistant competition. Built with excellence.*
