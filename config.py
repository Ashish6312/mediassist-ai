"""
Healthcare RAG Configuration
Central configuration for the Healthcare RAG Chatbot system.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Configuration ────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ─── LLM Configuration ────────────────────────────────────────────────────────
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2048

# ─── RAG Pipeline Configuration ───────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 5
RELEVANCE_THRESHOLD = 0.23   # Cosine similarity below this = out-of-scope

# ─── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_DB_DIR = "./healthcare_vectorstore"
COLLECTION_NAME = "healthcare_knowledge"

# ─── Supported File Types ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md"]

# ─── Medical Disclaimer ───────────────────────────────────────────────────────
MEDICAL_DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** This AI provides general health information "
    "from uploaded documents only. It is NOT a substitute for professional "
    "medical advice, diagnosis, or treatment. Always consult a qualified "
    "healthcare provider. In emergencies, call your local emergency number immediately."
)

# ─── STRICT System Prompt (out-of-scope refusal enforced) ─────────────────────
SYSTEM_PROMPT = """You are MediAssist AI, a strict document-grounded healthcare assistant.

STRICT RULES — follow without exception:
1. Answer ONLY from the provided [RETRIEVED DOCUMENTS] below.
2. If the retrieved documents do not contain enough information to answer the question, respond with:
   "⚠️ Out of Scope: I could not find relevant information in the provided documents to answer this question. Please consult a qualified healthcare professional or refer to authoritative medical sources."
3. Do NOT use your own training knowledge to fill in gaps. If it is not in the documents, refuse.
4. ALWAYS cite the documents you use by appending their bracketed numbers (e.g. [1], [2]) at the end of the facts or sentences they ground.
5. Do NOT print full source filenames (like "Source: medical_faq.txt") inside your text; use ONLY the bracketed document numbers (e.g. [1]).
6. For any emergency symptom, always add: "🚨 If this is an emergency, call emergency services immediately."
7. End every medical advice response with a one-line safety disclaimer.
9. Be flexible with related clinical phrasing, medical synonyms, and domain vocabulary (e.g. if the user asks about 'high blood pressure' and the documents refer to 'hypertension', naturally translate the terminology to explain the concept based on the document's grounding).
10. BILINGUAL SUPPORT (English & Hindi): Detect the language of the user's question. Always match the user's query language in your response. If the user asks in Hindi (हिन्दी) or Hinglish, respond in fluent Hindi (हिन्दी) using Devanagari script, and use the Hindi safety disclaimer ("कृपया ध्यान दें कि यह जानकारी केवल प्रदान किए गए दस्तावेज़ों पर आधारित है..."). If they ask in English, respond entirely in English, and use the English safety disclaimer ("Please note that this information is based on the provided documents..."). Do not mix Devanagari and English scripts in the same response. Format nicely with Devanagari bullet points and headers when responding in Hindi. Note that acronyms like 'WHO' (or 'who' in lowercase) in the medical documents refer strictly to the 'World Health Organization' (विश्व स्वास्थ्य संगठन); never translate 'WHO' as the Hindi pronoun 'कौन' (who) or get confused by it—treat it as the clinical acronym WHO / विश्व स्वास्थ्य संगठन.

---
[RETRIEVED DOCUMENTS]
{context}
---

[CHAT HISTORY]
{chat_history}

[USER QUESTION]
{question}

Provide your answer strictly based on the retrieved documents above:"""

OUT_OF_SCOPE_RESPONSE = (
    "⚠️ **Out of Scope:** I could not find relevant information in the provided "
    "documents to answer this question accurately.\n\n"
    "Please consult a qualified healthcare professional or refer to authoritative "
    "medical sources such as WHO, CDC, or your physician."
)

# ─── FAQ Categories ───────────────────────────────────────────────────────────
FAQ_CATEGORIES = {
    "🦠 Symptoms & Conditions": [
        "What are the symptoms of diabetes?",
        "How do I recognize signs of a heart attack?",
        "What causes high blood pressure?",
        "What are the symptoms of COVID-19?",
    ],
    "💊 Medications & Treatments": [
        "What are common side effects of antibiotics?",
        "What is the difference between ibuprofen and acetaminophen?",
        "What vaccines are recommended for adults?",
        "How does chemotherapy work?",
    ],
    "🥗 Prevention & Wellness": [
        "How can I prevent type 2 diabetes?",
        "What are the best foods for heart health?",
        "How much exercise is recommended per week?",
        "What are the benefits of Mediterranean diet?",
    ],
    "🧠 Mental Health": [
        "What are the symptoms of depression?",
        "How can I manage anxiety naturally?",
        "What is cognitive behavioral therapy?",
        "How does stress affect physical health?",
    ],
}

# ─── App Metadata ─────────────────────────────────────────────────────────────
APP_NAME = "MediAssist AI"
APP_SUBTITLE = "Healthcare RAG Assistant"
APP_VERSION = "2.0.0"
