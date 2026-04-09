# BillingIQ: AI-Powered Healthcare Coverage Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for healthcare billing and coverage inquiries. Built with modern AI/ML stack: Streamlit UI, LangChain orchestration, ChromaDB vector storage, and Groq LLM inference.

**Note:** This is a recreated/sanitized version of a production system for demonstration purposes.

---

## Overview

BillingIQ answers user questions about healthcare billing and coverage by:
1. **Retrieving** relevant information from a knowledge base document
2. **Augmenting** the retrieval with conversational context
3. **Generating** grounded, factual responses using an LLM

Unlike generic chatbots that hallucinate, RAG grounds every answer in actual document content, making it ideal for domain-specific Q&A systems.

---

## Core Components

### 1. **Document Ingestion Module**
- **Loaders:** Support for `.pdf` (PyPDFLoader), `.docx` (Docx2txtLoader), and `.txt` (TextLoader)
- **Purpose:** Extract raw text from domain-specific documents
- **Output:** Unstructured document objects

### 2. **Text Chunking Engine**
- **Strategy:** `RecursiveCharacterTextSplitter` with semantic boundaries
- **Chunk Size:** 1,100 characters
- **Overlap:** 150 characters (prevents context fragmentation)
- **Purpose:** Break long documents into manageable, overlapping chunks
- **Output:** List of text chunks with metadata

### 3. **Embedding Model**
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFaceEmbeddings)
- **Dimensions:** 384-dimensional vectors
- **Purpose:** Convert text chunks into semantic vectors
- **Benefit:** Captures semantic meaning, not just keyword matching
- **Output:** Vector embeddings for all chunks

### 4. **Vector Store (Chroma)**
- **Backend:** SQLite database with vector indices
- **Purpose:** Persistent storage and fast similarity search
- **Search Type:** Maximum Marginal Relevance (MMR)
  - Retrieves diverse chunks (avoids redundant results)
  - k=6 final results, fetch_k=20 pool before diversity filtering
- **Persistence:** Stored in `./chroma_index/`
- **Output:** Top-k most relevant chunks per query

### 5. **LLM Inference Engine**
- **Provider:** Groq API (llama-3.3-70b-versatile model)
- **Temperature:** 0.25 (low randomness, factual responses)
- **Max Tokens:** 500 per response
- **Top-p:** 0.9 (nucleus sampling for diversity control)
- **Multi-key Fallback:** Supports up to 4 API keys for rate limiting
- **Output:** Grounded response text

### 6. **Conversational Memory**
- **Type:** Session-state chat history
- **Storage:** Streamlit `st.session_state`
- **Persistence:** Per-browser-session (lost on refresh)
- **Redaction:** Outgoing user messages redacted before storage (PII safety)

### 7. **Safety & Validation Layer**
- **PHI Detection:** Regex patterns for email, phone, SSN, account IDs
- **Redaction:** Masks sensitive data before storing in conversation history
- **Support Escalation:** Detects support requests and routes to human agents
- **Email Integration:** Optional SMTP for escalation notifications

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BILLINGIQ SYSTEM                           │
└─────────────────────────────────────────────────────────────────┘

                           USER INTERFACE
                              Streamlit
                        (Web UI + Chat History)
                                 │
                                 ▼
                     ┌──────────────────────┐
                     │  Chat Input Handler  │
                     │  + PII Redaction     │
                     └──────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌──────────────────┐    ┌─────────────────────┐
            │ Support Escalate?│    │  Query to RAG       │
            │  (Keyword Match) │    │  Pipeline           │
            └──────────────────┘    └─────────────────────┘
                    │                         │
                    ▼                         ▼
            ┌──────────────────┐    ┌─────────────────────┐
            │ Email Support    │    │ User Query Vector   │
            │ Notification     │    │ Embedding           │
            └──────────────────┘    │ (Sentence-Trans)    │
                                    └─────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────────┐
                                    │ Chroma Vector Store │
                                    │ (SQLite Backend)    │
                                    │ MMR Retrieval       │
                                    │ k=6, fetch_k=20     │
                                    └─────────────────────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────┐
                        │  Retrieved Context Chunks + History │
                        └─────────────────────────────────────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────┐
                        │  LLM Prompt Template (LangChain)    │
                        │  + Grounding Instructions           │
                        │  + Language Detection               │
                        │  + Safety Rules                     │
                        └─────────────────────────────────────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────┐
                        │  Groq LLM Inference                 │
                        │  llama-3.3-70b-versatile            │
                        │  (with Multi-Key Fallback)          │
                        └─────────────────────────────────────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────┐
                        │  Generate Response                  │
                        │  (Grounded in KB + Context)         │
                        └─────────────────────────────────────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────┐
                        │  Display Response + Sources         │
                        │  Update Chat History                │
                        └─────────────────────────────────────┘
```

---

## RAG Mechanism (How It Works)

### The Problem: Generic Chatbots Hallucinate
Most LLMs generate plausible-sounding but wrong answers when asked domain-specific questions. Example:
```
User: "What's the difference between deductible and copay?"
Generic ChatGPT: [Makes up info that sounds right but might be incorrect]
```

### The Solution: Retrieval-Augmented Generation
Instead of relying on LLM training data, RAG:

1. **Retrieve:** Find the most relevant sections from your knowledge base
2. **Augment:** Insert retrieved context into the LLM prompt
3. **Generate:** Have the LLM answer based *only* on the retrieved context

```
User Question: "What's the difference between deductible and copay?"
        ↓
[Search Vector DB for similar chunks]
        ↓
Retrieved Context: "A deductible is the amount you pay before insurance kicks in.
                    A copay is a fixed fee per visit..."
        ↓
[Inject into LLM prompt with instruction to ground answer in this context]
        ↓
LLM Response: "Based on your knowledge base: A deductible is... A copay is..."
        ↓
Factual, grounded response ✓
```

### Why This Matters
- **Accuracy:** Answers come from your documents, not LLM hallucinations
- **Currency:** Easily update by replacing the KB document
- **Transparency:** Users see sources for every answer
- **Control:** No need to fine-tune; configuration-driven

---

## Complete Process Flow

```
START
  │
  ├─ [ONCE] Initialize App
  │   ├─ Load .env configuration
  │   ├─ Read KB_PATH from config
  │   ├─ Check if vector index exists (chroma_index/)
  │   ├─ If stale or missing:
  │   │   ├─ Call load_docs(KB_PATH)
  │   │   ├─ Apply RecursiveCharacterTextSplitter
  │   │   ├─ Generate embeddings via HuggingFaceEmbeddings
  │   │   ├─ Build Chroma index from documents
  │   │   └─ Save document tracker (_indexed_doc.txt)
  │   └─ Build retriever (MMR search)
  │
  ├─ [PER SESSION] Initialize Session State
  │   └─ Create empty messages[] list
  │
  ├── CHAT LOOP (On User Input)
  │   │
  │   ├─ Render chat message history from session_state
  │   │
  │   ├─ Await user_q (st.chat_input)
  │   │
  │   ├─ PII Detection
  │   │   ├─ Scan user_q for emails, phones, SSN, account IDs
  │   │   └─ Store detection flag
  │   │
  │   ├─ Support Request Detection
  │   │   ├─ Check if user_q contains keywords ("support", "agent", etc.)
  │   │   └─ If yes → Trigger escalation flow (get name, email → send notification)
  │   │
  │   ├─ Redact PII from user_q
  │   │   └─ Replace emails with [EMAIL_REDACTED], phones with [PHONE_REDACTED], etc.
  │   │
  │   ├─ Display redacted message in chat UI
  │   │
  │   ├─ Call get_qa_response(user_q)
  │   │   │
  │   │   ├─ Embed user_q: vector = embedding_model(user_q)
  │   │   │
  │   │   ├─ Retrieve: chunks = retriever.get_relevant_documents(user_q)
  │   │   │   └─ Chroma returns top-6 chunks (MMR filtered)
  │   │   │
  │   │   ├─ Build LLM Prompt
  │   │   │   ├─ Inject retrieved chunks as context
  │   │   │   ├─ Include user question
  │   │   │   ├─ Append conversation history
  │   │   │   ├─ Add grounding instruction (no hallucination)
  │   │   │   ├─ Add language detection rule (respond in user's language)
  │   │   │   └─ Add safety rules (no medical/legal advice, escalate if needed)
  │   │   │
  │   │   ├─ Invoke LLM
  │   │   │   ├─ Try GROQ_API_KEY_1 on Groq API
  │   │   │   ├─ If rate-limited: Try GROQ_API_KEY_2, _3, _4
  │   │   │   └─ Return generated response text
  │   │   │
  │   │   └─ Return response + source chunks
  │   │
  │   ├─ Display LLM Response
  │   │   └─ Show in chat UI with green/teal formatting
  │   │
  │   ├─ Display Sources (Expandable)
  │   │   └─ User can click "View Sources" to see retrieved chunks
  │   │
  │   ├─ Append to Chat History
  │   │   └─ Add {role: "assistant", content: response} to session_state
  │   │
  │   └─ Rerun to update UI
  │
  └─ REPEAT chat loop

END (on browser close or app restart)
```

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | 1.51.0 |
| **Orchestration** | LangChain | 0.3.28+ |
| **Vector DB** | ChromaDB | 0.5.5 |
| **Embeddings** | Sentence Transformers | 3.0.1 |
| **LLM API** | Groq (`langchain-groq`) | 0.2.5+ |
| **Document Loaders** | LangChain Community | 0.3.21+ |
| **Config Management** | python-dotenv | 1.0.1 |
| **Language** | Python | 3.10+ |

---

## Project Structure

```
chatbot-prototype/
├── app.py                      # Main Streamlit app + RAG pipeline
├── requirements.txt            # Python dependencies (pinned versions)
├── .env                        # Local secrets (excluded from Git)
├── .env.example               # Template for public sharing
├── .gitignore                 # Git exclusions (venv, chroma_index, .env)
├── README.md                  # This file
│
├── sample_knowledge_base.pdf  # Your domain-specific document
├── chroma_index/              # Vector DB (auto-created, excluded from Git)
│   ├── chroma.sqlite3         # SQLite DB with vectors
│   └── _indexed_doc.txt       # Tracker file (which KB is indexed)
│
└── venv/                      # Python virtual environment (excluded from Git)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Groq API key (free: [console.groq.com](https://console.groq.com))

### 1. Clone & Navigate
```bash
cd "Chatbot prototype for Github/Chatbot prototype"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY_1=gsk_YOUR_KEY_HERE
KB_PATH=sample_knowledge_base.pdf
VECTOR_DB_DIR=./chroma_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_TEMPERATURE=0.25
LLM_MAX_TOKENS=500
LLM_TOP_P=0.9
SUPPORT_EMAIL=support@example.com
```

### 5. Add Your Knowledge Base
Place your `.pdf`, `.docx`, or `.txt` file in the project root and set `KB_PATH` in `.env`.

### 6. Run the App
```bash
streamlit run app.py
```

Browser opens at `http://localhost:8501` (or configured port)

---

## Usage Examples

### Example 1: Basic Q&A
```
User: "What is a deductible?"

BillingIQ: "Based on your knowledge base, a deductible is..."
           [View Sources] ← Click to see retrieved document chunks
```

### Example 2: Multi-turn Conversation
```
User: "What's the difference between HMO and PPO?"

BillingIQ: "[Response grounded in KB]"

User: "Which one is cheaper?"

BillingIQ: "[Response uses both previous question and KB context]"
```

### Example 3: Support Escalation
```
User: "I need to speak to a human agent."

BillingIQ: "I can route this to support. What name should support use?"
           [Collects name, email → Sends notification email]
```

---

## Key Features

✅ **Accurate Answers:** Grounded in your knowledge base (no hallucination)
✅ **Multi-language:** Auto-detects user language, responds in same language
✅ **Transparent:** Shows source chunks for every answer
✅ **Stateful:** Conversation history with context awareness
✅ **Safe:** PII redaction (email, phone, SSN, account IDs)
✅ **Scalable:** Multi-API-key fallback for rate limit handling
✅ **Production-Ready:** Error handling, logging, graceful degradation
✅ **Easy to Customize:** Config-driven (no code changes needed)

---

## Privacy & Safety

### Data Handling
- **Chat History:** Stored only in browser session (lost on refresh)
- **User Input:** Redacted for sensitive patterns before storage
- **API Keys:** Read from `.env` only (never hardcoded)
- **Vectors:** Stored locally in SQLite (not sent to external services)

### Compliance Notes
- No PII is sent to external LLM APIs (redacted before transmission)
- SMTP is optional; app works without email notifications
- All data stays on-device except LLM inference (which doesn't include PII)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Chroma index stale** | Delete `chroma_index/` folder and restart app |
| **API rate limit** | Add more GROQ_API_KEY in `.env` (uses fallback) |
| **PDF not loading** | Ensure `KB_PATH` in `.env` points to correct file |
| **Slow responses** | Check network; Groq API may have latency |
| **Imports not found** | Run `pip install -r requirements.txt` again |

---

