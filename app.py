import os
import re
import shutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

APP_TITLE = "BillingIQ"
APP_SUBTITLE = "AI-Powered Healthcare Coverage Assistant"
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_index")
DEFAULT_KB_CANDIDATES = [
    "sample_knowledge_base.pdf",
    "sample_knowledge_base.docx",
    "sample_knowledge_base.txt",
]


st.set_page_config(
    page_title=f"{APP_TITLE} - RAG Demo",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --bg: #f6f7fb;
        --panel: #ffffff;
        --text: #1f2937;
        --muted: #6b7280;
        --primary: #0f766e;
        --primary-2: #115e59;
        --border: #d1d5db;
        --shadow: 0 6px 16px rgba(17, 24, 39, 0.08);
      }

      .stApp,
      body,
      html,
      #root {
        background: radial-gradient(circle at top left, #eefaf8 0%, #f6f7fb 55%, #f9fafb 100%) !important;
        color: var(--text) !important;
      }

      .main .block-container {
        max-width: 980px;
        padding-top: 1rem;
        padding-bottom: 1rem;
      }

      .hero-card,
      .info-card,
      .sidebar-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: var(--shadow);
      }

      .hero-card {
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
      }

      .hero-card h1 {
        margin: 0;
        color: var(--primary-2);
        font-size: 1.9rem;
      }

      .hero-card p {
        margin: 0.2rem 0 0;
        color: var(--muted);
        font-size: 1rem;
      }

      .info-card {
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
      }

      .info-card h3 {
        margin: 0 0 0.4rem;
        color: var(--primary-2);
      }

      .info-card li {
        margin-bottom: 0.25rem;
      }

      .sidebar-card {
        padding: 0.8rem;
        margin-bottom: 0.7rem;
      }

      .stButton button {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        background: #ffffff !important;
        color: var(--primary-2) !important;
        font-weight: 600 !important;
      }

      .stButton button:hover {
        border-color: var(--primary) !important;
        background: #ecfdf5 !important;
      }

      /* Chat Message Styling */
      .stChatMessage {
        padding: 1rem !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
      }

      /* Assistant message styling - light teal background */
      .stChatMessage-assistant {
        background-color: #f0fdf4 !important;
        border: 1px solid #86efac !important;
        border-left: 4px solid var(--primary) !important;
      }

      /* User message styling - light gray */
      .stChatMessage-user {
        background-color: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        border-right: 4px solid var(--primary-2) !important;
      }

      /* Ensure text is always dark and readable */
      .stChatMessage p,
      .stChatMessage div {
        color: #1f2937 !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
      }

      .stChatMessage strong {
        color: #0f3b38 !important;
        font-weight: 700 !important;
      }

      /* Links styling */
      .stChatMessage a {
        color: var(--primary) !important;
        text-decoration: underline !important;
      }

      .stChatMessage a:hover {
        color: var(--primary-2) !important;
      }

      /* Code blocks in chat */
      .stChatMessage code {
        background-color: rgba(15, 118, 110, 0.1) !important;
        color: #0f3b38 !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
      }

      /* Improve sidebar text readability */
      [data-testid="stSidebar"] {
        background: #ffffff !important;
      }

      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] div {
        color: #1f2937 !important;
      }

      [data-testid="stSidebar"] h3 {
        color: var(--primary-2) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-card">
      <h1>{APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def resolve_kb_path() -> str:
    env_path = os.getenv("KB_PATH")
    if env_path:
        return env_path

    for candidate in DEFAULT_KB_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    # Fallback to first supported local file in project root
    for ext in ("*.pdf", "*.docx", "*.txt"):
        files = list(Path(".").glob(ext))
        if files:
            return str(files[0])

    return "sample_knowledge_base.pdf"


DOC_PATH = resolve_kb_path()


GROUNDING_PROMPT = PromptTemplate.from_template(
    "You are a warm, empathetic assistant for end users asking domain-specific questions.\n\n"
    "LANGUAGE RULES:\n"
    "- Detect the language of the current user question.\n"
    "- Respond entirely in that same language.\n"
    "- Do not mix languages in one response.\n\n"
    "RAG RULES:\n"
    "- Use only the provided Context for factual answers.\n"
    "- If the answer is missing from Context, clearly say so and suggest contacting support.\n"
    "- Do not invent policies or account details.\n\n"
    "SAFETY RULES:\n"
    "- Do not provide legal, medical, or emergency advice.\n"
    "- If the user asks for urgent health or legal actions, suggest contacting qualified professionals.\n"
    "- If the user shares sensitive data, remind them to redact it.\n\n"
    "STYLE RULES:\n"
    "- Start with one short empathetic line.\n"
    "- Then provide 2-5 clear points in plain language.\n"
    "- Keep the response concise unless asked for detail.\n"
    "- Do not mention retrieval, context, or internal systems.\n\n"
    "Conversation Context:\n{conversation_context}\n\n"
    "Context:\n{context}\n\n"
    "User question: {question}\n\n"
    "Answer in the same language as the user question:"
)


# ingestion: load source documents

def load_docs(path: str):
    path_lower = path.lower()
    if path_lower.endswith(".docx"):
        loader = Docx2txtLoader(path)
    elif path_lower.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path_lower.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Use .docx, .pdf, or .txt")

    docs = loader.load()

    cleaned = []
    for doc in docs:
        text = re.sub(r"[^\S\r\n]+", " ", (doc.page_content or "").strip())
        if text and len(text) > 120:
            cleaned.append(doc.__class__(page_content=text, metadata=doc.metadata))
    return cleaned


# chunking + embedding + retrieval
@st.cache_resource(show_spinner=True)
def build_or_load_retriever(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Knowledge base file not found: {path}. "
            "Set KB_PATH in .env or place a .pdf/.docx/.txt file in this folder."
        )

    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    doc_tracker_file = os.path.join(CHROMA_DIR, "_indexed_doc.txt")
    rebuild_needed = False

    if os.path.isdir(CHROMA_DIR):
        if os.path.exists(doc_tracker_file):
            with open(doc_tracker_file, "r", encoding="utf-8") as f:
                indexed_doc = f.read().strip()
            if indexed_doc != path:
                rebuild_needed = True
                shutil.rmtree(CHROMA_DIR)
        else:
            rebuild_needed = True
            shutil.rmtree(CHROMA_DIR)
    else:
        rebuild_needed = True

    if os.path.isdir(CHROMA_DIR) and not rebuild_needed:
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
        return vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

    docs = load_docs(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(chunks, emb, persist_directory=CHROMA_DIR)

    with open(doc_tracker_file, "w", encoding="utf-8") as f:
        f.write(path)

    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})


def get_all_api_keys():
    keys = [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
        os.environ.get("GROQ_API_KEY_4"),
    ]
    return [key for key in keys if key]


def create_llm_client(api_key: str):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.25")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
        top_p=float(os.getenv("LLM_TOP_P", "0.9")),
    )


def invoke_llm_with_fallback(prompt: str):
    api_keys = get_all_api_keys()
    if not api_keys:
        raise RuntimeError("No Groq API keys found in environment variables GROQ_API_KEY_1..4")

    last_error = None
    for i, api_key in enumerate(api_keys, 1):
        try:
            client = create_llm_client(api_key)
            return client.invoke(prompt)
        except Exception as exc:
            error_msg = str(exc).lower()
            if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                print(f"API key #{i} hit rate limits. Trying next key.")
                last_error = exc
                continue
            raise

    raise RuntimeError(
        f"All {len(api_keys)} configured API keys are rate-limited. Last error: {last_error}"
    )


@st.cache_resource(show_spinner=True)
def get_llm_with_fallback():
    api_keys = get_all_api_keys()
    if not api_keys:
        raise RuntimeError("No Groq API keys found in environment variables")

    llm = create_llm_client(api_keys[0])
    return llm


try:
    retriever = build_or_load_retriever(DOC_PATH)
    llm = get_llm_with_fallback()
    RAG_SYSTEM_AVAILABLE = True
except Exception as exc:
    st.error(f"System initialization issue: {exc}")
    st.info("You can still use support escalation. Type: contact support")
    retriever = None
    llm = None
    RAG_SYSTEM_AVAILABLE = False


def redact_sensitive(text: str) -> str:
    if not text:
        return text

    out = text
    out = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]", out)
    out = re.sub(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b", "[PHONE_REDACTED]", out)
    out = re.sub(r"\b\d{3}-?\d{2}-?\d{4}\b", "[SSN_REDACTED]", out)
    out = re.sub(r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b", "[DATE_REDACTED]", out)
    out = re.sub(r"\b(?:account|member|policy|patient|user)\s*(?:number|id|#)?\s*:?\s*[A-Z0-9-]{6,}\b", "[ACCOUNT_REDACTED]", out, flags=re.IGNORECASE)
    return out


def detect_sensitive_in_message(text: str):
    if not text:
        return False, None

    detections = []
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
        detections.append("email")
    if re.search(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b", text):
        detections.append("phone")
    if re.search(r"\b\d{3}-?\d{2}-?\d{4}\b", text):
        detections.append("ssn")
    if re.search(r"\b(?:account|member|policy|patient|user)\s*(?:number|id|#)?\s*:?\s*[A-Z0-9-]{6,}\b", text, re.IGNORECASE):
        detections.append("account id")

    if detections:
        warning = (
            "Privacy notice: I detected sensitive information "
            f"({', '.join(detections)}). It was redacted from chat history. "
            "For account-specific help, use Contact Support."
        )
        return True, warning

    return False, None


def get_conversation_context():
    if "messages" not in st.session_state or len(st.session_state.messages) < 2:
        return "No previous conversation."

    recent_messages = st.session_state.messages[-4:]
    lines = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@example.com")
SUPPORT_TRIGGERS = [
    "contact support",
    "human support",
    "speak to someone",
    "talk to a person",
    "i need a representative",
    "this did not help",
    "this didn't help",
    "quiero hablar con alguien",
    "necesito un representante",
    "contactar soporte",
    "soporte humano",
]


def detect_support_request(message: str) -> bool:
    lower = message.lower()
    return any(trigger in lower for trigger in SUPPORT_TRIGGERS)


def send_support_email(user_name: str, user_email: str, issue_description: str):
    try:
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_email = os.getenv("SMTP_EMAIL")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if not smtp_email or not smtp_password:
            print("=== SUPPORT REQUEST (SMTP NOT CONFIGURED) ===")
            print(f"To: {SUPPORT_EMAIL}")
            print(f"From User: {user_name} ({user_email})")
            print(f"Issue: {issue_description}")
            print("===========================================")
            return True, "Support request logged locally (SMTP not configured)."

        msg = MIMEMultipart()
        msg["From"] = smtp_email
        msg["To"] = SUPPORT_EMAIL
        msg["Subject"] = "New Support Request from RAG Chatbot"
        msg["Reply-To"] = user_email

        body_html = f"""
<html>
<body>
<p>A new support request was submitted through the chatbot.</p>
<p><strong>Name:</strong> {user_name}</p>
<p><strong>Email:</strong> {user_email}</p>
<p><strong>Issue:</strong><br>{issue_description}</p>
</body>
</html>
"""
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.send_message(msg)

        return True, "Email sent successfully"
    except Exception as exc:
        return False, str(exc)


def initialize_escalation_state():
    if "escalation_flow" not in st.session_state:
        st.session_state.escalation_flow = {
            "active": False,
            "step": 0,
            "user_name": "",
            "user_email": "",
            "issue_description": "",
        }


def detect_question_type(question: str):
    q = question.lower()
    if any(k in q for k in ["my bill", "my account", "my payment", "my claim"]):
        return "personal"
    if any(k in q for k in ["what is", "define", "difference between", "explain"]):
        return "educational"
    if any(k in q for k in ["how to", "how do i", "steps", "process", "procedure"]):
        return "procedural"
    return "general"


def format_response(response_text: str):
    key_terms = [
        "deductible",
        "copay",
        "coinsurance",
        "eob",
        "premium",
        "claim",
        "in-network",
        "out-of-network",
        "provider",
    ]
    out = response_text
    for term in key_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        out = pattern.sub(lambda m: f"**{m.group(0)}**", out)
    return out


def add_contextual_guidance(response_text: str, _question_type: str):
    return response_text


def get_qa_response(question: str):
    if not RAG_SYSTEM_AVAILABLE or retriever is None or llm is None:
        raise ConnectionError("The AI system is currently unavailable. Please contact support.")

    question_type = detect_question_type(question)
    conversation_context = get_conversation_context()
    relevant_docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = GROUNDING_PROMPT.format(
        context=context_text,
        question=question,
        conversation_context=conversation_context,
    )

    response = invoke_llm_with_fallback(prompt)
    enhanced = format_response(response.content)
    enhanced = add_contextual_guidance(enhanced, question_type)

    return {"result": enhanced, "source_documents": relevant_docs}


with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### Settings")
    show_sources = st.checkbox("Show sources by default", value=False)
    st.caption("Toggle to automatically view retrieved chunks.")
    st.caption(f"Knowledge base file: {DOC_PATH}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    if st.button("Clear Conversation"):
        st.session_state.pop("messages", None)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-card">
      <h3>How This Demo Helps</h3>
      <ul>
        <li>Answers document-grounded domain questions with retrieval-augmented generation (RAG)</li>
        <li>Provides transparent source snippets for each response</li>
        <li>Redacts sensitive user input from stored chat history</li>
      </ul>
      <p style="margin:0.4rem 0 0; color:#0f766e; font-weight:600;">This is a sanitized demonstration project.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

initialize_escalation_state()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_q = st.chat_input("Ask a question about the loaded document...")

if "pending_question" in st.session_state and st.session_state["pending_question"]:
    user_q = st.session_state["pending_question"]
    st.session_state["pending_question"] = None

if user_q:
    phi_detected = False
    phi_warning = None

    if not st.session_state.escalation_flow["active"]:
        phi_detected, phi_warning = detect_sensitive_in_message(user_q)

    redacted_user_q = redact_sensitive(user_q) if not st.session_state.escalation_flow["active"] else user_q

    with st.chat_message("user"):
        st.write(redacted_user_q)
    st.session_state.messages.append({"role": "user", "content": redacted_user_q})

    if detect_support_request(user_q) and not st.session_state.escalation_flow["active"]:
        st.session_state.escalation_flow["active"] = True
        st.session_state.escalation_flow["step"] = 1
        answer = (
            "I can route this to support. I'll ask two quick questions.\n\n"
            "What name should support use when contacting you?"
        )

        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    elif st.session_state.escalation_flow["active"]:
        flow = st.session_state.escalation_flow

        if flow["step"] == 1:
            flow["user_name"] = user_q
            flow["step"] = 2
            answer = "What is the best email address for support to reach you?"
        elif flow["step"] == 2:
            flow["user_email"] = user_q
            flow["step"] = 3
            answer = "Please briefly describe your issue so support can follow up."
        else:
            flow["issue_description"] = user_q
            success, message = send_support_email(
                flow["user_name"],
                flow["user_email"],
                flow["issue_description"],
            )

            if success:
                answer = (
                    f"Thanks. Your request has been sent to support at {SUPPORT_EMAIL}. "
                    "You can continue using the chatbot while waiting for a response."
                )
            else:
                answer = (
                    "I could not submit the support request automatically. "
                    f"Please contact support directly at {SUPPORT_EMAIL}. Error: {message}"
                )

            if len(st.session_state.messages) >= 6:
                st.session_state.messages = st.session_state.messages[:-6]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "[Support request submitted - personal details are not kept in chat history]",
                }
            )

            flow["active"] = False
            flow["step"] = 0
            flow["user_name"] = ""
            flow["user_email"] = ""
            flow["issue_description"] = ""

        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    else:
        if phi_detected and phi_warning:
            with st.chat_message("assistant"):
                st.write(phi_warning)
            st.session_state.messages.append({"role": "assistant", "content": phi_warning})
            st.stop()

        with st.spinner("Thinking..."):
            try:
                result = get_qa_response(user_q)
                answer = result["result"]
            except ConnectionError as exc:
                st.error(f"Connection Error: {exc}")
                answer = (
                    "I am having trouble connecting to the AI service right now. "
                    "Try again shortly or type 'contact support'."
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                answer = (
                    "I could not access the knowledge base for that request. "
                    "Try rephrasing your question or type 'contact support'."
                )

        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("View Sources", expanded=show_sources):
            if "result" in locals() and result.get("source_documents"):
                for i, doc in enumerate(result.get("source_documents", []), 1):
                    content = doc.page_content
                    question_keywords = set(user_q.lower().split())
                    for drop in ["what", "how", "why", "when", "where", "is", "the", "a", "an"]:
                        question_keywords.discard(drop)

                    best_start = 0
                    for keyword in question_keywords:
                        if keyword in content.lower():
                            best_start = max(0, content.lower().find(keyword) - 100)
                            break

                    excerpt = content[best_start:best_start + 250].strip()
                    if best_start > 0:
                        excerpt = "..." + excerpt
                    if best_start + 250 < len(content):
                        excerpt = excerpt + "..."

                    st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Knowledge Base')}")
                    st.markdown(f"*{excerpt}*")
                    st.markdown("---")
            else:
                st.write("No sources available for this response.")


st.markdown("### Quick Questions")
q1, q2, q3, q4 = st.columns(4)
quick_questions = [
    "Deductible vs copay",
    "Why multiple bills",
    "How to read EOB",
    "Support options",
]
full_questions = [
    "What is the difference between deductible and copay?",
    "Why might someone receive multiple bills for one visit?",
    "How should I read an explanation of benefits?",
    "What support options are available if I still need help?",
]

with q1:
    if st.button(quick_questions[0], key="q1", use_container_width=True, type="secondary"):
        st.session_state["pending_question"] = full_questions[0]
        st.rerun()
with q2:
    if st.button(quick_questions[1], key="q2", use_container_width=True, type="secondary"):
        st.session_state["pending_question"] = full_questions[1]
        st.rerun()
with q3:
    if st.button(quick_questions[2], key="q3", use_container_width=True, type="secondary"):
        st.session_state["pending_question"] = full_questions[2]
        st.rerun()
with q4:
    if st.button(quick_questions[3], key="q4", use_container_width=True, type="secondary"):
        st.session_state["pending_question"] = full_questions[3]
        st.rerun()
