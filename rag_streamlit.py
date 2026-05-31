import os
import uuid
import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
import tempfile

# ---- Load API -----
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        st.error("GEMINI_API_KEY not found.")
        st.stop()

# ----- Page configuration -----
st.set_page_config(page_title = "Document Q&A", page_icon = "📄", layout = "wide")
st.title("📄 Document Q&A")
st.caption("Upload documents and ask questions across all of them instantly")

# ----- Initialize models -----
@st.cache_resource
def init_models():
    embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", google_api_key = api_key)
    llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50, separators = ["\n\n", "\n", ".", " "])
    return embeddings, llm, splitter 

embeddings, llm, splitter = init_models()

# ----- Session State -----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"session_{st.session_state.session_id[:8]}"

# ----- Initialize ChromaDB -----
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path = "./chroma_db")
    return client

chroma_client  = init_chromadb()

# Get or create this session's collection
collection_name = f"session_{st.session_state.session_id}"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    collection = chroma_client.create_collection(collection_name)

# ----- Helper: index a document -----
def index_document(uploaded_file) -> tuple:
    """Load, chunk, embed and store a document. Returns (success, message)."""

    file_name = uploaded_file.name

    # Skip if already indexed
    if file_name in st.session_state.indexed_files:
        return False, "Already indexed"

    try:
        # Save uploaded file to temp location
        suffix = os.path.splitext(file_name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load document based on file type
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
        else:
            os.unlink(tmp_path)
            return False, f"Unsupported file type: {suffix}"

        documents = loader.load()
        os.unlink(tmp_path)  # clean up temp file

        # Chunk the document
        chunks = splitter.split_documents(documents)

        if not chunks:
            return False, "No text could be extracted from this document"

        # Embed and store each chunk
        for i, chunk in enumerate(chunks):
            embedding = embeddings.embed_query(chunk.page_content)
            collection.add(
                ids=[f"{file_name}_{i}"],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[{
                    "source": file_name,
                    "chunk": str(i),
                    "total_chunks": str(len(chunks)),
                    "page": str(chunk.metadata.get("page", ""))
                }]
            )

        # Mark as indexed
        st.session_state.indexed_files.append(file_name)
        return True, len(chunks)

    except Exception as e:
        return False, str(e)

# ----- Helper: query documents -----
def query_documents(question: str, top_k: int = 4) -> tuple:
    """Search documents and generate an answer with sources."""
    # Embed question
    question_embedding = embeddings.embed_query(question)

    # Search
    results = collection.query(query_embeddings = [question_embedding], n_results = min(top_k, collection.count()))

    if not results['documents'][0]:
        return "No relevant documents found.", []
    
    # extract cunks and sources
    chunks = results['documents'][0]
    sources = list(set(m['source'] for m in results['metadatas'][0]))
    context = "\n\n---\n\n".join(chunks)

    # Build report
    messages = [
        SystemMessage(content="""You are a helpful assistant that answers questions
based on provided document excerpts.
RULES:
- Only answer from the provided context
- If not found say "I could not find this in the uploaded documents"
- Cite which document the information came from
- Be specific with exact figures when available
- Format your answer clearly with bullet points when listing multiple items"""),
        HumanMessage(content=f"""Context from documents:
{context}

Question: {question}

Answer based only on the context above:""")
    ]

    # Extract text from response
    response = llm.invoke(messages)
    if isinstance(response.content, list):
        answer = " ".join(b['text'] for b in response.content
                         if isinstance(b, dict) and b.get('type') == 'text')
    else:
        answer = response.content

    return answer, sources

# ----- Layout: sidebar for uploads -----
with st.sidebar:
    st.header("📁 Upload Documents")
    st.caption("PDF, Word (.docx), and Text files supported")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Process newly uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.indexed_files:
                with st.spinner(f"Indexing {file.name}..."):
                    success, result = index_document(file)
                if success:
                    st.success(f"✅ {file.name} — {result} chunks indexed")
                else:
                    if result == "Already indexed":
                        pass  # silently skip
                    else:
                        st.error(f"❌ {file.name}: {result}")

    st.divider()

    # Show indexed documents
    if st.session_state.indexed_files:
        st.markdown(f"**{len(st.session_state.indexed_files)} document(s) indexed:**")
        for fname in st.session_state.indexed_files:
            st.markdown(f"📄 {fname}")

        st.divider()

        # Settings
        top_k = st.slider(
            "Chunks to retrieve",
            min_value=1, max_value=8, value=4,
            help="More = more context, but slower"
        )

        # Clear button
        if st.button("🗑️ Clear all", width="stretch"):
            # Delete and recreate collection
            chroma_client.delete_collection(collection_name)
            collection = chroma_client.create_collection(collection_name)
            st.session_state.indexed_files = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        top_k = 4

# ----- Main area: chat interface -----
if not st.session_state.indexed_files:
    st.info("👈 Upload documents in the sidebar to get started")
    st.markdown("""
**What you can do:**
- Upload multiple PDFs, Word docs, or text files
- Ask questions across all documents at once
- Get answers with source citations
- Documents are searchable by meaning, not just keywords
""")
else:
    st.markdown(f"**{len(st.session_state.indexed_files)} document(s) ready** - ask anything!")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption(f"Sources: {', '.join(message['sources'])}")

    # Chat input
    question = st.chat_input("Ask a question about your documents...")  
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.status("Searching documents...", expanded = False) as status:
                answer, sources = query_documents(question, top_k)
                status.update(label = "✅ Done!", state = "complete")

            st.markdown(answer)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")

        # Save to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })