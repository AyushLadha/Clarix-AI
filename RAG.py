import os
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ----- Initialize ChromaDB -----
# ChromaDN runs locally 
client = chromadb.PersistentClient(path = "./chroma_db")
collection = client.get_or_create_collection(name = "document")
print("ChromaDB initialized")

# ----- Initalize embedding model -----
# Google's free embedding model - same API
embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", google_api_key = api_key)
print("Embedding model ready")

# ----- Initialize LLM -----
llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)

# ----- Text Splitter -----
# RecursiveCharacterTextSplitter - best all-around chunking strategy
# chunk_size=500 tokens, overlap=50 tokens
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap  = 50, separators = ["\n\n", "\n", ".", " "])
print("Text splitter ready")

# ----- Function to load and index a document -----
def index_document(file_path: str):
    """Load a document, chunk it, embed it, store in ChromaDB."""

    print(f"\n Loading: {file_path}")

    # Load based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return
    
    # load document
    documents = loader.load()
    print(f" Loaded {len(documents)} pages/sections")

    # Chunk the document
    chunks = splitter.split_documents(documents)
    print(f" Split into {len(chunks)} chunks")

    # Embed and store each chunk
    for i, chunk in enumerate(chunks):
        # Generate embedding for this chunk
        embedding = embeddings.embed_query(chunk.page_content)

        # Store in ChromaDB
        collection.add(
            ids = [f"{file_path}_{i}"],
            embeddings = [embedding],
            documents = [chunk.page_content],
            metadatas = [{
                "source": file_path,
                "chunk": i,
                "total_chunks": len(chunks)
            }]
        )

    print(f" Indexed {len(chunks)} chunks into ChromaDB")

# ----- Step 6: Function to query the documents -----
def query_documents(question: str, top_k: int = 3) -> str:
    """Find relevant chunks and generate an answer."""

    print(f"\n Question: {question}")

    # Embed the question
    question_embedding = embeddings.embed_query(question)

    # Search ChromaDB for similar chunks
    results = collection.query(query_embeddings = [question_embedding], n_results = top_k)

    # Extract retrieved chunks
    retrieved_chunks = results['documents'][0]
    sources = [m['source'] for m in results['metadatas'][0]]

    print(f"   Found {len(retrieved_chunks)} relevant chunks")
    for i, (chunk, source) in enumerate(zip(retrieved_chunks, sources)):
        print(f"   Chunk {i+1} from {source}: {chunk[:80]}...")

    # Build prompt with retrieved context
    context = "\n\n---\n\n".join(retrieved_chunks)

    messages = [
        SystemMessage(content="""You are a helpful assistant that answers questions 
based on the provided document excerpts.
RULES:
- Only answer from the provided context
- If the answer is not in the context say "I could not find this in the provided documents"
- Always mention which document the information came from
- Be specific and cite exact figures when available"""),
        HumanMessage(content=f"""Context from documents:
{context}

Question: {question}

Answer based only on the context above:""")
    ]

    response = llm.invoke(messages)

    if isinstance(response.content, list):
        answer = " ".join(b['text'] for b in response.content
                         if isinstance(b, dict) and b.get('type') == 'text')
    else:
        answer = response.content

    return answer

# ----- Step 7: Test it -----
# Create a simple test text file
test_content = """
Q3 2023 Financial Report — Adventure Works Cycles

Executive Summary:
Total revenue for Q3 2023 reached $8.2 million, representing a 15% increase 
from Q3 2022. The Bikes category led performance with $4.1 million in revenue 
and a profit margin of 48%.

Regional Performance:
The United States remained our strongest market with $2.9 million in revenue.
Australia showed significant growth with $2.7 million, up 22% year over year.
European markets combined contributed $2.6 million.

Product Insights:
Mountain bikes outperformed road bikes by 31% in unit sales.
Accessories revenue grew 18% driven by helmet and safety gear demand.
The Mountain-200 Black series was our top selling product line.

Outlook:
Q4 2023 projections indicate continued growth with expected revenue of $9.5 million.
We plan to expand the accessories product line and increase marketing spend in Canada.
"""

# Save test file
with open("test_report.txt", "w") as f:
    f.write(test_content)
print("\n Created test_report.txt")

# Index it
index_document("test_report.txt")

# Ask questions
questions = [
    "What was the total revenue in Q3 2023?",
    "Which product category had the highest profit margin?",
    "What are the Q4 projections?"
]

print("\n" + "="*60)
print("TESTING RAG PIPELINE")
print("="*60)

for question in questions:
    answer = query_documents(question)
    print(f"\n Answer: {answer}")
    print("-"*40)