# ✦ Clarix - AI-Powered Data & Document Intelligence

An end-to-end AI platform that transforms raw data and documents into 
actionable insights - built with Python, LangChain, LangGraph, and Google Gemini.

## 🌐 Live Demo
🔗 [Clarix](https://ai-report-generator-zh3az7tzybjdchxa6atywt.streamlit.app/)

---

## ⚡ Data Analysis Mode

Upload any CSV or Excel file and get a structured business report instantly.

**What it does:**
- Analyzes any dataset - sales, HR, finance, operations
- Generates a structured report with Executive Summary, Key Findings, 
  Data Quality Notes, and Recommendations
- AI selects the most meaningful columns and auto-generates charts
- Download the full report as Word (.docx) with embedded charts or plain text
- Ask follow-up questions - a LangGraph AI agent queries your raw data directly

**AI Agent tools:**
- `get_dataframe_info` - understands dataset structure
- `query_dataframe` - runs live pandas queries for exact numbers
- `get_column_values` - explores categorical columns
- `generate_chart` - creates any visualization on demand
- `generate_insights` - produces business recommendations

---

## 🔍 Doc Explore Mode

Upload PDFs, Word docs, or text files and ask questions across all of them.

**What it does:**
- Indexes uploaded documents into a ChromaDB vector database
- Searches by meaning - not just keywords
- Answers questions with exact citations from source documents
- Supports multiple documents simultaneously
- Each session gets its own isolated collection

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini (gemini-3.1-flash-lite) |
| Embeddings | Google gemini-embedding-001 |
| Agent framework | LangChain + LangGraph |
| Agent memory | LangGraph MemorySaver |
| Vector database | ChromaDB (in-memory) |
| Data processing | Pandas, Matplotlib |
| Document parsing | LangChain community loaders |
| Export | Python-docx |
| Deployment | Streamlit Cloud |

---

## 🚀 How to run locally

### 1. Clone the repo
git clone https://github.com/AyushLadha/ai-report-generator.git
cd ai-report-generator

### 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Get a free Gemini API key
- Go to https://aistudio.google.com
- Click Get API Key → Create API key
- Free tier: 1,500 requests/day, no credit card required

### 5. Add your API key
Create a .env file:
GEMINI_API_KEY=your_key_here

### 6. Run the app
streamlit run app.py

---

## 🔒 Edge Cases Handled
- Mixed data type columns - auto-converted for compatibility
- No column headers - first row used as headers with warning
- Large files (100K+ rows) - performance warning shown
- Unnamed/numeric column names - charts skipped with explanation
- New file upload - full session reset automatically
- PDF rate limits - batch delay between embeddings
- Complex PDF metadata - simplified before ChromaDB storage

---

## 📈 Sample outputs

**Data Analysis on 56,000 row enterprise sales dataset:**
- $24.91M total revenue with 41.97% profit margin
- United States leading with $7.9M revenue
- Bikes category: 48% profit margin - highest performer
- Agent answers "top 5 products by profit" with exact figures instantly

**Doc Explore on a 15-page research PDF:**
- Indexes in under 60 seconds
- Answers "what models were used?" with exact methodology citations
- Searches across multiple documents simultaneously

---

## 💡 Key concepts learned
- LLM API calls and prompt engineering with LangChain
- ReAct agent pattern - Reason → Act → Observe → Repeat
- Building custom agent tools with @tool decorator
- Session memory with LangGraph MemorySaver
- RAG pipeline - chunking, embedding, vector search, retrieval
- ChromaDB vector database for semantic document search
- Streamlit session state for persistent data across reruns
- Saving matplotlib charts as bytes for session persistence
- Secure API key management with .env and Streamlit secrets
- Multi-mode Streamlit app architecture

---

## 🔮 What's next
- [ ] Phase 4: Combined mode - query data + search documents simultaneously
- [ ] Persistent vector storage across sessions
- [ ] Support for more file types (CSV to Doc Explore, images)
