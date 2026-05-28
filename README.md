# 📊 AI Report Generator

An AI-powered data insights tool that analyzes any CSV or Excel file and generates
a professional business report instantly - with AI-selected charts, downloadable
in multiple formats, and an interactive follow-up chat.

## 🌐 Live Demo
🔗 [AI Report Generator](https://ai-report-generator-zh3az7tzybjdchxa6atywt.streamlit.app/)

## 🎯 What it does
- Upload any CSV or Excel file (.csv, .xlsx) via a clean web interface
- Select specific sheets to analyze (supports multi-sheet Excel files)
- Choose a report focus area (Sales, HR, Financial, Operations, etc.)
- AI analyzes your data and generates a structured report with:
  - Executive Summary
  - Key Findings
  - Data Quality Notes
  - Actionable Recommendations
- AI selects the most meaningful columns and auto-generates charts:
  - Key metric distributions (histograms)
  - Top categorical column bar chart
  - Metric vs Metric scatter plot
  - Missing values breakdown
- Charts persist across interactions - no regeneration on sidebar changes
- Download the report as:
  - 📄 Plain text (.txt)
  - 📝 Word document (.docx) with charts embedded
- 💬 Interactive follow-up chat with smart context management:
  - Answers fresh for new topics
  - References history for related follow-ups
  - Resets automatically when a new file is uploaded

## 🤖 Phase 2 — AI Data Agent
The follow-up chat is powered by a LangGraph ReAct agent that queries your raw data directly:

- **Live data queries** - exact numbers from raw dataframe, not report summaries
- **On-demand charts** - ask for any chart in plain English, agent generates it
- **Session memory** - agent remembers context across questions in the same session
- **4 tools available:**
  - `get_dataframe_info` - understands dataset structure
  - `query_dataframe` - runs pandas queries on raw data
  - `get_column_values` - explores categorical columns
  - `generate_chart` - creates any visualization on demand

## 🛠️ Tech Stack
- **Python** - core language
- **Streamlit** - web interface and session state management
- **LangChain** - LLM orchestration (prompt chains, message history)
- **Google Gemini** - AI model (free tier, 1,500 requests/day)
- **Pandas** - data processing and summarization
- **Matplotlib** - automatic chart generation
- **Python-docx** - Word document generation with embedded charts

## 🚀 How to run locally

### 1. Clone the repo
git clone https://github.com/AyushLadha/ai-report-generator.git
cd ai-report-generator

### 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Get a free Gemini API key
- Go to https://aistudio.google.com
- Click Get API Key → Create API key
- No credit card required - 1,500 free requests/day

### 5. Add your API key
Create a .env file in the project folder:
GEMINI_API_KEY=your_key_here

### 6. Run the app
streamlit run app.py

Open http://localhost:8501 in your browser

## 🌐 Deploy your own (free)
1. Fork this repo on GitHub
2. Go to https://share.streamlit.io
3. Connect your forked repo
4. Add GEMINI_API_KEY in Streamlit Secrets
5. Click Deploy — live in 2 minutes!

## 📁 Supported File Types
| Format | Single Sheet | Multi Sheet |
|--------|-------------|-------------|
| .csv   | ✅          | ➖ (CSV is always single sheet) |
| .xlsx  | ✅          | ✅ (select one or more sheets)  |

## 🔒 Edge Cases Handled
- Mixed data type columns - auto-converted for compatibility
- No column headers - first row used as headers with warning
- Large files (100K+ rows) - performance warning shown
- Unnamed/numeric column names - charts skipped with explanation
- New file upload - full session reset automatically
- API errors - friendly error messages with retry option

## 💡 Key concepts learned
- Calling LLM APIs with LangChain
- Prompt engineering with SystemMessage, HumanMessage and AIMessage
- Smart context management for multi-turn conversations
- Compressing large datasets into token-efficient summaries
- AI-driven column selection for meaningful chart generation
- Building interactive web apps with Streamlit
- Session state management for persistent data across reruns
- Saving matplotlib charts as bytes for session state storage
- Managing API keys securely with .env and Streamlit secrets
- Generating and embedding charts into Word documents
- Handling mixed data types in pandas DataFrames

## 📈 Sample output
Tested on a 56,000 row enterprise sales dataset — Gemini identified:
- $24.91M total revenue with 41.97% profit margin
- Cross-selling opportunity (avg order qty of only 1.5 units)
- North America driving 48% of order volume
- United States leading with 19,811 transactions
- Actionable recommendations for regional expansion

## 🔮 Next improvements
Phase 3 (RAG + Vector databases)! 🚀