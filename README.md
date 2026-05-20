# 📊 AI Report Generator

An AI-powered data insights tool that analyzes any CSV or Excel file and generates a professional business report instantly - with auto-generated charts, downloadable
in multiple formats, and an interactive follow-up chat.

## 🌐 Live Demo
🔗 [AI Report Generator](https://ai-report-generator-zh3az7tzybjdchxa6atywt.streamlit.app/)

## 🎯 What it does
- Upload any CSV or Excel file (.xlsx) via a clean web interface
- Select specific sheets to analyze (supports multi-sheet Excel files)
- AI analyzes your data and generates a structured report with:
  - Executive Summary
  - Key Findings
  - Data Quality Notes
  - Actionable Recommendations
- Auto-generates charts from your data:
  - Numeric column distributions
  - Top categorical column bar chart
  - Numeric vs Numeric scatter plot
  - Missing values breakdown
- Download the report as:
  - 📄 Plain text (.txt)
  - 📝 Word document (.docx) with charts embedded
- 💬 Interactive follow-up chat - ask questions about your report
  - Smart context management - remembers recent conversation
  - Answers fresh for new topics, references history for follow-ups
  - Resets automatically when a new file is uploaded

## 🛠️ Tech Stack
- **Python** - core language
- **Streamlit** - web interface
- **LangChain** - LLM orchestration
- **Google Gemini 1.5 Flash** - AI model (free tier)
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

## 🌐 Deploy to Streamlit Cloud (free)
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Add GEMINI_API_KEY in Streamlit Secrets
5. Deploy — your app is live!

## 📁 Supported File Types
| Format | Single Sheet | Multi Sheet |
|--------|-------------|-------------|
| .csv   | ✅          | ➖ (CSV is always single sheet) |
| .xlsx  | ✅          | ✅ (select one or more sheets)  |

## 💡 Key concepts learned
- Calling LLM APIs with LangChain
- Prompt engineering with SystemMessage, HumanMessage and AIMessage
- Smart context management for multi-turn conversations
- Compressing large datasets into token-efficient summaries
- Auto-detecting grouping and aggregation columns for any dataset type
- Building interactive web apps with Streamlit
- Managing API keys securely with .env files
- Generating and embedding charts into Word documents
- Handling mixed data types in pandas DataFrames
- Streamlit session state for persistent data across reruns

## 📈 Sample output
Tested on a 56,000 row enterprise sales dataset - Gemini identified:
- $24.91M total revenue with 41.97% profit margin
- Cross-selling opportunity (avg order qty of only 1.5 units)
- North America driving 48% of order volume
- United States leading with 19,811 transactions
- Actionable recommendations for regional expansion

## 🔮 Next improvements
- [ ] Chat with your data (Phase 2 - AI Agents)