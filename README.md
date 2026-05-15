# 📊 AI Report Generator

An AI-powered data insights tool that analyzes any CSV or Excel file and generates a professional business report instantly — with auto-generated charts and 
downloadable in multiple formats.

## 🎯 What it does
- Upload any CSV or Excel file (.xlsx) via a clean web interface
- Select specific sheets to analyze (supports multi-sheet Excel files)
- AI analyzes your data and generates a structured report with:
  - Executive Summary
  - Key Findings
  - Data Quality Notes
  - Actionable Recommendations
- Auto-generates 4 charts from your data:
  - Numeric column distributions
  - Top categorical column bar chart
  - Numeric vs Numeric scatter plot
  - Missing values breakdown
- Download the report as:
  - 📄 Plain text (.txt)
  - 📝 Word document (.docx) with charts embedded

## 🛠️ Tech Stack
- **Python** — core language
- **Streamlit** — web interface
- **LangChain** — LLM orchestration
- **Google Gemini 1.5 Flash** — AI model (free tier)
- **Pandas** — data processing and summarization
- **Matplotlib** — automatic chart generation
- **Python-docx** — Word document generation with embedded charts

## 🚀 How to run locally

### 1. Clone the repo
git clone https://github.com/AyushLadha/ai-report-generator.git
cd ai-report-generator

### 2. Install dependencies
pip install streamlit pandas langchain langchain-google-genai 
            google-generativeai python-dotenv matplotlib python-docx

### 3. Get a free Gemini API key
- Go to https://aistudio.google.com
- Click Get API Key → Create API key

### 4. Add your API key
Create a .env file in the project folder:
GEMINI_API_KEY=your_key_here

### 5. Run the app
streamlit run app.py

Open http://localhost:8501 in your browser

## 📁 Supported File Types
| Format | Single Sheet | Multi Sheet |
|--------|-------------|-------------|
| .csv   | ✅          | ➖ (CSV is always single sheet) |
| .xlsx  | ✅          | ✅ (select one or more sheets)  |

## 💡 Key concepts learned
- Calling LLM APIs with LangChain
- Prompt engineering with SystemMessage and HumanMessage
- Compressing large datasets into token-efficient summaries
- Building interactive web apps with Streamlit
- Managing API keys securely with .env files
- Generating and embedding charts into Word documents
- Handling mixed data types in pandas DataFrames

## 📈 Sample output
Tested on a 56,000 row enterprise sales dataset — Gemini identified:
- $24.91M total revenue with 41.97% profit margin
- Cross-selling opportunity (avg order qty of only 1.5 units)
- North America driving 48% of order volume
- Actionable recommendations for regional expansion

## 🔮 Next improvements
- [ ] Follow-up questions about the report (chat history)
- [ ] Deploy to Streamlit Cloud (free hosting)
- [ ] Chat with your data (AI Agents)

## 📁 Project structure
ai-report-generator/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md           # This file