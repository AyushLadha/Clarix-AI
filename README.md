# 📊 AI Report Generator

An AI-powered data insights tool that analyzes any CSV file and generates a professional business report instantly.

## 🎯 What it does
- Upload any CSV file via a clean web interface
- AI analyzes your data and generates a structured report with:
  - Executive Summary
  - Key Findings
  - Data Quality Notes
  - Actionable Recommendations
- Download the report as a text file

## 🛠️ Tech Stack
- **Python** — core language
- **Streamlit** — web interface
- **LangChain** — LLM orchestration
- **Google Gemini** — AI model (free tier)
- **Pandas** — data processing

## 🚀 How to run locally

### 1. Clone the repo
git clone https://github.com/AyushLadha/ai-report-generator.git
cd ai-report-generator

### 2. Install dependencies
pip install streamlit pandas langchain langchain-google-genai google-generativeai python-dotenv

### 3. Get a free Gemini API key
- Go to https://aistudio.google.com
- Click Get API Key → Create API key
- No credit card required

### 4. Add your API key
Create a .env file in the project folder:
GEMINI_API_KEY=your_key_here

### 5. Run the app
streamlit run app.py

Open http://localhost:8501 in your browser

## 💡 Key concepts learned
- Calling LLM APIs with LangChain
- Prompt engineering with SystemMessage and HumanMessage
- Compressing large datasets into token-efficient summaries
- Building interactive web apps with Streamlit
- Managing API keys securely with .env files

## 📈 Sample output
Tested on a 56,000 row enterprise sales dataset — Gemini identified:
- $24.91M total revenue with 41.97% profit margin
- Cross-selling opportunity (avg order qty of only 1.5 units)
- North America driving 48% of order volume
- Actionable recommendations for regional expansion

## 🔮 Next improvements
- [ ] Add Excel file support
- [ ] Auto-generate charts alongside the report
- [ ] Add follow-up questions about the report
- [ ] Deploy to Streamlit Cloud