import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ----- Load API
load_dotenv()
api_key = os.getenv("Gemini_API_KEY")
if not api_key:
    raise ValueError("Gemini_API_KEY is not set. Add it to your environment or .env file.")

# ----- Load the data
df = pd.read_csv("fact_sales.csv")
print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

# ----- Summarize the data
def summarize_dataframe(df):
    lines = []

    # Basic shape and column names
    lines.append(f"Dataset: {df.shape[0]:,} rows and {df.shape[1]} columns")
    lines.append(f"\nColumns: {df.columns.tolist()}")

    # Numeric stats
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        lines.append(f"\nNumeric Statistics:\n{numeric_cols.describe().round(2).to_string()}")

    # Categorical breakdown - auto detect text columns
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_columns[:5]:   # limit to first 5 categorical columns for brevity
        top = df[col].value_counts().head(5).to_dict()
        lines.append(f"\nTop values in '{col}': {top}")

    # Missing values
    nulls = df.isnull().sum()
    if nulls.any():
        lines.append(f"\nMissing values: {nulls[nulls > 0].to_dict()}")

    # Sample rows
    lines.append(f"\nSample rows: {df.head(5).to_string()}")
    
    return "\n".join(lines)

# ----- Send to gemini and Initializing the llm
summary = summarize_dataframe(df)
print("\n--- Data summary set to Gemini ---")
print(summary[:500], "...\n")    # for preview of first 500 characters

llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)

# ----- Create the prompt
messages = [
    SystemMessage(content="""You are a senior data analyst. You are given a summary of a dataset. 
                  Your task is to analyze the summary and provide insights, trends, and potential business recommendations based on the data.
                  Return the report with these sections:
                  ## Executive Summary
                  ## Key Findings
                  ## Recommendations"""),
    
    HumanMessage(content=f"Analyze this sales data:\n\n{summary}")
]

print("\n--- Prompt sent to Gemini ---")
response = llm.invoke(messages)
print("\n--- AI Analysis Report ---")
print(response.content[0]['text'])