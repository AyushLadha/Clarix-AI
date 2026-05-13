import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


#----- Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
    st.stop()

# ----- Page Configuration
st.set_page_config(
    page_title = "AI Report Generator",
    page_icon = "📊",
    layout = "wide"
)

# ----- Title
st.title("📊 AI Report Generator")
st.caption("Generate insights and business recommendations from your dataset using AI")

# ----- Sidebar
with st.sidebar:
    st.header("Settings")
    focus = st.text_area(
        "What should the report focus on?",
        value = "Provide a summary of the key findings, insights and recommendations from the dataset.",
        height = 175
    )

# ----- File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load the file
    df = pd.read_csv(uploaded_file)

    # Show quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Numeric Columns", f"{df.select_dtypes(include='number').shape[1]}")
    col4.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    # Preview the data
    with st.expander("Preview Data", expanded = True):
        st.dataframe(df.head(10), use_container_width = True)

    st.divider()

    # ----- Generate Button
    if st.button("Generate Report", type = "primary", use_container_width = True):
        with st.spinner("Analyzing your data with Gemini... (~15-30 seconds)"):

            # Summarize the data
            def summarize_dataframe(df):
                lines = []
                # Basic shape, column names and numeric stats, categorical columns and missing values
                lines.append(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
                
                lines.append(f"\nColumns: {df.columns.tolist()}")
                
                numeric_columns = df.select_dtypes(include = 'number')
                if not numeric_columns.empty:
                    lines.append(f"\nNumeric Statistics: {numeric_columns.describe().round(2).to_string() }")
                
                categorical_columns = df.select_dtypes(include = 'object').columns.tolist()
                for col in categorical_columns[:5]:
                    top = df[col].value_counts().head(5).to_dict()
                    lines.append(f"\nTop values in '{col}': {top}")
                
                nulls = df.isnull().sum()
                if nulls.any():
                    lines.append(f"\nMissing values: {nulls[nulls > 0].to_dict()}")
                
                lines.append(f"\nSample rows: {df.head(5).to_string()}")
                
                return "\n".join(lines)

            summary = summarize_dataframe(df)

            # Initialize the llm
            llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)

            # Create the prompt
            messages = [
                SystemMessage(content = f"""You are a senior data analyst. You are given a summary of a dataset. 
                              Your task is to analyze the summary and provide insights, trends, and potential business recommendations based on the data.
                              Return the report with these sections:
                              ## Executive Summary
                              ## Key Findings
                              ## Data Quality Notes
                              ## Recommendations
                              Focus on: {focus}"""),
                
                HumanMessage(content = f"Analyze this data:\n\n{summary}")
            ]

            response = llm.invoke(messages)
            report = response.content[0]['text']

            # Display report
            st.success("Report generated successfully!")
            st.markdown("---")
            st.markdown("report")

            # Download button
            st.download_button(
                label = "Download Report as Text File",
                data = report,
                file_name = "AI_Data_Report.txt",
                mime = "text/plain",
                use_container_width = True
            )
