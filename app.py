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
uploaded_file = st.file_uploader("Upload your  CSV file or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Load the file
    file_name = uploaded_file.name
    if file_name.endswith(".csv"): 
        # CSV has only 1 sheet. Load it directly.
        sheets_data = {"Sheet 1": pd.read_csv(uploaded_file)}   
    else:
        x1 = pd.ExcelFile(uploaded_file)
        sheets_names = x1.sheet_names

        if len(sheets_names) == 1:
            # Only 1 sheet, load it directly
            sheets_data = {sheets_names[0]: x1.parse(sheets_names[0])} 
        else:  
            # for multiple sheets, let user select which one to analyze
            st.info(f"This file contains {len(sheets_names)} sheets.")
            selected_sheet = st.multiselect("Select the sheets to analyze:", sheets_names, default = sheets_names[0])

            if not selected_sheet:
                st.warning("Please select at least one sheet to proceed.")
                st.stop()
            
            # Load the selected sheets into a dictionary
            sheets_data = {sheet: x1.parse(sheet) for sheet in selected_sheet}

    # Show quick stats and # Preview the data for each sheet
    for sheet_name, df in sheets_data.items():
        if len(sheets_data) > 1:
            st.subheader(f"Sheet: {sheet_name}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        col3.metric("Numeric Columns", f"{df.select_dtypes(include='number').shape[1]}")
        col4.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        with st.expander(f"Preview - {sheet_name}", expanded = True):
            st.dataframe(df.head(10), width = "stretch")

    st.divider()

    # ----- Generate Button
    if st.button("Generate Report", type = "primary", width = "stretch"):
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
                
                categorical_columns = df.select_dtypes(include = ['object', 'string']).columns.tolist()
                for col in categorical_columns[:5]:
                    top = df[col].value_counts().head(5).to_dict()
                    lines.append(f"\nTop values in '{col}': {top}")
                
                nulls = df.isnull().sum()
                if nulls.any():
                    lines.append(f"\nMissing values: {nulls[nulls > 0].to_dict()}")
                
                lines.append(f"\nSample rows: {df.head(5).to_string()}")
                
                return "\n".join(lines)

            # Initialize the llm
            llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)

            # Generate a report for each sheet and combine them into one final report

            sheet_reports = {}

            # Loop through each sheet and generate a report for it
            for sheet_name, df in sheets_data.items(): 
                # Summarize the data for the current sheet
                summary = summarize_dataframe(df) 

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
                
                    HumanMessage(content = f"Analyze this data from '{sheet_name}':\n\n{summary}")
                ]

                response = llm.invoke(messages) 
                sheet_reports[sheet_name] = response.content[0]['text']

            # Combine all sheet reports into one final report
            if len(sheet_reports) == 1:
                # If only one sheet, no need to add sheet names in the report
                full_report = list(sheet_reports.values())[0]
            else:
                # If multiple sheets, add sheet names as headers in the report
                full_report = ""
                for sheet_name, report in sheet_reports.items():
                    full_report += f"\n\n{'='*50}\n"
                    full_report += f"\nReport for Sheet: {sheet_name}\n"
                    full_report += f"\n{'='*50}\n\n"
                    full_report += report

            # Display report
            st.success("Report generated successfully!")
            st.markdown("---")
            st.markdown(full_report)

            # Download button
            base_name = os.path.splitext(uploaded_file.name)[0]
            st.download_button(
                label = "Download Report as Text File",
                data = full_report,
                file_name = f"{base_name}_Report.txt",
                mime = "text/plain",
                width = "stretch"
            )
