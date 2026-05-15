import os
import pandas as pd
from dotenv import load_dotenv
from pyparsing import col
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
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
        for col in sheets_data.columns:
            if sheets_data[col].dtype == 'object':
                sheets_data[col] = sheets_data[col].astype(str)
            
    else:
        x1 = pd.ExcelFile(uploaded_file)
        sheets_names = x1.sheet_names

        if len(sheets_names) == 1:
            # Only 1 sheet, load it directly
            sheets_data = {sheets_names[0]: x1.parse(sheets_names[0])}
            for col in sheets_data.columns:
                if sheets_data[col].dtype == 'object':
                    sheets_data[col] = sheets_data[col].astype(str)
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

            def charts_generator(df, sheet_name):
                matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
                
                charts_created = 0

                st.subheader(f"Auto Generated Charts for {sheet_name}" if len(sheets_data) > 1 else "Auto Generated Charts")

                # Chart 1:  Distribution of numeric columns (histograms)
                numeric_cols = df.select_dtypes(include = 'number').columns.tolist()
                if numeric_cols:
                    st.markdown("**Distribution of Numeric Columns**")
                    # Show up to 4 columns
                    cols_to_plot = numeric_cols[:4]
                    fig, axes = plt.subplots(1, len(cols_to_plot), figsize = (5 * len(cols_to_plot), 4))

                    # If only one column, axes is not a list, so we make it a list for consistency
                    if len(cols_to_plot) == 1:
                        axes = [axes]
                    
                    for ax, col in zip(axes, cols_to_plot):
                        df[col].dropna().hist(ax = ax, bins = 20, color = 'skyblue', edgecolor = 'black')
                        ax.set_title(col, fontsize = 10)
                        ax.set_xlabel("")
                        ax.tick_params(labelsize = 8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    charts_created += 1

                # Chart 2: Bar chart of top categorical columns
                categorical_cols = df.select_dtypes(include = ['object', 'string']).columns.tolist()

                if categorical_cols:
                    # pick the categorical column with the most unique values but less than 20 (to avoid overcrowded charts)
                    best_cat_col = None
                    for col in categorical_cols:
                        unique_vals = df[col].nunique()
                        if 1 < unique_vals <= 20:  # We want some variability but not too many unique values
                            best_cat_col = col
                            break
                    
                    if best_cat_col:
                        st.markdown(f"**Distribution of Categorical Column: {best_cat_col}**")
                        fig, ax = plt.subplots(figsize = (6, 4))
                        top_values = df[best_cat_col].value_counts().head(10)
                        top_values.plot(kind = 'bar', ax = ax, color = 'salmon', edgecolor = 'black')
                        ax.set_title(f"Top Values in '{best_cat_col}'", fontsize = 10)
                        ax.set_xlabel("")
                        ax.tick_params(axis = 'x', rotation = 45, labelsize = 8)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        charts_created += 1
                
                # Chart  3: Numeric vs Numeric (scatter plot)
                if len(numeric_cols) >= 2:
                    # Drop rows where either of the two columns is null to avoid issues in plotting
                    scatter_df = df[[numeric_cols[0], numeric_cols[1]]].dropna()

                    if len(scatter_df) > 0:  # Only plot if there are valid rows to plot
                        st.markdown(f"**{numeric_cols[0]} vs {numeric_cols[1]}**")
                        fig, ax = plt.subplots(figsize = (6, 4))
                        ax.scatter(scatter_df[numeric_cols[0]], scatter_df[numeric_cols[1]], alpha = 0.5, color = 'lightgreen', s = 20, edgecolor = 'black')
                        ax.set_xlabel(numeric_cols[0], fontsize = 8)
                        ax.set_ylabel(numeric_cols[1], fontsize = 8)
                        ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}", fontsize = 10)
                        ax.tick_params(labelsize = 8)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        charts_created += 1
                
                # Chart 4: Missing values bar chart (if there are missing values)
                nulls = df.isnull().sum()
                nulls = nulls[nulls > 0]

                if not nulls.empty:
                    st.markdown("**Missing Values Bar Chart**")
                    fig, ax = plt.subplots(figsize = (6, 4))
                    nulls.plot(kind = 'bar', ax = ax, color = 'lightcoral', edgecolor = 'black')
                    ax.set_title("Missing Values by Column", fontsize = 10)
                    ax.set_xlabel("Columns", fontsize = 8)
                    ax.set_ylabel("Number of Missing Values", fontsize = 8)
                    ax.tick_params(axis = 'x', rotation = 45, labelsize = 8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    charts_created += 1

                # If no charts were created, show a message
                if charts_created == 0:
                    st.info("No suitable charts could be generated for this dataset. Try uploading a different file or adjusting the data.")

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

            # Auto-generated charts for each sheet
            st.markdown("---")
            for sheet_name, df in sheets_data.items():
                st.markdown(f"### Charts for Sheet: {sheet_name}")
                charts_generator(df, sheet_name)

            # Download button
            base_name = os.path.splitext(uploaded_file.name)[0]
            st.download_button(
                label = "Download Report as Text File",
                data = full_report,
                file_name = f"{base_name}_Report.txt",
                mime = "text/plain",
                width = "stretch"
            )
