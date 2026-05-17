import os
import pandas as pd
from dotenv import load_dotenv
from pyparsing import col
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --------------- Helper function to clean dataframe columns
def clean_dataframe(df):
    """Fix mixed type columns that cause Arrow/PyArrow errors in Streamlit"""
    for col in df.columns:
        # If column is object type but has mixed int/str values — convert all to str
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    return df

# --------------- Function to summarize the data
def summarize_dataframe(df):
    lines = []
    # Basic shape, column names and numeric stats, categorical columns and missing values
    lines.append(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
                
    lines.append(f"\nColumns: {df.columns.tolist()}")
                
    numeric_columns = df.select_dtypes(include = 'number')
    if not numeric_columns.empty:
        lines.append(f"\nNumeric Statistics: {numeric_columns.describe().round(2).to_string() }")
                
    categorical_columns = df.select_dtypes(include = ['object', 'string']).columns.tolist()

     # For categorical columns, we want to show the distribution of values for the top categorical columns, 
     # so that we can get a sense of what the data looks like and what kind of insights we might be able to generate from it.
    for col in categorical_columns:
        top = df[col].value_counts().head(10).to_dict()
        lines.append(f"\nTransaction/record count by '{col}':\n{top}")

    # For grouping and aggregation, we want to pick categorical columns that have a reasonable number of unique values (not too many, not too few) 
    # to avoid overwhelming the model with too much information or having meaningless groupings.
    good_group_cols = [
        col for col in categorical_columns
        if 2 <= df[col].nunique() <= 30  # sweet spot for grouping
    ]
    # For aggregation columns, we want to pick numeric columns that have a good amount of variability and are not just identifiers or constant values, 
    # as those would not provide meaningful insights when aggregated.
    good_agg_cols = numeric_columns.columns.tolist()

    # We will show some sample aggregations by key groups to give the model a sense of what kind of insights can be generated from the data, 
    # and to provide it with some specific examples of trends and patterns in the data that it can reference when generating the report.
    if good_group_cols and good_agg_cols:
        lines.append(f"\n--- Aggregations by Key Groups ---")
        for group_col in good_group_cols[:5]:   # top 5 grouping cols
            for agg_col in good_agg_cols[:5]:   # top 5 numeric cols
                try:
                    agg = df.groupby(group_col)[agg_col].agg(['sum', 'mean', 'count'])
                    agg = agg.sort_values('sum', ascending=False).head(10).round(2)
                    lines.append(f"\n{agg_col} by {group_col}:\n{agg.to_string()}")
                except:
                    pass
                
    nulls = df.isnull().sum()
    if nulls.any():
        lines.append(f"\nMissing values: {nulls[nulls > 0].to_dict()}")
                
    lines.append(f"\nSample rows: {df.head(5).to_string()}")
                
    return "\n".join(lines)

# --------------- Function to generate charts for a given dataframe and display in Streamlit
def charts_generator(df, sheet_name, sheets_data):
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

# --------------- Function to generate Word report with charts
def generate_word_report(full_report, sheets_data):
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import io
    matplotlib.use('Agg')

    doc = Document()

    # Title and styling 
    title = doc.add_heading('AI Analysis Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Report content
    doc.add_heading('Analysis', level = 1)
    for line in full_report.split('\n'):
        line = line.strip()
        if not line:
            pass # Add a blank line for spacing
        elif line.startswith('## '):
            doc.add_heading(line.replace('## ', ''), level = 2)
        elif line.startswith('# '):
            doc.add_heading(line.replace('# ', ''), level = 1)
        elif line.startswith('='*10):
            doc.add_paragraph('─' * 40)
        elif line.startswith('* ') or line.startswith('- '):
            doc.add_paragraph(line[2:], style = 'List Bullet')
        else:
            doc.add_paragraph(line)

    # Charts
    doc.add_page_break()
    doc.add_heading('Auto-Generated Charts', level=1)

    for sheet_name, df in sheets_data.items():
        if len(sheets_data) > 1:
            doc.add_heading(f'Charts - {sheet_name}', level=2)

        numeric_cols = df.select_dtypes(include = 'number').columns.tolist()
        cat_cols = df.select_dtypes(include = ['object', 'string']).columns.tolist()

        # Chart 1: Numeric distributions
        if numeric_cols:
            cols_to_plot = numeric_cols[:4]
            fig, axes = plt.subplots(1, len(cols_to_plot), figsize = (5 * len(cols_to_plot), 4))
            if len(cols_to_plot) == 1:
                axes = [axes]
            for ax, col in zip(axes, cols_to_plot):
                df[col].dropna().hist(ax = ax, bins = 20, color = '#4f46e5', edgecolor = 'white')
                ax.set_title(col, fontsize = 11)
                ax.tick_params(labelsize = 9)
            plt.tight_layout()

            # Save chart to memory buffer and insert into Word
            buf = io.BytesIO()
            fig.savefig(buf, format = 'png', dpi = 150, bbox_inches = 'tight')
            buf.seek(0)
            doc.add_paragraph('Numeric Column Distributions').runs[0].bold = True
            doc.add_picture(buf, width=Inches(6))
            plt.close()

        # Chart 2: Top categorical bar chart
        best_cat = None
        for col in cat_cols:
            if df[col].nunique() != 'nan' and 2 <= df[col].nunique() <= 15:
                best_cat = col
                break

        if best_cat:
            fig, ax = plt.subplots(figsize=(7, 4))
            df[best_cat].value_counts().head(10).plot(
                kind = 'bar', ax = ax, color = '#4f46e5', edgecolor = 'white'
            )
            ax.set_title(f"Top values in '{best_cat}'", fontsize=11)
            ax.tick_params(axis = 'x', rotation = 45, labelsize = 9)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format = 'png', dpi = 150, bbox_inches = 'tight')
            buf.seek(0)
            doc.add_paragraph(f'Top Values — {best_cat}').runs[0].bold = True
            doc.add_picture(buf, width = Inches(6))
            plt.close()

        # Chart 3: Scatter plot
        if len(numeric_cols) >= 2:
            scatter_df = df[[numeric_cols[0], numeric_cols[1]]].dropna()
            if len(scatter_df) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(
                    scatter_df[numeric_cols[0]],
                    scatter_df[numeric_cols[1]],
                    alpha = 0.4, color = '#4f46e5', s = 15
                )
                ax.set_xlabel(numeric_cols[0], fontsize = 10)
                ax.set_ylabel(numeric_cols[1], fontsize = 10)
                ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}", fontsize = 11)
                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format = 'png', dpi = 150, bbox_inches = 'tight')
                buf.seek(0)
                doc.add_paragraph(f'{numeric_cols[0]} vs {numeric_cols[1]}').runs[0].bold = True
                doc.add_picture(buf, width = Inches(6))
                plt.close()

        # Chart 4: Missing values
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            fig, ax = plt.subplots(figsize=(7, 3))
            nulls.plot(kind = 'bar', ax = ax, color = '#e11d48', edgecolor = 'white')
            ax.set_title("Missing values per column", fontsize = 11)
            ax.tick_params(axis = 'x', rotation = 45, labelsize = 9)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format = 'png', dpi = 150, bbox_inches = 'tight')
            buf.seek(0)
            doc.add_paragraph('Missing Values per Column').runs[0].bold = True
            doc.add_picture(buf, width = Inches(6))
            plt.close()

    # Save the Word document to a BytesIO buffer and return it for download
    doc_buf = io.BytesIO()
    doc.save(doc_buf)
    doc_buf.seek(0)
    return doc_buf

# --------------- Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
    st.stop()

# --------------- Page Configuration
st.set_page_config(page_title = "AI Report Generator", page_icon = "📊", layout = "wide")
st.title("📊 AI Report Generator")
st.caption("Generate insights and business recommendations from your dataset using AI")

# --------------- Sidebar
with st.sidebar:
    st.header("Settings")
    focus = st.text_area(
        "What should the report focus on?",
        value = "Provide a summary of the key findings, insights and recommendations from the dataset.",
        height = 175
    )

#  --------------- Session State Initialization 

# Must be initialized before any widget that depends on them
if "report_generated" not in st.session_state: # We use this to keep track of whether a report has been generated yet, so that we can conditionally show the follow-up question section and charts only after a report is generated.
    st.session_state.report_generated = False
if "full_report" not in st.session_state:
    st.session_state.full_report = "" # We store the full report in session state so that we can refer back to it when answering follow-up questions, and also so that we can include it in the Word document download.
if "report_context" not in st.session_state:
    st.session_state.report_context = "" # We use report_context to store the context that we will provide to the model when answering follow-up questions. This can include the original report, and we could also choose to include additional context such as the original data summary or even the raw data if needed. For now, we will just include the full report as the context for follow-up questions, but this can be adjusted based on your needs and experimentation with what provides the best results for follow-up questions.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # We use chat_history to keep track of the conversation history for follow-up questions, so that we can provide that history as context to the model when answering follow-up questions. This allows the model to have the full context of the conversation and provide more coherent and relevant answers to follow-up questions.
if "sheets_data_cache" not in st.session_state:
    st.session_state.sheets_data_cache = {} # We use sheets_data_cache to store the original data from the uploaded sheets in session state, so that we can refer back to the original data when answering follow-up questions and generating charts for follow-up questions. This is important because the user might ask follow-up questions that require referencing the original data, and we want to have that data readily available in session state without needing to re-upload or re-process the file.
if "current_file" not in st.session_state:
    st.session_state.current_file = None # We keep track of the name of the currently uploaded file in session state, so that we can detect when the user uploads a new file and reset the session state accordingly. This is important because if the user uploads a new file, we want to make sure that we clear out any previous report, data, and chat history that was related to the old file, to avoid confusion and ensure that the app is always showing information relevant to the currently uploaded file.

# --------------- File upload
uploaded_file = st.file_uploader("Upload your  CSV file or Excel file", type=["csv", "xlsx", "xls"])

# Reset session state if no file is uploaded, to clear out any previous report and data when the user removes the uploaded file or uploads a new file. 
# This ensures that the app is always in a clean state when starting to analyze a new dataset, and prevents any confusion from leftover data or reports from previous uploads. 
# It also helps to manage memory usage by clearing out old data that is no longer needed when a new file is uploaded.
if uploaded_file is None:
    # No file uploaded — reset everything
    st.session_state.report_generated = False
    st.session_state.full_report = ""
    st.session_state.report_context = ""
    st.session_state.chat_history = []
    st.session_state.sheets_data_cache = {} 
    st.session_state.current_file = None

# We also keep track of the name of the currently uploaded file in session state, so that we can detect when the user uploads a new file and reset the session state accordingly. 
# This is important because if the user uploads a new file, we want to make sure that we clear out any previous report, data, and chat history that was related to the old file, 
# to avoid confusion and ensure that the app is always showing information relevant to the currently uploaded file.

if uploaded_file is not None:
    # If a different file is uploaded, reset everything
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.report_generated = False
        st.session_state.full_report = ""
        st.session_state.report_context = ""
        st.session_state.chat_history = []
        st.session_state.sheets_data_cache = {} 

if uploaded_file:
    # ----- Load the file
    file_name = uploaded_file.name
    if file_name.endswith(".csv"): 
        # CSV has only 1 sheet. Load it directly.
        sheets_data = {"Sheet 1": clean_dataframe(pd.read_csv(uploaded_file))}
    else:
        x1 = pd.ExcelFile(uploaded_file)
        sheets_names = x1.sheet_names

        if len(sheets_names) == 1:
            # Only 1 sheet, load it directly
            sheets_data = {sheets_names[0]: clean_dataframe(x1.parse(sheets_names[0]))}
        else:  
            # for multiple sheets, let user select which one to analyze
            st.info(f"This file contains {len(sheets_names)} sheets.")
            selected_sheet = st.multiselect("Select the sheets to analyze:", sheets_names, default = sheets_names[0])

            if not selected_sheet:
                st.warning("Please select at least one sheet to proceed.")
                st.stop()
            
            # Load the selected sheets into a dictionary
            sheets_data = {sheet: clean_dataframe(x1.parse(sheet)) for sheet in selected_sheet}

    # ----- Show quick stats and Preview the data for each sheet
    for sheet_name, df in sheets_data.items():
        if len(sheets_data) > 1:
            st.subheader(f"Sheet: {sheet_name}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        col3.metric("Numeric Columns", f"{df.select_dtypes(include='number').shape[1]}")
        col4.metric("Categorical Columns", f"{df.select_dtypes(include=['object', 'string']).shape[1]}")
        col5.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        with st.expander(f"Preview - {sheet_name}", expanded = True):
            st.dataframe(df.head(10), width = "stretch")

    st.divider()

    # ----- Generate Button 

    # When the user clicks the button, we will generate the report using Gemini and display it on the page. 
    # We will also generate some auto-charts based on the data and include them in the report.
    # We will use session state to keep track of the generated report and the conversation history for follow-up questions.
    if st.button("Generate Report", type = "primary", width = "stretch"):
        # Reset session state for new report generation
       # Reset for new report
        st.session_state.report_generated = False 
        st.session_state.chat_history = []
        st.session_state.full_report = ""
        st.session_state.report_context = ""
        st.session_state.sheets_data_cache = sheets_data

        # --- Generate the report using Gemini
        with st.spinner("Analyzing your data with Gemini... (~15-30 seconds)"):
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

            # Store the full report and sheets data in session state so that we can refer back to it when answering follow-up questions and generating charts for follow-up questions
            st.session_state.full_report = full_report
            st.session_state.report_context = full_report  # We use the full report as the context for follow-up questions, but this can be adjusted based on your needs and experimentation with what provides the best results for follow-up questions. For example, you could choose to include the original data summary or even the raw data as part of the context for follow-up questions if you find that leads to better answers from the model.
            st.session_state.report_generated = True  # We set this to True so that we can conditionally show the follow-up question section and charts only after a report has been generated. This helps to keep the UI clean and focused, and only show the relevant sections when they are applicable.

    # ----- Display report
    if st.session_state.report_generated: # We only show the report and charts if a report has been generated, to keep the UI clean and focused.
            
        st.success("Report generated successfully!")
        st.markdown("---")
        st.markdown(st.session_state.full_report)

        # --- Auto-generated charts for each sheet
        st.markdown("---")
        for sheet_name, df in sheets_data.items():
            charts_generator(df, sheet_name, st.session_state.sheets_data_cache) # We pass the original sheets data from session state to the charts generator, so that we can refer back to the original data when generating charts for follow-up questions.

        # --- Download button
        st.markdown("---")
        st.markdown("### Download Report")
        base_name = os.path.splitext(uploaded_file.name)[0]

        # Provide both Word and Text download options side by side
        col1, col2 = st.columns([1, 1])
        # Text file download
        with col1:            
            st.download_button(
            label = "Download Report as Text File",
            data = st.session_state.full_report,
            file_name = f"{base_name}_Report.txt",
            mime = "text/plain",
            width = "stretch"
            )
        # Word document download
        with col2:
            doc_buf = generate_word_report(st.session_state.full_report, st.session_state.sheets_data_cache)
            st.download_button(
            label = "Download Report as Word Document",
            data = doc_buf,
            file_name = f"{base_name}_Report.docx",
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width = "stretch"
            )

        # --- Follow-up Questions
        st.markdown("---")
        st.markdown("### Any questions about the report or the data? Ask below!")
        st.caption("You can ask for clarifications, deeper insights, or even request new charts based on the data.")
            
        # Display existing chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f"**You:** {message['content']}")
            else:
                with st.chat_message("Assistant"):
                    st.markdown(f"**Assistant:** {message['content']}")
        
        # Input chat message
        user_input = st.chat_input("Type your question here... ")
        if user_input:
            # Show user's message in the chat
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build the message history for the LLM, including the system prompt, the original report context, and the user's follow-up question
            messages = [
                SystemMessage(content  = """You are a senior data analyst who generated a report.

                            IMPORTANT RULES:
                            1. Answer each question directly and independently
                            2. Only reference previous questions if the current question is clearly related to them
                            3. If the question is about a new topic, answer it fresh without bringing up previous topics
                            4. Always reference specific numbers from the report when relevant
                            5. Be concise and clear
                            6. If you are unsure about something, say so clearly rather than guessing
                            7. If the user asks for simpler explanation, avoid technical jargon and use real-world analogies
                            8. If the user asks a question that cannot be answered from the report, say so and explain what data would be needed"""),

                HumanMessage(content = f"""Here is the original report you generated for reference:\n\n{st.session_state.report_context}"""),

                AIMessage(content = """Understood. I have the original report for reference. I have read and understood the report. 
                          I am ready to answer follow-up questions based on that report. Please provide the follow-up question.""")
            ]
            
            # We then add the recent chat history to the messages that we will pass to the model, so that it has the context of the conversation when answering the follow-up question. 
            # However, we only pass a limited number of recent messages to avoid overwhelming the model with too much context and to keep the conversation focused on the most relevant information. 
            # In this case, we will pass the last 8 messages (4 exchanges) as context for the model when answering the follow-up question.
            recent_history = st.session_state.chat_history[-8:]

            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content = msg["content"]))
                else:
                    messages.append(AIMessage(content = msg["content"]))

            # New question always last
            messages.append(HumanMessage(content = user_input))

            # Get the model's response to the follow-up question
            with st.chat_message("Assistant"):
                with st.spinner("Thinking..."):
                    llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)
                    response = llm.invoke(messages)
                    ai_response = response.content[0]['text']
                    st.markdown(ai_response)

            # Save both the user's question and the AI's response to the chat history in session state
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "Assistant", "content": ai_response})
