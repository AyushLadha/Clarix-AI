import os
import io
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import uuid

# ---- API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ---- Load data
df = pd.read_csv("fact_sales.csv")
print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns\n")

# ---- Shared memory - one MemorySaver for all agents
shared_memory = MemorySaver()
thread_id = str(uuid.uuid4())
llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)

# ------------------ DATA Agent ------------------
def create_data_agent_standalone(df, llm, memory, thread_id):

    @tool
    def get_dataframe_info() -> str:
        """Use this first to understand the dataset structure.
        Returns column names, data types, shape and sample rows."""
        print("📋 [DATA AGENT] Reading dataset structure")
        info = []
        info.append(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        info.append(f"Columns: {df.columns.tolist()}")
        info.append(f"Data types:\n{df.dtypes.to_string()}")
        info.append(f"Sample rows:\n{df.head(3).to_string()}")
        return "\n".join(info)

    @tool
    def query_dataframe(pandas_code: str) -> str:
        """Query the dataframe using pandas code.
        The dataframe is available as 'df'.
        IMPORTANT:
        - Never use print() — just write the expression directly
        - Never assign new columns — read only
        - Never use = to assign
        Examples:
        - df['revenue'].sum()
        - df.groupby('country')['revenue'].sum().nlargest(5)
        - df.groupby('category_name').apply(lambda x: x['profit'].sum()/x['revenue'].sum())
        Always return the expression result directly."""
        print(f"⚡ [DATA AGENT] Running: {pandas_code[:60]}")

        # Guard against assignments
        if '=' in pandas_code and not '==' in pandas_code:
            return "Error: assignments not allowed. Use read-only expressions like df['col'].sum()"

        # Guard against print statements
        if 'print(' in pandas_code:
            # Strip print() wrapper automatically
            pandas_code = pandas_code.replace('print(', '').rstrip(')')

        try:
            result = eval(pandas_code, {"df": df, "pd": pd})
            return str(result)
        except Exception as e:
            return f"Error: {e}. Try a simpler expression."

    @tool
    def get_column_values(column_name: str) -> str:
        """See unique values in a specific column.
        Example: get_column_values('country')"""
        print(f"🔎 [DATA AGENT] Checking column: {column_name}")
        try:
            return f"Top values in '{column_name}':\n{df[column_name].value_counts().head(10).to_string()}"
        except Exception as e:
            return f"Error: {e}"

    tools = [get_dataframe_info, query_dataframe, get_column_values]
    agent = create_agent(model = llm, tools = tools, checkpointer = memory)
    return agent

# ------------------ CHART AGENT - generates visualizations ------------------
def create_chart_agent_standalone(df, llm, memory, thread_id):

    @tool
    def generate_chart(chart_request: str) -> str:
        """Generate a matplotlib chart based on the request.
        Examples:
        - 'bar chart of revenue by country'
        - 'histogram of annual income'
        - 'scatter plot of revenue vs profit'
        Returns CHART_GENERATED|<base64> when successful."""
        import base64
        print(f"📈 [CHART AGENT] Generating: {chart_request}")
        try:
            matplotlib.use('Agg')
            chart_llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)
            col_info = f"""
            Columns: {df.columns.tolist()}
            Data types: {df.dtypes.to_string()}
            Sample: {df.head(3).to_string()}
            """
            code_messages = [
                SystemMessage(content="""You are a matplotlib expert.
Write Python code to generate the requested chart.
The dataframe is available as 'df'.
RULES:
- matplotlib only
- Always set title, xlabel, ylabel
- plt.figure(figsize=(8,5))
- plt.tight_layout()
- No plt.show(), no file save
- ALWAYS format Y axis — no scientific notation
- For money columns: ax = plt.gca(); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'${x:,.0f}'))
- For counts: ax = plt.gca(); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'{x:,.0f}'))
- Rotate x labels: plt.xticks(rotation=45, ha='right')
- End with:
  buf = io.BytesIO()
  plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
  buf.seek(0)
  chart_bytes = buf.getvalue()
  plt.close()
Respond with ONLY Python code, no markdown."""),
                HumanMessage(content=f"Dataset:\n{col_info}\n\nGenerate: {chart_request}")
            ]
            code_response = chart_llm.invoke(code_messages)
            if isinstance(code_response.content, list):
                chart_code = " ".join(
                    b['text'] for b in code_response.content
                    if isinstance(b, dict) and b.get('type') == 'text'
                )
            else:
                chart_code = code_response.content
            chart_code = chart_code.replace('```python', '').replace('```', '').strip()
            local_vars = {"df": df, "pd": pd, "plt": plt, "io": io, "matplotlib": matplotlib}
            exec(chart_code, local_vars)
            chart_bytes = local_vars.get('chart_bytes')
            if chart_bytes:
                encoded = base64.b64encode(chart_bytes).decode('utf-8')
                return f"CHART_GENERATED|{encoded}"
            return "Chart generation failed - chart_bytes not found."
        except Exception as e:
            return f"Error generating chart: {str(e)}"

    tools = [generate_chart]
    agent = create_agent(model = llm, tools = tools, checkpointer = memory)
    return agent

# ------------------ REPORT AGENT - generates insights from data ------------------
def create_report_agent_standalone(df, llm, memory, thread_id):

    @tool
    def analyze_data(analysis_request: str) -> str:
        """Analyze the dataset and generate business insights.
        Examples:
        - 'analyze revenue trends by country'
        - 'identify top performing products'
        - 'find anomalies in profit margins'"""
        print(f"📊 [REPORT AGENT] Analyzing: {analysis_request}")
        try:
            # Build a quick summary for analysis
            summary = []
            summary.append(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
            summary.append(f"Columns: {df.columns.tolist()}")
            numeric_cols = df.select_dtypes(include='number')
            if not numeric_cols.empty:
                summary.append(f"Stats:\n{numeric_cols.describe().round(2).to_string()}")
            analysis_llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)
            messages = [
                SystemMessage(content="""You are a senior data analyst.
Analyze the dataset summary and provide:
1. Key insight (1-2 sentences with specific numbers)
2. Trend identified
3. One actionable recommendation
Be specific and concise."""),
                HumanMessage(content=f"Dataset summary:\n{chr(10).join(summary)}\n\nAnalyze: {analysis_request}")
            ]
            response = analysis_llm.invoke(messages)
            if isinstance(response.content, list):
                return " ".join(b['text'] for b in response.content if isinstance(b, dict) and b.get('type') == 'text')
            return response.content
        except Exception as e:
            return f"Error analyzing: {str(e)}"

    @tool
    def generate_recommendation(topic: str) -> str:
        """Generate a specific business recommendation based on the data.
        Examples:
        - 'recommend marketing strategy based on country revenue'
        - 'suggest pricing strategy based on profit margins'"""
        print(f"💡 [REPORT AGENT] Generating recommendation for: {topic}")
        try:
            rec_llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = api_key)
            messages = [
                SystemMessage(content="""You are a business strategy consultant.
Based on the data topic provided, give ONE specific, actionable recommendation.
Format: "Recommendation: [action] because [data-backed reason]"
Keep it to 2-3 sentences maximum."""),
                HumanMessage(content=f"Dataset columns: {df.columns.tolist()}\nTopic: {topic}")
            ]
            response = rec_llm.invoke(messages)
            if isinstance(response.content, list):
                return " ".join(b['text'] for b in response.content if isinstance(b, dict) and b.get('type') == 'text')
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    tools = [analyze_data, generate_recommendation]
    agent = create_agent(model = llm, tools = tools, checkpointer = memory)
    return agent


# ------------------ ORCHESTRATOR - routes and combines ------------------
def create_orchestrator(df, llm, memory, thread_id):

    # Give orchestrator access to all sub-agents as tools
    data_agent = create_data_agent_standalone(df, llm, memory, thread_id)
    chart_agent = create_chart_agent_standalone(df, llm, memory, thread_id)
    report_agent = create_report_agent_standalone(df, llm, memory, thread_id)

    @tool
    def call_data_agent(question: str) -> str:
        """Call the data agent to query the raw dataframe.
        Use for specific numbers, counts, or aggregations.
        The agent will return exact numbers from the raw data."""
        print(f"\n🔀 [ORCHESTRATOR] Calling Data Agent: {question}")
        result = data_agent.invoke(
            {"messages": [HumanMessage(content=f"""Answer this data question using the query_dataframe tool.
            Write pandas expressions directly — no print(), no assignments.
            Question: {question}""")]},
            config={"configurable": {"thread_id": f"{thread_id}_data"}}
        )
        last = result["messages"][-1].content
        if isinstance(last, list):
            return " ".join(b['text'] for b in last if isinstance(b, dict) and b.get('type') == 'text')
        return str(last)

    @tool
    def call_chart_agent(chart_request: str) -> str:
        """Call the chart agent to generate a visualization.
        Use this ONLY when the user explicitly asks for a chart, graph, or plot.
        Examples:
        - 'bar chart of revenue by country'
        - 'histogram of annual income'"""

        # Small delay to avoid rate limits
        import time
        time.sleep(2)

        print(f"\n🔀 [ORCHESTRATOR] Calling Chart Agent: {chart_request}")

        result = chart_agent.invoke(
            {"messages": [HumanMessage(content=f"Generate this chart: {chart_request}")]},
            config={"configurable": {"thread_id": f"{thread_id}_chart"}}
        )
        last = result["messages"][-1].content
        if isinstance(last, list):
            return " ".join(b['text'] for b in last if isinstance(b, dict) and b.get('type') == 'text')
        return str(last)

    @tool
    def call_report_agent(analysis_request: str) -> str:
        """Call the report agent for insights and recommendations.
        Use this when the user wants analysis, trends, or business recommendations.
        Examples:
        - 'explain what revenue trends mean'
        - 'give me business recommendations'
        - 'what should we focus on?'"""

        # Small delay to avoid rate limits
        import time
        time.sleep(1)

        print(f"\n🔀 [ORCHESTRATOR] Calling Report Agent: {analysis_request}")

        result = report_agent.invoke(
            {"messages": [HumanMessage(content = analysis_request)]},
            config={"configurable": {"thread_id": f"{thread_id}_report"}}
        )
        last = result["messages"][-1].content
        if isinstance(last, list):
            return " ".join(b['text'] for b in last if isinstance(b, dict) and b.get('type') == 'text')
        return str(last)

    tools = [call_data_agent, call_chart_agent, call_report_agent]
    orchestrator = create_agent(model = llm, tools = tools, checkpointer = memory)
    return orchestrator


# -------- TEST IT
print("Building multi-agent system...")
orchestrator = create_orchestrator(df, llm, shared_memory, thread_id)
print("✅ All agents ready!\n")

# Test questions
# Test all 3 agents together
questions_full = [
    "What is total revenue by country and give me business recommendations?",
    "Show me a chart of profit by category and explain what it means",
]

print("\n\n" + "="*60)
print("TESTING FULL MULTI-AGENT SYSTEM")
print("="*60)

for question in questions_full:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    result = orchestrator.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}}
    )

    last = result["messages"][-1].content
    if isinstance(last, list):
        answer = " ".join(b['text'] for b in last if isinstance(b, dict) and b.get('type') == 'text')
    else:
        answer = str(last)

    print(f"\n✅ Answer: {answer}")