import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ── Load data 
df = pd.read_csv("fact_sales.csv")
print(f" Loaded {df.shape[0]:,} rows and {df.shape[1]} columns\n")

# ── Define tools 
@tool
def get_dataframe_info() -> str:
    """Use this tool first to understand the dataset.
    Returns column names, data types, shape and sample rows."""
    info = []
    info.append(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    info.append(f"Columns: {df.columns.tolist()}")
    info.append(f"Data types:\n{df.dtypes.to_string()}")
    info.append(f"Sample rows:\n{df.head(3).to_string()}")
    return "\n".join(info)

@tool
def query_dataframe(pandas_code: str) -> str:
    """Use this tool to query the dataframe using pandas code.
    The dataframe is available as 'df'.
    Examples:
    - df['revenue'].sum()
    - df.groupby('category_name')['profit'].mean()
    - df[df['country']=='United States']['revenue'].sum()
    Always write safe read-only pandas code."""
    try:
        result = eval(pandas_code, {"df": df, "pd": pd})
        return str(result)
    except Exception as e:
        return f"Error: {e}. Try a different pandas expression."

# ── Create LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key
)

# ── Create agent 
tools = [get_dataframe_info, query_dataframe]

agent = create_agent(model=llm, tools=tools)

# ── Ask multiple questions 
questions = [
    "What product category had the highest profit margin?",
    "Which country generated the most revenue?",
    "What is the average order quantity per product category?"
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # ── Print the full ReAct loop 
    print("\n--- Agent's reasoning ---")
    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            print(f"\n User: {msg.content}")
        elif msg_type == "AIMessage":
            if msg.content:
                print(f"\n AI thought: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n Tool called: {tc['name']}")
                    print(f"   Input: {tc['args']}")
        elif msg_type == "ToolMessage":
            print(f"\n Tool result: {msg.content[:200]}...")

    print(f"\n Final Answer: {result['messages'][-1].content}")