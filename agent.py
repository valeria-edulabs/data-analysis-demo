import os
from typing import List

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


load_dotenv()

# To use Gemini, you need to set up Google Cloud authentication.
# In your terminal, you can run:
# gcloud auth application-default login
# Make sure to set your GOOGLE_API_KEY if you're not using ADC
# os.environ["GOOGLE_API_KEY"] = "your_google_api_key"

def create_csv_agent(csv_headers: List[str]):
    """
    Creates a LangGraph ReAct agent specifically for CSV analysis.

    Args:
        csv_headers: A list of strings representing the column headers of the CSV.

    Returns:
        A compiled LangGraph agent.
    """

    # This system prompt is crucial for guiding the agent's behavior.
    # It instructs the agent to use pandas and plotly and informs it of the CSV schema.
    system_prompt = f"""
    You are a powerful data analysis assistant. You are an expert in using the Python programming language for data manipulation and visualization.

    You have access to a Python REPL tool which you should use to answer the user's questions.

    When analyzing data, you MUST use the pandas library.
    When asked to create visualizations, you MUST use the plotly library.

    IMPORTANT: To display a plot, you must return the figure's JSON representation. Make `fig.to_json()` the final expression in your code. The Streamlit environment will parse this JSON to render the plot. DO NOT call `fig.show()` or return the raw figure object.

    A file named 'uploaded_data.csv' is available in the environment. It has the following columns: {', '.join(csv_headers)}.

    When the user asks a question, think step-by-step about how to answer it using Python, pandas, and plotly. Generate the necessary code and execute it using the Python REPL tool.

    Provide a clear, natural language explanation of your findings as the final answer. If you generate a plot or a table, also provide a brief summary of what it shows.
    """

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    tools = [PythonAstREPLTool()]

    # Create the ReAct agent with the custom system prompt
    agent_app = create_react_agent(llm, tools=tools, prompt=system_prompt, checkpointer=MemorySaver())
    return agent_app


