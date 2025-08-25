import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import agent as langgraph_agent
import io
import contextlib

# --- Page Configuration ---
st.set_page_config(
    page_title="CSV Analysis Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Title and Description ---
st.title("🤖 Conversational CSV Analysis Agent")
st.markdown("""
Welcome! This is a demo of a LangGraph agent that can analyze CSV files.
Upload a CSV, and then ask questions about it in the chat.
The agent can display text, dataframes, and even generate plots.
""")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"  # A simple thread ID for this session
if "agent_app" not in st.session_state:
    st.session_state.agent_app = None  # To store the initialized agent

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        try:
            # Read the csv to get headers and save it with a consistent name
            # Use a fresh buffer to avoid issues with already-read files
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            df.to_csv("uploaded_data.csv", index=False)
            csv_headers = df.columns.tolist()

            # Create and store the agent in session state.
            st.session_state.agent_app = langgraph_agent.create_csv_agent(csv_headers)

            st.success("File uploaded and agent is ready!")
            st.info(f"Columns found: {', '.join(csv_headers)}")
            # Clear previous chat history on new file upload
            st.session_state.messages = []

        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- Main Chat Interface ---
st.header("Chat with the Agent")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle different content types
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        elif isinstance(message["content"], go.Figure):
            st.plotly_chart(message["content"], use_container_width=True)
        else:
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your data..."):
    if st.session_state.agent_app is None:
        st.warning("Please upload a CSV file first to initialize the agent.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent's response
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    response = st.session_state.agent_app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config
                    )

                # --- REVISED LOGIC FOR EXTRACTING AND DISPLAYING ---

                # Extract artifacts and code from the intermediate steps
                tool_outputs = []
                generated_codes = []

                for message in response['messages']:
                    # AIMessage with tool_calls contains the generated code
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for tool_call in message.tool_calls:
                            if isinstance(tool_call.get('args'), dict) and 'code' in tool_call['args']:
                                generated_codes.append(tool_call['args']['code'])

                    # ToolMessage contains the output of the tool
                    if isinstance(message, ToolMessage):
                        # Handle dataframes directly
                        if isinstance(message.content, pd.DataFrame):
                            tool_outputs.append(message.content)
                        # Handle plotly JSON strings
                        elif isinstance(message.content, str):
                            try:
                                # Attempt to parse the string as JSON
                                plot_json = json.loads(message.content)
                                # Check if it looks like a plotly figure
                                if 'data' in plot_json and 'layout' in plot_json:
                                    fig = pio.from_json(message.content)
                                    tool_outputs.append(fig)
                            except (json.JSONDecodeError, TypeError):
                                # Not a valid JSON or not a plot, ignore
                                pass

                # Display generated code if any
                if generated_codes:
                    with st.expander("Show Generated Code"):
                        for code_block in generated_codes:
                            st.code(code_block, language='python')

                # Display captured stdout from the tool
                stdout_output = f.getvalue()
                if stdout_output:
                    st.code(stdout_output, language='text')

                # Display the final text response from the agent
                final_response = response['messages'][-1].content
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

                # Display any dataframes or plots returned by tools
                for output in tool_outputs:
                    if isinstance(output, pd.DataFrame):
                        st.dataframe(output)
                        st.session_state.messages.append({"role": "assistant", "content": output})
                    elif isinstance(output, go.Figure):
                        st.plotly_chart(output, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": output})
