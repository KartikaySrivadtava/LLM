import os
import base64
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain

# -------------------- Load Environment --------------------
load_dotenv()

required_env_vars = [
    "OPENAI_API_KEY",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "APP_PASSWORD"  # Add your app password in .env
]

for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"❌ Missing required environment variable: {var}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# -------------------- Password Gate --------------------
st.set_page_config(page_title="Syntebot Agentic AI Sample Q&A", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Secure Login")
    password = st.text_input("Enter password:", type="password")
    if st.button("Login"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Login successful! Welcome.")
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    st.stop()  # 🚨 Stop the app here if not logged in

# -------------------- Layout Setup --------------------
col1, col2 = st.columns([0.7, 3.3])

# -------------------- Sidebar --------------------
with col1:
    st.markdown("<h2 style='color:#00B16A; margin-bottom: 0; align:center'>SAMI</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:black; margin-bottom: 0;'>Syntegon</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:black; margin-bottom: 0;'>Accounting</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:black; margin-bottom: 0;'>Manual</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:black; margin-bottom: 0;'>IFRS</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🕘 Past Prompts")

    FEEDBACK_XLSX = "feedback_log.xlsx"
    if os.path.isfile(FEEDBACK_XLSX):
        log_df = pd.read_excel(FEEDBACK_XLSX, engine="openpyxl")
        recent_queries = log_df["question"].dropna().tail(10).iloc[::-1]
        for i, query in enumerate(recent_queries):
            st.button(query, key=f"past_query_{i}")
    else:
        st.write("No previous queries yet.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Contacts:")
    st.markdown(
        """
        <div style='font-size: 14px; line-height: 1.6;'>
            <b>Kartikay Srivastava (FC-TO)</b><br>
            Kartikay.Srivastava@syntegon.com<br>
            <b>Julia Kreft (FC-TO)</b><br>
            Julia.Kreft@syntegon.com<br>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Main App --------------------
with col2:
    def get_base64_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    image_base64 = get_base64_image("Syntegon_Logo.png")
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{image_base64}' width='500'/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """<h1 style='text-align:left; font-size: 3em;'>
        <span style='color:black;'>Synte</span><span style='color:#00B16A;'>Bot</span></h1>""",
        unsafe_allow_html=True,
    )

    # -------------------- Embeddings & Vectorstores --------------------
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_key=OPENAI_API_KEY,
        chunk_size=500,
    )

    vectorstore = Chroma(persist_directory="./chroma_db_chunk_1000_overlap_150", embedding_function=embeddings)
    style_vectorstore = Chroma(persist_directory="./chroma_style_db", embedding_function=embeddings)

    def search_pdf(query: str, k: int = 5) -> str:
        results = vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])

    def retrieve_style_examples(query: str, k: int = 5) -> str:
        results = style_vectorstore.similarity_search(query, k=k)
        style_text = ""
        for i, doc in enumerate(results):
            style_text += f"Example {i+1}:\n{doc.page_content}\n\n"
        return style_text

    tools = [
        Tool(
            name="PDFRetriever",
            func=search_pdf,
            description="Useful for answering questions about the content of the PDF.",
        )
    ]

    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.2,
        max_tokens=3000,
    )

    prefix = """You are a helpful and professional research assistant.
You have access to a tool named PDFRetriever that lets you search an academic PDF.
Always use PDFRetriever to find accurate facts unless you are 100% sure.
You also have multiple examples of answers that follow a specific style.
Use these examples to write the answer in the same style, tone, and format.
Integrate the factual information with the style of the examples, paragraph by paragraph."""

    suffix = """Begin!

Question: {input}
Style Examples: {style_examples}
{agent_scratchpad}"""

    custom_prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad", "style_examples"],
    )

    llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    user_query = st.text_area(
        "Hi, I am able to answer questions related to our SAMI and I feature Agentic AI solution. \n\n How can I help you today?",
        value=st.session_state.get("user_query_prefill", ""),
        height=100,
    )

    if user_query:
        with st.spinner("Generating..."):
            style_examples = retrieve_style_examples(user_query)
            result = agent_executor.run({
                "input": user_query,
                "style_examples": style_examples
            })
            answer = result

            st.markdown("### 🤖 Answer")
            st.text_area("Answer", value=answer, height=500)

            # Save log
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            base_log = {
                "timestamp": timestamp,
                "question": user_query,
                "answer": answer,
                "feedback": "Not provided",
                "comments": "None",
            }

            FEEDBACK_XLSX = "feedback_log.xlsx"
            if os.path.isfile(FEEDBACK_XLSX):
                existing_df = pd.read_excel(FEEDBACK_XLSX, engine="openpyxl")
                updated_df = pd.concat([existing_df, pd.DataFrame([base_log])], ignore_index=True)
            else:
                updated_df = pd.DataFrame([base_log])
            updated_df.to_excel(FEEDBACK_XLSX, index=False, engine="openpyxl")

            # Feedback section
            st.markdown("---")
            st.markdown("### 🗳️ Feedback")
            feedback = st.radio("Did you find this answer helpful?", ["Like", "Dislike"], index=None)
            comments = st.text_area("Any additional feedback or suggestions? (Optional)")

            if st.button("Submit Feedback"):
                feedback_data = {
                    "timestamp": timestamp,
                    "question": user_query,
                    "answer": answer,
                    "feedback": feedback if feedback else "No response",
                    "comments": comments if comments else "None",
                }
                df = pd.read_excel(FEEDBACK_XLSX, engine="openpyxl")
                df.iloc[-1] = feedback_data
                df.to_excel(FEEDBACK_XLSX, index=False, engine="openpyxl")
                st.success("✅ Feedback submitted. Thank you!")
