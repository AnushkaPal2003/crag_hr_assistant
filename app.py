import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
# Page config
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👔",
    layout="centered"
)

st.title("👔 HR Policy Assistant")
st.caption("Powered by CRAG · LangGraph · ChromaDB · OpenAI · Tavily · LangSmith")


# Session state init 

if "messages"  not in st.session_state: st.session_state.messages  = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "keys_set"  not in st.session_state: st.session_state.keys_set  = False
if "agent"     not in st.session_state: st.session_state.agent     = None
if "prefill"   not in st.session_state: st.session_state.prefill   = ""


# Sidebar

with st.sidebar:

    st.header("⚙️ API Keys")
    openai_key    = st.text_input("OpenAI API Key",           type="password", placeholder="sk-...")
    tavily_key    = st.text_input("Tavily API Key",           type="password", placeholder="tvly-...")
    langsmith_key = st.text_input("LangSmith Key (optional)", type="password")

    if st.button("Apply & Start", type="primary"):
        if openai_key and tavily_key:
            os.environ["OPENAI_API_KEY"]  = openai_key
            os.environ["TAVILY_API_KEY"]  = tavily_key
            if langsmith_key:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"]    = langsmith_key
                os.environ["LANGCHAIN_PROJECT"]    = "hr_assistant"

            # import and build ONLY after keys are set
            from graph.workflow import build_graph
            with st.spinner("Building agent..."):
                st.session_state.agent   = build_graph()
                st.session_state.keys_set = True
            st.success("Ready!")
            st.rerun()
        else:
            st.error("OpenAI and Tavily keys are required.")

    st.divider()

    st.header("🔄 How CRAG Works")
    st.markdown("""
    ```
    User Question
         ↓
    Retrieve Docs
         ↓
    Grade Relevance (LLM)
         ↓
    Relevant?
      Yes → Generate Answer
      No  → Rewrite Query
              ↓
          Web Search
              ↓
          Generate Answer
    ```
    """)

    st.divider()

    st.header("💡 Sample Questions")
    sample_questions = [
        "What is the leave encashment policy?",
        "How many sick leaves am I entitled to?",
        "What is the work from home policy?",
        "How does the performance review process work?",
        "What is the maternity leave policy?",
        "How do I apply for a promotion?",
        "What are the travel reimbursement rules?"
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.prefill = q
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")

# Gate — stop if keys not set
if not st.session_state.keys_set:
    st.info("Please enter your OpenAI and Tavily API keys in the sidebar to get started.")
    st.stop()


# Chat history display

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "source" in msg:
            icon = "🌐" if "Web" in msg["source"] else "📚"
            st.caption(f"{icon} Source: {msg['source']}")


# Chat input

prefill = st.session_state.pop("prefill", "")
prompt  = st.chat_input("Ask an HR policy question...", key="chat_input")

if prefill:
    prompt = prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR knowledge base..."):
            try:
                cfg    = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = st.session_state.agent.invoke(
                    {
                        "question"        : prompt,
                        "documents"       : [],
                        "generation"      : "",
                        "web_search_used" : False,
                        "rewritten_query" : "",
                        "messages"        : [HumanMessage(content=prompt)]
                    },
                    cfg
                )

                answer   = result["generation"]
                web_used = result.get("web_search_used", False)
                source   = "Web Search (Tavily)" if web_used else "HR Knowledge Base (ChromaDB)"

                st.write(answer)

                if web_used:
                    st.caption("🌐 Source: Web Search — knowledge base had no relevant docs, fell back to Tavily")
                else:
                    st.caption("📚 Source: HR Knowledge Base (ChromaDB)")

                st.session_state.messages.append({
                    "role"   : "assistant",
                    "content": answer,
                    "source" : source
                })

            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})