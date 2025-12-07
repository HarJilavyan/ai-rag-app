import requests
import streamlit as st

BACKEND_URL = "http://backend:8000/chat"


st.set_page_config(page_title="RAG Demo UI", page_icon="💬", layout="centered")

st.title("RAG Chat Demo")
st.write("FastAPI backend + Qdrant + OpenAI + Streamlit frontend")

# Simple session-based chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts: {role, content}


def send_message_to_backend(message: str):
    try:
        payload = {
            "user_id": "demo-user",
            "message": message,
        }
        response = requests.post(BACKEND_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling backend: {e}")
        return None


# Chat input
user_input = st.text_area("Your message", height=100, placeholder="Ask about the system architecture, RAG, etc.")

col1, col2 = st.columns([1, 3])
with col1:
    send_btn = st.button("Send")

if send_btn and user_input.strip():
    # Append user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Call backend
    with st.spinner("Thinking..."):
        result = send_message_to_backend(user_input.strip())

    if result:
        reply = result.get("reply", "")
        used_context = result.get("used_context", [])
        latency_ms = result.get("latency_ms", None)

        st.session_state["messages"].append({"role": "assistant", "content": reply})

        # Show latest response nicely
        st.subheader("Assistant reply")
        st.markdown(reply)

        if used_context:
            st.subheader("Retrieved context")
            for idx, ctx in enumerate(used_context, start=1):
                st.markdown(f"**Chunk {idx}:** {ctx}")

        if latency_ms is not None:
            st.caption(f"Latency: {latency_ms} ms")


st.markdown("---")
st.subheader("Conversation history")

for msg in st.session_state["messages"]:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
    st.markdown(f"**{role}:** {msg['content']}")
