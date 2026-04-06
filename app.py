import streamlit as st
import ollama
import pickle
import numpy as np


pickle_file = r"C:\Users\Client\Desktop\LUMINAR TECHNOLAB\ollama\ml.pkl"

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

chunks = data['chunks']
doc_embeddings = data["embeddings"]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if "history" not in st.session_state:
    st.session_state.history = []

if "selected" not in st.session_state:
    st.session_state.selected = None

st.title("📘 ML RAG Chatbot")
st.sidebar.title("🕘 History")

for i, (q, a) in enumerate(st.session_state.history):
    if st.sidebar.button(f"Q{i+1}: {q[:40]}..."):
        st.session_state.selected = i

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.history = []
    st.session_state.selected = None

with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("Enter your question:")
    submit = st.form_submit_button("Ask")

if submit and user_query:
    k = 8

    if user_query.lower() == "exit":
        st.session_state.history = []
        st.session_state.selected = None
        st.rerun()

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=user_query
    )
    query_embedding = response["embedding"]

    similarities = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    retrieved_chunks = [chunks[i] for i in top_k_indices]

    if not retrieved_chunks:
        answer = "No relevant information found."
    else:
        combined_context = "\n---\n".join(retrieved_chunks)

        prompt = f"""Answer the question using ONLY the context below.

Give a COMPLETE answer.

If multiple points exist (like types, steps, etc.), include ALL of them.

If not present, say: I don't have information about that.

CONTEXT:
{combined_context}

QUESTION: {user_query}
"""

        output = ollama.generate(
            model="gemma3:1b",
            prompt=prompt,
            options={"temperature": 0.0}
        )

        answer = output["response"]

    st.session_state.history.append((user_query, answer))
    st.session_state.selected = len(st.session_state.history) - 1

if st.session_state.selected is not None and st.session_state.history:
    q, a = st.session_state.history[st.session_state.selected]
    st.write("🧑‍💻 You:", q)
    st.write("🤖 Bot:", a)