#!/usr/bin/env python
# coding: utf-8



# In[10]:
!pip install accelerate

# streamlit_app.py - Lightweight deployment version
import streamlit as st
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import os

DB_FAISS_PATH = './saved_models/faiss_index_v0.2' # Folder containing the index files
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_KWARGS = {"device": "CPU"}
EMBEDDING_ENCODE_KWARGS = {"normalize_embeddings": True}
LLM_MODEL_NAME = "microsoft/phi-1.5"
LLM_MODEL_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
@st.cache_resource
def load_embedding_model():
    return OpenVINOEmbeddings(
        model_name_or_path=EMBEDDING_MODEL_NAME,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS)
@st.cache_resource
def load_faiss_index(embeddings):
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db
@st.cache_resource
def load_llm_pipeline():
    rag_pipeline = pipeline(
        "question-answering",
        model=LLM_MODEL_NAME,
        tokenizer=LLM_MODEL_NAME,
        device=-1 )
    return rag_pipeline


def main():
    st.set_page_config(page_title="FedAcq Chatbot", layout="wide")
    st.title("Federal Acquisition Chatbot")
    st.markdown("Query federal acquisition regulations using RAG with Phi-1.5.")

    with st.spinner("Initializing models and database..."):
        try:
            vector_store = load_faiss_index(embeddings)
            rag_pipeline = load_llm_pipeline()
            st.sidebar.success("LLM RAG Pipeline loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Error loading LLM RAG Pipeline: {e}")
            st.stop()

    st.sidebar.title("Search Settings")
    n_results = st.sidebar.slider("Context Documents", 0, 4, 3)

    user_question = st.text_area(
        "Ask a question about federal acquisition policies:",
        placeholder="e.g., What are the requirements for small business set-asides?"
    )

    if st.button("Get Answer"):
        if not user_question:
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving documents and generating response..."):
            results = vector_store.similarity_search(user_query, k=3)

            retrieved_docs = results["documents"][0]
            retrieved_metas = results["metadatas"][0]

            context_segments = []
            for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
                source = meta.get('title', 'Unknown Source')
                context_segments.append(f"[Doc {i + 1} - {source}]: {doc}")

            full_context = "\n\n".join(context_segments)

            answer = generate_answer(user_question, full_context, rag_pipeline)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View Source Documents"):
                for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
                    st.markdown(f"**Source {i + 1}: {meta.get('title', 'N/A')}**")
                    st.caption(f"URL: {meta.get('source_url', 'N/A')}")
                    st.text(doc)
                    st.divider()
