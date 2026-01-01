#!/usr/bin/env python
# coding: utf-8



# In[10]:


# streamlit_app.py - Lightweight deployment version
import streamlit as st
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import os
import chromadb


# In[11]:


@st.cache_resource
def load_rag_resources():
    """Load the model, tokenizer, and persistent DB client once."""
    model_id = "microsoft/phi-1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    
    gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="RAG_Assistant")
    
    return gen_pipeline, collection


# In[12]:


def generate_answer(query, context, gen_pipeline):
    """Phi-1.5 specific prompt formatting"""
    prompt = (
        f"Instruct: Answer the question based on the provided federal documents.\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        f"Output:"
    )
    results = gen_pipeline(prompt, max_new_tokens=300, temperature=0.1, return_full_text=False)
    return results[0]['generated_text'].strip()


# In[13]:


def main():
    st.set_page_config(page_title="FedAcq Chatbot", layout="wide")
    st.title("FedAcq Chatbot App")
    st.markdown("Query federal acquisition policies using RAG with Phi-1.5.")

    with st.spinner("Initializing models and database..."):
        gen_pipeline, collection = load_rag_resources()

    st.sidebar.title("Search Settings")
    n_results = st.sidebar.slider("Context Documents", 1, 5, 3)

    user_question = st.text_area(
        "Ask a question about federal acquisition policies:", 
        placeholder="e.g., What are the requirements for small business set-asides?"
    )

    if st.button("Get Answer"):
        if not user_question:
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving documents and generating response..."):
            results = collection.query(
                query_texts=[user_question], 
                n_results=n_results
            )

            retrieved_docs = results["documents"][0]
            retrieved_metas = results["metadatas"][0]
            
            context_segments = []
            for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
                source = meta.get('title', 'Unknown Source')
                context_segments.append(f"[Doc {i+1} - {source}]: {doc}")
            
            full_context = "\n\n".join(context_segments)

            answer = generate_answer(user_question, full_context, gen_pipeline)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View Source Documents"):
                for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
                    st.markdown(f"**Source {i+1}: {meta.get('title', 'N/A')}**")
                    st.caption(f"URL: {meta.get('source_url', 'N/A')}")
                    st.text(doc)
                    st.divider()

if __name__ == "__main__":
    main()
