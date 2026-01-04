---
title: Federal Acquisition Policy Chatbot
emoji: 📜
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: false
---

# FedAcq-RAG-Chatbot
## Federal Acquisition Regulation Retrieval-Augmented Generation (RAG) Chatbot 

### Problem:
Sought to create a chatbot to answer all questions related to federal acquisition policies and laws, providing a prototype for a tool that cites the specific statutes. This is a tool that can be used by private industry, the federal government, and can be a template for state, local, and tribal governments.

### Simple Solution:
Create Library of Policies --> Query Library and Generate Answer with Citations --> Deployment of Data and Model to Chatbot

### 'Technical' Solution:
#### A. Create Library of Policies
(1) pdfplumber extracted text from pdfs, (2) RecursiveCharacterTextSplitter created chunks of text based on certain non-alphanumeric characters, (3) ChromaDB created a collection of embedded texts to be sorted through quickly, (4) the ChromaDB collection was then pickled to save all precomputed documents
#### B. Query Library and Generate Answer with Citations
(5) HuggingFace AutoTokenizer was loaded to improve model accuracy, (6) HuggingFace pipeline transformer leveraged Phi-1.5 as a pretrained large language model, (7) the ChromaDB collection was set to identify what documents answer a user's question, (8) the HuggingFace pipeline was then prompted to generate text based on a prompt that combines the user query and results of the ChromaDB collection information and include footnotes of the ChromaDB collection findings, (9) save the models for deployment
#### C. Deployment of Data and Model to Chatbot
(10) load  the model, tokenizer, and ChromaDB collection data, (11) create a minimum viable product using Streamlit with the loaded model, tokenizer, and data plugged in appropriately 
