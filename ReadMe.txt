FedAcq-RAG-Chatbot
Federal Acquisition Regulation Retrieval-Augmented Generation (RAG) Chatbot
Problem:
Sought to create a chatbot to answer all questions related to federal acquisition policies and laws, providing a prototype for a tool that cites the specific statutes. This is a tool that can be used by private industry, the federal government, and can be a template for state, local, and tribal governments.

Simple Solution:
Create Library of Policies --> Query Library and Generate Answer with Citations --> Deployment of Data and Model to Chatbot

'Technical' Solution:
A. Create Library of Policies
(1) extract text from pdfs, (2) create chunks of data, (3) create collection of embedded data, (4) save data collection for RAG and application

B. Query Library and Generate Answer with Citations
(5) HuggingFace pipeline transformer leveraged Phi-1.5 as a pretrained large language model with Hugging face AutoTokenizer, (6) based on user input, the data  collection identifies relevant text, (7) a pipeline generator then generates text based on user input and identified relevant text; the relevant text is also provided as a citation (8) save the models for deployment

C. Deployment of Data and Model to Chatbot
(9) load the model, tokenizer, and data collection , (11) create a minimum viable product using Streamlit with the loaded model, tokenizer, and data referenced appropriately appropriately

Iterations:
v0.1: pdfplumber >> RecursiveCharacterTextSplitter >> ChromaDB >> HuggingFace >> Streamlit
v0.2: pdfplumber >> RecursiveCharacterTextSplitter >> FAISS >> HuggingFace >> Streamlit