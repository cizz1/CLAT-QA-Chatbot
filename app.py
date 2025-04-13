import sys
import importlib.util
import os
import tempfile
import shutil

# SQLite replacement - keep this at the top
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-l6-V2")

st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")
st.title("Conversational RAG Chatbot")

llm = ChatGroq(
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),  
    model_name="meta-llama/llama-4-scout-17b-16e-instruct" 
)

session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

if 'retriever' not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        knowledge_dir = "./knowledge"
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)
            st.sidebar.warning(f"Created empty knowledge directory at {knowledge_dir}. Please add your PDF and TXT files there.")
        
        if len(os.listdir(knowledge_dir)) == 0:
            st.sidebar.warning(f"Knowledge directory '{knowledge_dir}' is empty. Please add PDF or TXT files.")
            documents = []
        else:
            st.sidebar.info(f"Loading documents from {knowledge_dir}...")
            
            pdf_loader = DirectoryLoader(
                knowledge_dir, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
             
            txt_loader = DirectoryLoader(
                knowledge_dir, 
                glob="**/*.txt", 
                loader_cls=TextLoader
            )
            
            documents = pdf_loader.load() + txt_loader.load()
            st.sidebar.success(f"Loaded {len(documents)} documents")
        
        # Process documents 
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=500,
                add_start_index=True,
            )
            splits = text_splitter.split_documents(documents)
            
            # Create a temporary directory for ChromaDB that is writable
            with tempfile.TemporaryDirectory() as temp_dir:
                # Check if we have pre-created embeddings in the git repo
                pre_created_dir = "./chroma_db"
                
                # If pre-created embeddings exist, try to copy them to the temp directory
                if os.path.exists(pre_created_dir) and os.path.isdir(pre_created_dir):
                    try:
                        for item in os.listdir(pre_created_dir):
                            s = os.path.join(pre_created_dir, item)
                            d = os.path.join(temp_dir, item)
                            if os.path.isdir(s):
                                shutil.copytree(s, d)
                            else:
                                shutil.copy2(s, d)
                        
                        # Try to load existing embeddings
                        try:
                            vectorstore = Chroma(
                                persist_directory=temp_dir,
                                embedding_function=embeddings,
                                collection_metadata={"hnsw:space": "cosine"}
                            )
                            st.sidebar.success("Loaded pre-created embeddings")
                        except Exception as e:
                            st.sidebar.warning(f"Could not load existing embeddings, creating new ones: {e}")
                            vectorstore = Chroma.from_documents(
                                documents=splits,
                                embedding=embeddings,
                                persist_directory=temp_dir,
                                collection_metadata={"hnsw:space": "cosine"}
                            )
                    except Exception as e:
                        st.sidebar.warning(f"Error copying embeddings: {e}. Creating new embeddings.")
                        vectorstore = Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings,
                            persist_directory=temp_dir,
                            collection_metadata={"hnsw:space": "cosine"}
                        )
                else:
                    # Create new embeddings if no pre-created ones exist
                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory=temp_dir,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                
                # Save retriever to session state - note we don't need persist() call since the temporary
                # directory will be used during this session only
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={'k': 6, 'lambda_mult': 0.25, "fetch_k": 20, "filter": None}
                )
                
                # Keep the temporary directory reference alive for the session
                st.session_state.temp_dir = temp_dir
                
                st.sidebar.success(f"Created vector store with {len(splits)} text chunks")
        else:
            st.session_state.retriever = None
            st.sidebar.warning("No documents loaded. The chatbot will not be able to answer questions.")
