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

# Create a temp directory for the entire session instead of using a context manager
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

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
            
            # Use the session-level temp directory
            temp_dir = st.session_state.temp_dir
            
            # Check if we have pre-created embeddings in the git repo
            pre_created_dir = "./chroma_db"
            
            try:
                # Try to load existing embeddings
                if os.path.exists(pre_created_dir) and os.path.isdir(pre_created_dir):
                    try:
                        # First attempt: try to use the pre-created embeddings directly
                        vectorstore = Chroma(
                            persist_directory=pre_created_dir,
                            embedding_function=embeddings,
                            collection_metadata={"hnsw:space": "cosine"},
                            read_only=True  # Important: read-only mode
                        )
                        st.sidebar.success("Using read-only pre-created embeddings")
                    except Exception as e:
                        st.sidebar.warning(f"Could not use existing embeddings directly: {e}")
                        # Fall back to copying to temp dir
                        try:
                            for item in os.listdir(pre_created_dir):
                                s = os.path.join(pre_created_dir, item)
                                d = os.path.join(temp_dir, item)
                                if os.path.isdir(s):
                                    shutil.copytree(s, d, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(s, d)
                            
                            vectorstore = Chroma(
                                persist_directory=temp_dir,
                                embedding_function=embeddings,
                                collection_metadata={"hnsw:space": "cosine"}
                            )
                            st.sidebar.success("Loaded copied embeddings to temp directory")
                        except Exception as e:
                            st.sidebar.warning(f"Error copying embeddings: {e}. Creating new in-memory embeddings.")
                            vectorstore = Chroma.from_documents(
                                documents=splits,
                                embedding=embeddings,
                                collection_metadata={"hnsw:space": "cosine"}  # No persist_directory = in-memory
                            )
                else:
                    # Create new in-memory embeddings
                    st.sidebar.info("No pre-created embeddings found. Creating in-memory embeddings.")
                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        collection_metadata={"hnsw:space": "cosine"}  # No persist_directory = in-memory
                    )
            except Exception as e:
                st.sidebar.error(f"Error setting up vector store: {e}")
                # Final fallback: pure in-memory version
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    collection_metadata={"hnsw:space": "cosine"}  # No persist_directory = in-memory
                )
            
            # Save retriever to session state
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 6, 'lambda_mult': 0.25, "fetch_k": 20, "filter": None}
            )
            
            st.sidebar.success(f"Created vector store with {len(splits)} text chunks")
        else:
            st.session_state.retriever = None
            st.sidebar.warning("No documents loaded. The chatbot will not be able to answer questions.")

# The rest of your existing code follows here...
# ... 

if st.session_state.retriever:
    retriever = st.session_state.retriever
    
    contextualize_q_system_prompt = """
    Given a chat history and latest user question,
    which might reference context in chat history,
    reformulate it into a standalone question that can be understood
    without that chat history. Do not answer the question, just reformulate it
    if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    #The main prompt for the llm
    system_prompt = """
        You are a document-based question answering assistant. Your task is to provide accurate, well-structured answers strictly based on the provided documents. Follow these guidelines:
        REMEMBER : This is the year 2025

        1. SOURCE ATTRIBUTION:
        - Every factual statement must include a source reference.
        - Use the format: [Document Name, Page/Section] for each reference.
        - If the answer cannot be found in the provided documents, respond with: "I cannot find information about this in the provided documents."

        2. RESPONSE STRUCTURE:
        - Begin each response with: "Based on the provided documents:"
        - Organize the answer using clear headings if appropriate.
        - Use bullet points for clarity when listing multiple points.
        - Make sure to only answer about what is asked and do not deviate from the question.
        - If the user query is too vague or bad reformulate it as needed but do not make up your own asnwers only used the provided documents to answer

        3. METADATA HANDLING:
        - Use metadata actively to determine relevance (year, exam_type, section, topic,etc)
        - For information from different years than requested, clearly indicate: "While not from 2025, the 2024 syllabus includes..."

        3. ACCURACY REQUIREMENTS:
        - Only include information that is explicitly stated in the documents.
        - Do not infer, assume, or combine information to create new conclusions.
        - If the information is partial or incomplete, state what is missing or unclear.
        - Avoid duplicacy in your answers.

        4. TERMINOLOGY AND FORMATTING:
        - Use exact phrasing and terminology from the documents.
        - Maintain original formatting such as numbers, units, and names.

        5. LIMITATIONS:
        - Do not provide opinions or personal insights.
        - Do not speculate or guess missing details.
        - If the question is outside the scope of the documents, state this clearly.

        Context: {context}
        """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    def get_session_history(session: str) -> ChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Modified chain creation to expose source documents
    def create_chain():
        retrieval_chain = RunnablePassthrough.assign(
            context=history_aware_retriever.with_config(run_name="retrieve_documents")
        )
        
        response_chain = retrieval_chain.assign(
            answer=create_stuff_documents_chain(llm, qa_prompt)
        )
        
        return response_chain.with_config(run_name="retrieval_chain")

    chain = create_chain()
    
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Show status of the knowledge base in sidebar
    with st.sidebar:
        st.header("Knowledge Base Status")
        st.success("Knowledge base loaded and ready")
        
    
    user_input = st.text_input("Your question")

    if user_input:
        session_history = get_session_history(session_id)
        with st.spinner("Getting response..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
        st.write("### Assistant Response:")
        st.markdown(response['answer'])
        
        # Display source chunks
        st.write("### Source Chunks Used:")
        if 'context' in response:
            for idx, doc in enumerate(response['context']):
                with st.expander(f"Source Chunk {idx + 1}"):
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                    st.markdown("**Content:**")
                    st.markdown(doc.page_content)
        
else:
    st.error("No documents loaded in the knowledge base. Please add PDF or TXT files to the 'knowledge' folder and restart the application.")
