# Conversational RAG Chatbot for CLAT Queries

A locally hosted chatbot leveraging Retrieval-Augmented Generation (RAG) to answer CLAT-related queries using a curated knowledge base. This assistant employs history-aware retrieval and a performant LLM (LLaMA-4 via Groq) to deliver accurate, document-grounded responses.

---

## Live Demo

To experience the chatbot:

1.  Navigate to the project directory.
2.  Ensure all requirements are installed (refer to the [Setup Instructions](#setup-instructions) section).
3.  Execute the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  Interact with the chatbot by entering your CLAT-related questions in the provided text input field.

---

## Architecture Overview

The system follows a Retrieval-Augmented Generation (RAG) architecture:

1.  **User Query:** The user inputs a question related to CLAT.
2.  **Query Reformulation (History Aware):** The input query is potentially reformulated, taking into account the conversation history to improve contextual relevance.
3.  **Retriever:** The reformulated query is used to search a vector store containing embeddings of the knowledge base documents.
4.  **Relevant Documents:** The retriever identifies and retrieves the most semantically similar document chunks from the knowledge base.
5.  **LLM (Groq/LLaMA-4):** The original or reformulated query, along with the retrieved relevant document chunks, are passed to the LLaMA-4 model hosted on Groq for answer generation.
6.  **Generated Answer with Sources:** The LLM generates a response grounded in the provided documents, including citations to the source material.
7.  **User:** The generated answer is presented to the user.

## Knowledge Base Creation

The knowledge base is constructed from manually collected data from the official CLAT portal ([https://consortiumofnlus.ac.in/](https://consortiumofnlus.ac.in/)) and other reliable public sources. This data undergoes AI-driven cleaning and refinement to ensure semantic clarity, metadata enrichment, and consistency. The current knowledge base comprises documents related to CLAT UG and PG for 2024 and 2025, as well as college cut-off information.

## Conversational RAG Flow

1.  **Document Ingestion and Chunking:** PDF and TXT documents are loaded and segmented into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve contextual information.
2.  **Vectorization and Storage:** The text chunks are converted into vector embeddings using the `all-MiniLM-L6-v2` (HuggingFace) model and stored in a local ChromaDB vector store for efficient similarity search.
3.  **Retriever Configuration:** A `Similarity` retriever is configured with specific parameters ($k=6$, $\lambda_{mult}=0.25$, $fetch\_k=20$) to retrieve the most relevant document chunks based on the query and conversation history.
4.  **LLM and Answer Generation:** The reformulated query and the top retrieved document chunks are fed into the Groq-hosted `meta-llama/llama-4-scout-17b` model. The LLM generates a response strictly based on the provided context, ensuring source attribution and avoiding unsupported claims.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/clat-rag-chatbot.git](https://github.com/your-username/clat-rag-chatbot.git)
    cd clat-rag-chatbot
    ```

2.  **Create .env File:**
    Create a `.env` file in the root directory and define your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key
    GEMINI_API_KEY=optional_if_using_gemini
    ```
    *(Note: The `GEMINI_API_KEY` is optional and only required if you intend to experiment with Gemini models.)*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Knowledge Base Files:**
    Place your `.pdf` and `.txt` files within the `/knowledge` directory. The application will automatically process these files.

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Features

- Locally hosted and operational without reliance on OpenAI.
- History-aware retrieval enhances conversational context understanding.
- Answers are grounded in provided documents with clear source citations.
- Utilizes Groq and LLaMA 4 for fast and cost-effective inference.
- Modular design allows for potential expansion to other domains.

## Future Enhancements

- Fine-tuning the system using content specific to NLTI guidance and mentorship.
- Implementing a user feedback mechanism for evaluating answer quality.
- Deployment on cloud platforms like Fly.io or Hugging Face Spaces for broader accessibility.
- Integration of voice input and text-to-speech output functionalities.

## Scaling to a GPT-Based Model Fine-tuned on NLTI Content

Scaling this architecture to incorporate a GPT-based model fine-tuned on NLTI-specific content can be approached through two primary strategies:

**Approach 1: Direct Fine-tuning and Application**

1.  **Dataset Acquisition and Annotation:** Gather a comprehensive dataset of CLAT-related materials, specifically focusing on NLTI-provided content such as study guides, past papers, mentorship session transcripts, and any other relevant textual data. This data needs careful annotation and preprocessing to align with the fine-tuning requirements of the chosen language model.
2.  **Model Selection and Fine-tuning:** Select a powerful language model such as a variant of GPT-3, GPT-3.5, GPT-4, or Meta Llama 3.x with sufficient capacity. Fine-tune this model on the prepared NLTI-specific dataset. This process will adapt the model's weights to better understand and generate responses relevant to NLTI's unique perspective and content.
3.  **Direct Application:** Once fine-tuned, this model can be directly used for question answering. User queries are input, and the fine-tuned model generates responses based on its learned knowledge.

**Approach 2: RAG with a Fine-tuned LLM and Scalable Vector Store**

1.  **Dataset Acquisition and Preprocessing:** Similar to the direct fine-tuning approach, gather and preprocess NLTI-specific CLAT content.
2.  **Model Selection and Fine-tuning:** Choose and fine-tune a suitable language model (e.g., GPT-3.5, Llama 3.x) on the prepared NLTI dataset.
3.  **Vector Store Implementation:** Utilize a scalable vector database such as Pinecone or Weaviate to store embeddings of the NLTI content. This allows for efficient retrieval of relevant information at scale.
4.  **RAG Integration:** When a user asks a question, the query is embedded and used to retrieve relevant chunks from the vector store. These chunks, along with the original query, are then passed to the fine-tuned LLM to generate a contextually accurate and NLTI-specific answer.
5.  **Cloud Deployment:** To ensure scalability and accessibility, the entire architecture (fine-tuned LLM and vector store) can be deployed on a cloud platform like AWS, Google Cloud, or Azure. This allows for handling a large volume of user requests and managing the underlying infrastructure effectively.

Both approaches offer pathways to leverage the capabilities of GPT-based models for NLTI-specific CLAT query answering. The RAG approach with a fine-tuned LLM and a scalable vector store offers better modularity, allows for continuous updates to the knowledge base without retraining the entire model, and can be more efficient in handling large datasets. The direct fine-tuning approach might offer more nuanced and integrated responses if the fine-tuning data is sufficiently comprehensive.

