import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PDF_PATH = "data/your_document.pdf"  # ⚠️ UPDATE THIS WITH YOUR PDF FILENAME

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--color-primary);
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: var(--color-surface);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--color-primary);
        margin: 1rem 0;
    }
    .question-box {
        background-color: rgba(33, 128, 141, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource(show_spinner=False)
def load_and_process_pdf():
    """Load and process the fixed PDF (cached for efficiency)"""
    try:
        # Check if PDF exists
        if not os.path.exists(PDF_PATH):
            return None, 0, False, f"PDF file not found at: {PDF_PATH}"
        
        # Load PDF
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        if not documents:
            return None, 0, False, "No content found in PDF"
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings using free HuggingFace model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore, len(documents), True, "Success"
        
    except Exception as e:
        return None, 0, False, str(e)

def create_qa_chain(vectorstore, hf_token, model_name="google/flan-t5-large"):
    """Create the QA chain with custom prompt"""
    
    # Custom prompt template for better answers
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Provide a clear and concise answer.

Context: {context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Initialize HuggingFace LLM
    llm = HuggingFaceHub(
        repo_id=model_name,
        huggingfacehub_api_token=hf_token,
        model_kwargs={
            "temperature": 0.7,
            "max_length": 512,
            "max_new_tokens": 256
        }
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def get_answer(question, vectorstore, hf_token, model_name):
    """Get answer from the PDF using LLM"""
    try:
        qa_chain = create_qa_chain(vectorstore, hf_token, model_name)
        result = qa_chain({"query": question})
        
        return {
            'answer': result['result'],
            'sources': result['source_documents'],
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'answer': None,
            'sources': [],
            'success': False,
            'error': str(e)
        }

# Sidebar Configuration
with st.sidebar:
    st.image("https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png", width=100)
    st.title("⚙️ Configuration")
    
    # HuggingFace API token input
    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        value=os.environ.get("HUGGINGFACE_API_TOKEN", ""),
        help="Enter your free HuggingFace API token"
    )
    
    if not hf_token:
        st.warning("⚠️ API token required")
        st.markdown("[Get free token here →](https://huggingface.co/settings/tokens)")
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🤖 Model Selection")
    model_options = {
        "FLAN-T5-Large (Recommended)": "google/flan-t5-large",
        "FLAN-T5-Base (Faster)": "google/flan-t5-base",
        "FLAN-T5-XL (Better Quality)": "google/flan-t5-xl"
    }
    
    selected_model_name = st.selectbox(
        "Choose Model",
        options=list(model_options.keys()),
        help="Larger models give better answers but are slower"
    )
    selected_model = model_options[selected_model_name]
    
    st.markdown("---")
    
    # Instructions
    st.subheader("📖 How to Use")
    st.markdown("""
    1. ✅ Enter your HuggingFace token above
    2. ⏳ Wait for document to load
    3. ❓ Type your question
    4. 🎯 Get AI-powered answers!
    """)
    
    st.markdown("---")
    
    # Document info
    if st.session_state.loaded:
        st.success("✅ Document Loaded")
        if st.button("🔄 Reload Document"):
            st.cache_resource.clear()
            st.session_state.loaded = False
            st.rerun()
    
    st.markdown("---")
    
    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & HuggingFace")

# Main Content
st.markdown('<h1 class="main-header">📚 Document Q&A Assistant</h1>', unsafe_allow_html=True)
st.markdown("Ask questions about your document and get AI-powered answers")

# Load PDF on startup (cached)
if not st.session_state.loaded:
    with st.spinner("🔄 Loading and processing document... This may take a minute on first load."):
        vectorstore, num_pages, success, message = load_and_process_pdf()
        
        if success:
            st.session_state.vectorstore = vectorstore
            st.session_state.loaded = True
            st.success(f"✅ Document successfully loaded! ({num_pages} pages processed)")
        else:
            st.error(f"❌ Failed to load document: {message}")
            st.info("💡 Make sure your PDF is in the `data/` folder and update `PDF_PATH` in the code")
            st.stop()

# Main Q&A Interface
if st.session_state.loaded and st.session_state.vectorstore:
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Question")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is the main topic of this document?",
            height=100,
            key="question_input",
            label_visibility="collapsed"
        )
        
        # Buttons
        button_col1, button_col2 = st.columns([3, 1])
        with button_col1:
            ask_button = st.button("🔍 Get Answer", type="primary", disabled=not hf_token)
        with button_col2:
            if st.button("🔄 New Question"):
                st.rerun()
        
        if not hf_token:
            st.warning("⚠️ Please enter your HuggingFace API token in the sidebar to ask questions")
        
        # Process question
        if ask_button and question and hf_token:
            with st.spinner("🤔 Analyzing document and generating answer..."):
                result = get_answer(
                    question,
                    st.session_state.vectorstore,
                    hf_token,
                    selected_model
                )
                
                if result['success']:
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(f"""
                    <div class="answer-box">
                        {result['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result['answer'],
                        "sources": result['sources'],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "model": selected_model_name
                    })
                    
                    # Show source documents
                    if result['sources']:
                        with st.expander("📄 View Source Context (Click to expand)"):
                            st.markdown("*These are the relevant sections from the document used to generate the answer:*")
                            for i, doc in enumerate(result['sources']):
                                st.markdown(f"**📌 Source {i+1}:**")
                                st.text_area(
                                    f"source_{i}",
                                    value=doc.page_content,
                                    height=150,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                                if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                                    st.caption(f"📄 Page: {doc.metadata['page'] + 1}")
                                st.markdown("---")
                else:
                    st.error(f"❌ Error: {result['error']}")
                    st.info("💡 Try rephrasing your question or check your API token")
    
    with col2:
        st.markdown("### 💡 Example Questions")
        
        example_questions = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key findings?",
            "Explain the methodology used",
            "What are the conclusions?"
        ]
        
        st.markdown("Click to use:")
        for eq in example_questions:
            if st.button(eq, key=f"example_{eq}", use_container_width=True):
                st.session_state.question_input = eq
                st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.chat_history:
            st.markdown("### 📊 Session Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))

    # Chat History Section
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("## 📜 Chat History")
        
        # Display in reverse order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(
                f"❓ {chat['question'][:80]}{'...' if len(chat['question']) > 80 else ''} - {chat['timestamp']}",
                expanded=(i == 0)
            ):
                st.markdown(f"**Question:**")
                st.markdown(f'<div class="question-box">{chat["question"]}</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Answer:**")
                st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
                
                st.caption(f"🤖 Model: {chat['model']} | ⏰ Time: {chat['timestamp']}")

else:
    st.info("⏳ Initializing... Please wait for the document to load.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: var(--color-text-secondary); padding: 20px;'>
    <p>Powered by <strong>HuggingFace 🤗</strong> | Built with <strong>Streamlit 🎈</strong></p>
    <p style='font-size: 0.8rem;'>Using free and open-source models for document Q&A</p>
</div>
""", unsafe_allow_html=True)
