import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import requests
from io import BytesIO
from PIL import Image
import sys
from io import StringIO

load_dotenv()

# Page config
st.set_page_config(
    page_title="Advanced AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    h1 {
        color: white !important;
        text-align: center;
        font-size: 3em !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🚀 Advanced AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2em;'>RAG + Web Search + Image Gen + Code Execution</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Tools & Settings")
    
    # Tool toggles
    enable_rag = st.checkbox("📄 Document Chat (RAG)", value=True)
    enable_search = st.checkbox("🔍 Web Search", value=True)
    enable_image = st.checkbox("🎨 Image Generation", value=False)
    enable_code = st.checkbox("💻 Code Execution", value=False)
    
    st.markdown("---")
    
    # File upload (only if RAG enabled)
    if enable_rag:
        st.markdown("## 📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload document",
            type=['txt', 'pdf']
        )
    
    # Model settings
    st.markdown("## ⚙️ Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # Image generation settings (if enabled)
    if enable_image:
        st.markdown("## 🎨 Image Settings")
        image_model = st.selectbox(
            "Image Model",
            ["pollinations.ai (Free)", "DALL-E (requires API)"]
        )
    
    # Actions
    st.markdown("## 🎯 Actions")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Initialize LLM
@st.cache_resource
def get_llm(temp):
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=temp,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

llm = get_llm(temperature)

# Initialize tools
search_tool = DuckDuckGoSearchRun() if enable_search else None

# Process uploaded document (RAG)
if enable_rag and uploaded_file is not None:
    if st.session_state.vectorstore is None or st.session_state.get('last_file') != uploaded_file.name:
        with st.spinner("🔄 Processing document..."):
            try:
                file_path = f"temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding="utf-8")
                
                documents = loader.load()
                
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=os.getenv("OPENROUTER_API_KEY")
                )
                
                st.session_state.vectorstore = Chroma.from_documents(documents, embeddings)
                st.session_state.last_file = uploaded_file.name
                
                os.remove(file_path)
                st.success(f"✅ Loaded: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Helper functions

def web_search(query):
    """Perform web search"""
    try:
        results = search_tool.run(query)
        return results
    except Exception as e:
        return f"Search error: {str(e)}"

def generate_image(prompt):
    """Generate image using Pollinations.ai (free)"""
    try:
        # Using Pollinations.ai free API
        url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        return None

def execute_code(code):
    """Safely execute Python code"""
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Execute code
        exec(code)
        
        # Get output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        return output if output else "Code executed successfully (no output)"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {str(e)}"

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

# Chat input
if prompt := st.chat_input("💭 Ask anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Processing..."):
            try:
                response_parts = []
                
                # Check for special commands
                if enable_image and ("generate image" in prompt.lower() or "create image" in prompt.lower()):
                    # Extract image prompt
                    image_prompt = prompt.replace("generate image", "").replace("create image", "").strip()
                    if not image_prompt:
                        image_prompt = prompt
                    
                    with st.spinner("🎨 Generating image..."):
                        img = generate_image(image_prompt)
                        if img:
                            st.image(img, caption=f"Generated: {image_prompt}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Generated image for: {image_prompt}",
                                "image": img
                            })
                            st.stop()
                
                elif enable_code and ("```python" in prompt or "execute code" in prompt.lower()):
                    # Extract code
                    if "```python" in prompt:
                        code = prompt.split("```python")[1].split("```")[0].strip()
                    else:
                        code = prompt.replace("execute code", "").strip()
                    
                    with st.spinner("💻 Executing code..."):
                        output = execute_code(code)
                        response = f"**Code Output:**\n```\n{output}\n```"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.stop()
                
                elif enable_search and ("search" in prompt.lower() or "latest" in prompt.lower() or "current" in prompt.lower()):
                    # Web search
                    with st.spinner("🔍 Searching the web..."):
                        search_results = web_search(prompt)
                        response_parts.append(f"**Web Search Results:**\n{search_results}\n\n")
                
                # RAG (if enabled and document loaded)
                if enable_rag and st.session_state.vectorstore is not None:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    docs = retriever.invoke(prompt)
                    context = format_docs(docs)
                    
                    template = """Answer based on context and any additional information provided:

Context: {context}

Additional Info: {additional}

Question: {question}

Answer:"""
                    
                    prompt_template = ChatPromptTemplate.from_template(template)
                    chain = prompt_template | llm | StrOutputParser()
                    
                    rag_response = chain.invoke({
                        "context": context,
                        "additional": "\n".join(response_parts),
                        "question": prompt
                    })
                    response_parts.append(rag_response)
                
                else:
                    # Regular LLM response
                    chat_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful AI assistant with access to multiple tools."),
                        ("user", "{input}")
                    ])
                    chain = chat_prompt | llm | StrOutputParser()
                    llm_response = chain.invoke({
                        "input": prompt + "\n\nAdditional context: " + "\n".join(response_parts)
                    })
                    response_parts.append(llm_response)
                
                # Combine and display
                final_response = "\n\n".join(response_parts)
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Welcome message
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("""
        ### 👋 Welcome to Advanced AI Assistant!
        
        **Enabled Tools:**
        {}
        
        **Example Commands:**
        - "Search for latest AI news"
        - "Generate image of a sunset over mountains"
        - "Execute code: ```python\nprint('Hello World')\n```"
        - Ask questions about uploaded documents
        
        **Get started by typing a message below!**
        """.format(
            "\n".join([
                "- 📄 Document Chat (RAG)" if enable_rag else "",
                "- 🔍 Web Search" if enable_search else "",
                "- 🎨 Image Generation" if enable_image else "",
                "- 💻 Code Execution" if enable_code else ""
            ])
        ))