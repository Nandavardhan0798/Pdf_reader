import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import tempfile

# Load environment variables
load_dotenv()

st.set_page_config(page_title="PDF QA Agent", page_icon="📄")
st.title("📄 PDF Question Answering Agent")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # --------------------------
    # 2. Load PDF
    # --------------------------
    loader = PDFPlumberLoader(tmp_file_path)
    documents = loader.load()

    # 2. Split PDF
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings + vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embeddings)

    # 4. Create retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 2})

    # 5. Define tool
    def ans_pdf(question: str) -> str:
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate(
            template="""You are a helpful assistant.
Please answer the following question using the provided content.
Question: {question}
Content: {content}
Only give the direct answer, no extra explanation.""",
            input_variables=['question', 'content']
        )

        chain = prompt | ChatOpenAI() | StrOutputParser()
        return chain.invoke({"question": question, "content": context})

    tools = [
        Tool(
            name="PDF_QA",
            func=ans_pdf,
            description="Use this to answer questions from the PDF content."
        )
    ]

    # 6. Setup memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 7. Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=ChatOpenAI(),
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # 8. Streamlit user input
    user_question = st.text_input("Ask a question about the PDF:")

    if user_question:
        with st.spinner("Agent is thinking..."):
            response = agent.run(user_question)
        st.markdown(f"**Answer:** {response}")
else:
    print("something went wrong")
  
