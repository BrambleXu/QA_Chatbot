import os

import streamlit as st
import tiktoken
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# loading from streamlit secrets
os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"]
os.environ["DEPLOYMENT_NAME"] = st.secrets["DEPLOYMENT_NAME"]


def ask_and_get_answer(vector_store, q, k=3):
    """Ask a question and return the answer"""
    template = """
        You are asked to answer the following question based on the context.
        ------
        {context}
        ------
        Question: {question}
        Answer:
        """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = AzureChatOpenAI(
        client=None,
        # deployment_name="gpt-35-turbo",
        deployment_name=os.environ["DEPLOYMENT_NAME"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_version=os.environ["OPENAI_API_VERSION"] or "",
        openai_api_key=os.environ["OPENAI_API_KEY"] or "",
        temperature=1,
        request_timeout=180,
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    answer = chain.run(q)
    return answer


def load_document(file):
    """Loading PDF, DOCX and TXT files as LangChain Documents"""

    _, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader

        print(f"Loading {file}")
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader

        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain.document_loaders import TextLoader

        loader = TextLoader(file)
    else:
        print("Document format is not supported!")
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """Splitting data in chunks"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    """Create embeddings and save them in a Chroma vector store"""
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store


def calculate_embedding_cost(texts):
    """Calculate embedding cost using tiktoken"""

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    """Clear the chat history from streamlit session state"""
    if "history" in st.session_state:
        del st.session_state["history"]


if __name__ == "__main__":

    # # loading the OpenAI api key from .env
    # load_dotenv(find_dotenv(), override=True)

    st.image("img.jpg")
    st.subheader("ChatGPT with Documents 🤖")
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        # api_key = st.text_input("OpenAI API Key:", type="password")
        # if api_key:
        #     os.environ["OPENAI_API_KEY"] = api_key

        # show the environment variables
        st.write("DB username:", os.environ["OPENAI_API_TYPE"])
        st.write("DB password:", os.environ["OPENAI_API_BASE"])
        st.write("My cool secrets:", os.environ["OPENAI_API_KEY"])

        # file uploader widget
        uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])

        # chunk size number widget
        chunk_size = st.number_input(
            "Chunk size:",
            min_value=100,
            max_value=2048,
            value=512,
            on_change=clear_history,
        )

        # k number input widget
        k = st.number_input(
            "k", min_value=1, max_value=20, value=3, on_change=clear_history
        )

        # add data button widget
        add_data = st.button("Add Data", on_click=clear_history)

        if uploaded_file and add_data:  # if the user browsed a file
            with st.spinner("Reading, chunking and embedding file ..."):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"Embedding cost: ${embedding_cost:.4f}")

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success("File uploaded, chunked and embedded successfully.")

    # user's question text input widget
    q = st.text_input("Upload a file and ask a question about the content:")
    if q:  # if the user entered a question and hit enter
        if (
            "vs" in st.session_state
        ):  # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f"k: {k}")
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area("LLM Answer: ", value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            if "history" not in st.session_state:
                st.session_state.history = ""

            # the current question and answer
            value = f"Q: {q} \nA: {answer}"

            st.session_state.history = (
                f'{value} \n {"-" * 100} \n {st.session_state.history}'
            )
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label="Chat History", value=h, key="history", height=400)
