import glob
import os
import zipfile

import langchain
import streamlit as st
import tiktoken
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

langchain.debug = True

# loading from streamlit secrets
os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"]
os.environ["DEPLOYMENT_NAME"] = st.secrets["DEPLOYMENT_NAME"]
os.environ["EMBEDDING_NAME"] = st.secrets["EMBEDDING_NAME"]


def ask_and_get_answer_with_custom_input(vector_store, q, cs_info, cs_lps, k=3):
    """ask_and_get_answer_with_custom_input

    Args:
        vector_store (_type_): _description_
        q (_type_): _description_
        cs_info (_type_): customer inforamtion
        cs_lps (_type_): customer life plan simulation
        k (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """

    template = """
        - 指示: 君がファイナンシャルプランナー。提供された「カスタマー情報」、「ライフプランシミュレーション結果」、「コンテキスト」を参考し、カスタマーの質問を回答する。回答は詳しくしてください。注意点として、以下の制約条件をしたかう
        - 制約条件: 回答するとき、具体的な保険会社と保険商品を推薦しない。答えがわからない場合は単に「わかりません」と発言し、無理に回答を作ろうとしない
        - カスタマー情報:
        ------
        {cs_info}
        ------
        - ライフプランシミュレーション結果:
        ------
        {cs_lps}
        ------
        - コンテキスト:
        ------
        {context}
        ------
        - 入力質問: {question}
        - 出力指示: 日本語で答えなさい
        """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
        partial_variables={"cs_info": cs_info, "cs_lps": cs_lps},
    )

    # llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    llm = AzureChatOpenAI(
        client=None,
        # deployment_name="gpt-35-turbo",
        openai_api_type=os.environ["OPENAI_API_TYPE"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["DEPLOYMENT_NAME"],
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
        return_source_documents=True,
    )

    answer = chain(q)
    return answer


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
        openai_api_type=os.environ["OPENAI_API_TYPE"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["DEPLOYMENT_NAME"],
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


def create_embeddings(chunks):
    """Create embeddings and save them in a Chroma vector store"""
    embeddings = OpenAIEmbeddings(deployment=os.environ["EMBEDDING_NAME"])
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store


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


def read_upload_file(uploaded_file):
    """writing the file from RAM to the current directory on disk"""
    bytes_data = uploaded_file.read()
    file_name = os.path.join("./", uploaded_file.name)
    with open(file_name, "wb") as f:
        f.write(bytes_data)
    # data = load_document(file_name)
    with open(file_name, "r") as file:
        file_content = file.read()
    return file_content


def read_plain_txt(file_name):
    with open(file_name, "r") as file:
        file_content = file.read()
    return file_content


def custom_chunk_data(file_name, file_content):
    """Chunk data with custom delimiter

    Args:
        data (_type_): _description_
    """

    chunks_with_doc_type = []
    chunks = file_content.split("---------------------")
    for chunk in chunks:
        new_doc = Document(page_content=chunk.strip(), metadata={"source": file_name})
        chunks_with_doc_type.append(new_doc)
    return chunks_with_doc_type


# def dump_files_to_disk(uploaded_file, context_path="./context"):
#     if not os.path.exists(context_path):
#         os.mkdir(context_path)
#     if len(uploaded_file) > 0:
#         for file in uploaded_file:
#             # If zip file, extract contents
#             if file.type == "application/zip":
#                 with zipfile.ZipFile(file, "r") as z:
#                     z.extractall(context_path)
#             else:
#                 bytes_data = uploaded_file.read()
#                 file_name = os.path.join(context_path, uploaded_file.name)
#                 with open(file_name, "wb") as f:
#                     f.write(bytes_data)

#     return os.path.join(context_path, uploaded_file.name)
#     # return context_path


def dump_file_to_disk(uploaded_file, context_path):
    """writing the file from RAM to the current directory on disk"""
    bytes_data = uploaded_file.read()
    file_name = os.path.join(context_path, uploaded_file.name)
    with open(file_name, "wb") as f:
        f.write(bytes_data)
    return file_name


def dump_files_to_disk(uploaded_files, context_path="./context"):
    if not os.path.exists(context_path):
        os.mkdir(context_path)
    file_list = []
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            file_name = dump_file_to_disk(uploaded_file, context_path)
            file_list.append(file_name)
    return file_list


def get_chunks_with_context_data(uploaded_file):
    file_list = dump_files_to_disk(uploaded_file)

    chunks_with_doc_type = []
    for txt_file in file_list:
        file_name = os.path.splitext(os.path.basename(txt_file))[
            0
        ]  # get file name without extension
        with open(txt_file, "r") as file:
            file_content = file.read()
            chunked_documents = custom_chunk_data(file_name, file_content)
            chunks_with_doc_type.extend(chunked_documents)

    return chunks_with_doc_type


if __name__ == "__main__":

    # # loading the OpenAI api key from .env
    # load_dotenv(find_dotenv(), override=True)
    cs_info = ""
    cs_lps = ""

    st.image("img.jpg")
    st.subheader("ChatGPT with Documents 🤖")
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        # OPENAI_API_BASE = st.text_input("OPENAI_API_BASE", type="password")
        # if OPENAI_API_BASE:
        #     os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

        # OPENAI_API_KEY = st.text_input("OPENAI_API_KEY", type="password")
        # if OPENAI_API_KEY:
        #     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # DEPLOYMENT_NAME = st.text_input("DEPLOYMENT_NAME", type="password")
        # if DEPLOYMENT_NAME:
        #     os.environ["DEPLOYMENT_NAME"] = DEPLOYMENT_NAME

        # # show the environment variables
        # # st.write("OPENAI_API_TYPE:", os.environ["OPENAI_API_TYPE"])
        # st.write("OPENAI_API_BASE:", os.environ["OPENAI_API_BASE"])
        # st.write("OPENAI_API_KEY:", os.environ["OPENAI_API_KEY"])
        # # st.write("OPENAI_API_VERSION:", os.environ["OPENAI_API_VERSION"])
        # st.write("DEPLOYMENT_NAME:", os.environ["DEPLOYMENT_NAME"])

        # file uploader widget
        uploaded_file_context = st.file_uploader(
            "Upload a zip file:",
            type=["docx", "txt", "pdf"],
            accept_multiple_files=True,
        )

        # file uploader widget
        uploaded_file_cs_info = st.file_uploader(
            "Upload a file about customer information:", type=["txt"]
        )

        # file uploader widget
        uploaded_file_cs_lps = st.file_uploader(
            "Upload a file about customer life plan simulation result:", type=["txt"]
        )

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

        if uploaded_file_context and add_data:  # if the user browsed a file
            with st.spinner("Reading, chunking and embedding file ..."):

                # read txt file from uploaded file
                if uploaded_file_cs_info and uploaded_file_cs_lps:
                    cs_info = read_upload_file(uploaded_file_cs_info)
                    cs_lps = read_upload_file(uploaded_file_cs_lps)

                # # writing the file from RAM to the current directory on disk
                # bytes_data = uploaded_file.read()
                # file_name = os.path.join("./", uploaded_file.name)
                # with open(file_name, "wb") as f:
                #     f.write(bytes_data)

                # data = load_document(file_name)

                # chunks = chunk_data(data, chunk_size=chunk_size)
                chunks = get_chunks_with_context_data(uploaded_file_context)
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
            if cs_info and cs_lps:
                answer = ask_and_get_answer_with_custom_input(
                    vector_store, q, cs_info, cs_lps, k
                )
            else:
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
