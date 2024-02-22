from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

import shutil
import uuid
import os


def get_documents_from_msword(uri):
    loader = Docx2txtLoader(uri)
    docs = loader.load()

    return docs


def get_documents_from_pdf(uri):
    loader = PyPDFLoader(uri)
    docs = loader.load()

    return docs


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    return docs


def get_doc_splits(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    return split_docs


def get_vector_embeddings_from_docs(split_docs):
    return FAISS.from_documents(split_docs, embedding=OpenAIEmbeddings())


def get_context_and_embeddings(docs):
    split_docs = get_doc_splits(docs)
    vector_embeddings = get_vector_embeddings_from_docs(split_docs)

    context_id = uuid.uuid4().__str__()
    return {"context_id": context_id, "embeddings": vector_embeddings}


def process_file_and_get_docs(file):
    curr_dir = os.getcwd()

    # Create temporary directory for storing file
    os.mkdir(f"{curr_dir}/temp")

    file_path = f"{curr_dir}\\temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    file_type = os.path.splitext(file.filename)[1].lower()
    file_loc = f"temp/{file.filename}"

    if file_type == ".pdf":
        docs = get_documents_from_pdf(file_loc)
    elif file_type in (".doc", ".docx"):
        docs = get_documents_from_msword(file_loc)
    else:
        shutil.rmtree(f"{curr_dir}\\temp")
        raise ValueError("Unsupported file type")

    # Remove the temporary directory created for storing the file
    shutil.rmtree(f"{curr_dir}\\temp")

    return docs
