from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

model = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo"
) 

template = """
Answer the following question based only on the provided context:
Context: {context}
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)

document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
def get_retrieval_chain(vector_store):
    return create_retrieval_chain(vector_store.as_retriever(), document_chain)
