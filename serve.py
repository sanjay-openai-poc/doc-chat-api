from fastapi import FastAPI, UploadFile, Form, File
from typing import Annotated

from models.Query import Query
from models.UrlContext import UrlContext
from utils import (
    get_documents_from_web,
    get_context_and_embeddings,
    process_file_and_get_docs,
)
from retriever import get_retrieval_chain
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server Started")
    yield
    print("Server Shutting down")


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
    lifespan=lifespan,
)


persistent_store = {}


@app.post("/query")
async def query_doc(query: Query):
    try:
        if query.context_id:
            global persistent_store
            vector_embeddings = persistent_store.get(query.context_id)

            if vector_embeddings:
                chain = get_retrieval_chain(vector_embeddings)
                response = chain.invoke(
                    {
                        "input": query.query,
                    }
                )
                return response
            else:
                return f"Could not find context for the provided context_id: {query.context_id}"
        else:
            return "Invalid Query, must include context_id"
    except Exception as e:
        return f"Exception occurred: {e}"


@app.post("/set/context/url")
async def set_url_context(context: UrlContext):
    try:
        docs = get_documents_from_web(context.value)
        if docs:
            context_id, embeddings = get_context_and_embeddings(docs).values()

            set_global_context(context_id, embeddings)

            return {"message": "Context set successfully", "context_id": context_id}
        else:
            return f"Cound not retrieve documents from: {context.value}"
    except Exception as e:
        return f"Exception occurred: {e}"


# TODO: Identify file type and parse accordingly

@app.post("/set/context/file")
async def upload_file_with_json_data(file: UploadFile):
    try:
        docs = process_file_and_get_docs(file)
        context_id, embeddings = get_context_and_embeddings(docs).values()

        set_global_context(context_id, embeddings)
        return {
            "message": "Context set successfully",
            "context_id": context_id,
        }
    except Exception as e:
        return f"Exception occurred: {e}"


def set_global_context(context_id, embeddings):
    global persistent_store
    persistent_store.update({context_id: embeddings})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
