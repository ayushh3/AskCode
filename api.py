from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from rag import (
    answer_question,
    get_indexed_sources,
    get_provider,
    index_document,
    remove_document,
)


app = FastAPI(title="AskCode API", version="2.0.0")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "provider": get_provider()}


@app.get("/documents")
def list_documents():
    return {"documents": get_indexed_sources()}


@app.post("/index")
async def index_files(files: List[UploadFile] = File(...)):
    indexed = []
    for uploaded_file in files:
        contents = await uploaded_file.read()
        try:
            sections = await run_in_threadpool(
                index_document,
                uploaded_file.filename or "upload.txt",
                contents,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        indexed.append({"filename": uploaded_file.filename, "sections": sections})
    return {"indexed": indexed}


@app.post("/ask")
async def ask(request: AskRequest):
    try:
        answer, sources = await run_in_threadpool(answer_question, request.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"answer": answer, "sources": sources}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    remove_document(filename)
    return {"removed": filename}
