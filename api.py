from fastapi import FastAPI, UploadFile, File
import os
import shutil
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()


chat_history = InMemoryChatMessageHistory()
app = FastAPI()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

def load_rag():
    loader1 = TextLoader("docs/doc1.txt")
    loader2 = TextLoader("docs/doc2.txt")
    loader3 = TextLoader("docs/doc3.txt")

    doc1 = loader1.load()
    doc2 = loader2.load()
    doc3 = loader3.load()

    all_docs = doc1 + doc2 + doc3

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encoding_kwargs={"normalize_embeddings": True}
        cache_folder="/tmp/embeddings"
        )

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

vectorstore = load_rag()
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
threshold = 0.6

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Welcome to the AI Engineer API!"}


@app.post("/chat")
def chat(request: ChatRequest):
    response = vectorstore.similarity_search_with_score(request.question, k=3)
    relevant_docs = [(doc, score) for doc, score in response if score < threshold]

    if relevant_docs:
        context = "\n\n".join([doc.page_content for doc, score in relevant_docs])
        source_docs = list(set([doc.metadata['source'] for doc, score in relevant_docs]))

        llm_response = llm.invoke(
            [
                SystemMessage(content="Answer using only context provided, be concise."),
                HumanMessage(content=f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {request.question}\n\nPlease provide the answer and cite the sources from the context.")
            ]
        )
        return {"answer": llm_response.content, "sources": source_docs}
    else:
        return {"answer": "No relevant information found.", "sources": []}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = f"docs/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = TextLoader(file_path)
    new_docs = loader.load()
    new_chunks = splitter.split_documents(new_docs)

    vectorstore.add_documents(new_chunks)

    return {"message": f"File '{file.filename}' uploaded and processed successfully."}

@app.post("/chat-with-memory")
def chat_with_memory(request: ChatRequest):
    response = vectorstore.similarity_search_with_score(request.question, k=3)
    relevant_docs = [(doc, score) for doc, score in response if score < threshold]

    if relevant_docs:
        context = "\n\n".join([doc.page_content for doc, score in relevant_docs])
        source_docs = list(set([doc.metadata['source'] for doc, score in relevant_docs]))
    else:
        context = []
        source_docs = []

    message = []

    for msg in chat_history.messages:
        message.append(msg)
    
    message.append(HumanMessage(content=f"Question: {request.question} , context: {context}"))
    
    llm_response = llm.invoke(message)

    chat_history.add_user_message(request.question)
    chat_history.add_ai_message(llm_response.content)

    return {"answer": llm_response.content, "sources": source_docs}