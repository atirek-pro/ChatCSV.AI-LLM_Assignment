from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Any

import os
import pandas as pd
from io import StringIO
import tempfile

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys


class Response(BaseModel):
    result: str | None

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(question: str = Form(...), csv_file: UploadFile = File(...)):
    # Read the contents of the CSV file
    contents = await csv_file.read()
    
    # Print the question and CSV contents
    print("Question:", question)

    # Convert contents to pandas DataFrame
    csv_string = str(contents, 'utf-8')
    csv_data = StringIO(csv_string)
    df = pd.read_csv(csv_data)

    # Save the CSV file temporarily
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
        df.to_csv(temp_file.name, index=False)
        csv_file_path = temp_file.name

    loader = CSVLoader(file_path=csv_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    print(data)

    # Split the text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    print(len(text_chunks))

    # Downloading Sentence Transformers Embedding From Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    # Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
    docsearch = FAISS.from_documents(text_chunks, embeddings)

    # Open-source LLama-2 Quantised model is used to generate the responses. 
    # To Download the model used visit https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q5_K_M.gguf
    # After downloading add the downloaded model in the models foler
    # Then add the path of the model as the value for the argument model as done bellow for llm variable.
    # It will load the model into the llm variable 

    llm = CTransformers(model="C:\\Users\\Lenovo\\Desktop\\llm-assignmen_By_AtirekGupta\\backend\\models\\llama-2-7b-chat.Q5_K_M.gguf",
                    model_type="llama",
                    config={'max_new_tokens': 600,
                              'temperature': 0.01,
                              'context_length': 700})
    
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
    chat_history=[]
    result = qa({"question":question, "chat_history":chat_history})
    response = result['answer']
    print(response)
    
    return {"result": response}