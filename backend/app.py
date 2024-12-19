from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# load bge embeddings 
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# setting file path, just put filename
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "input_apport.pdf")

# loading pdf
loader = PDFMinerLoader(file_path)
docs = loader.load()
logger.info("Document loaded")

# Maximum number of tokens in a chunk
max_tokens = 256

# splitting the docs on basis of chunks
tokenizer = Tokenizer.from_pretrained("bert-base-uncased") 
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer,max_tokens)
chunks = splitter.chunks(docs[0].page_content)
logger.info("Chunking done")

logger.info(f"Number of chunks: {len(chunks)}")

# Build semantic vector database
semantic_vector_db = Chroma.from_texts(chunks, embedding=bge_embeddings)
logger.info("Document ingestion completed successfully")

# Checking number of documents in your Chroma collection
doc_count = semantic_vector_db._collection.count()

logger.info(f"Number of documents in Chroma: {doc_count}")




# loading chat model from groq via api
load_dotenv(find_dotenv())
os.environ["GROQ_API_KEY"] = os.getenv("groq_api_key")
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Message Models
class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    conversation: List[Message]

# Prompt Template
#  additional safety check is put to only give relevant responses
prompt_template = """
You are a helpful assistant. Answer the user's query based on the context provided. 
If the context does not seem relevant to the query, respond with: 
"Sorry, I didn't understand your question. Do you want to connect with a live agent?"

Context:
{context}

Query:
{query}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# similarity threshold, only those documents would be retrieved whose similarity is greater than 0.5 to input query
score_threshold = 0.5  # Example threshold value, adjust as needed

# Use the Chroma vectorstore as a retriever with a score threshold
retriever = semantic_vector_db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold}
)

# In-memory conversation tracking for rendering on the webpage
single_conversation = {"conversation": []}

# FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/inference/")
async def service(conversation: Conversation):
    global single_conversation

    # Append the latest user message
    latest_message = conversation.conversation[-1]
    single_conversation["conversation"].append(latest_message)

    query = latest_message.content
   
    if not query:
        raise HTTPException(status_code=400, detail="Query content cannot be empty.")

  
    # Retrieve relevant documents with the score threshold
    retrieved_docs = retriever.invoke(query)

        # Check if any documents were retrieved above the threshold
        # fallback check 
    if not retrieved_docs:
        assistant_response = "Sorry, I didn't understand your question. Do you want to connect with a live agent?"
    else:
        # If relevant docs exist, generate the assistant's response
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        chain = prompt | chat | StrOutputParser()
        assistant_response = chain.invoke({"query": query, "context": context})


    single_conversation["conversation"].append({"role": "assistant", "content": assistant_response})

    # Retain only the last 5 messages
    if len(single_conversation["conversation"]) > 6:
        single_conversation["conversation"] = single_conversation["conversation"][-6:]

    return {"conversation": single_conversation["conversation"]}
