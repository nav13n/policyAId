import os
from operator import itemgetter
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import CustomUserType
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceEndpoint

# Setup Embedding and LLM Models
model_api_gateway = os.environ["HF_MODEL_API_GATEWAY"] 
endpoint_url = (
    model_api_gateway
)

llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    task="text-generation"
)

embedding_api_gateway = os.environ["HF_EMBEDDING_API_GATEWAY"] 
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HF_TOKEN"], 
    api_url=embedding_api_gateway)

# # Load Data
def load_multiple_pdfs(pdf_paths):
    # This list will hold the loaded PDF content or document objects
    loaded_pdfs = []

    for path in pdf_paths:
        try:
            loader = PyMuPDFLoader(path)
            pdf_content = loader.load()
            loaded_pdfs.append(pdf_content)
            print(f"Successfully loaded PDF: {path}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    return loaded_pdfs

pdf_paths= ["./data/HRPolicyManual2023-11-208.pdf"]
documents = load_multiple_pdfs(pdf_paths)


# # Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 50
)

all_splits = []

for doc in documents:
    split_text = text_splitter.split_documents(doc)
    all_splits.extend(split_text)

# # Ingest into VectorStore 
embeddings = []
for i in range(0, len(all_splits) - 1, 32):
    embeddings.append(
        embeddings_model.embed_documents(
            [document.page_content for document in all_splits[i:i+32]]
        )
    )
embeddings = [item for sub_list in embeddings for item in sub_list]

text_embedding_pairs = list(zip([document.page_content for document in all_splits], embeddings))
vector_store = FAISS.from_embeddings(text_embedding_pairs, embeddings_model)
retriever = vector_store.as_retriever(search_kwargs={"k" : 4})


RAG_PROMPT_TEMPLATE = """\
Using the provided context, please answer the user's question. If you don't know, say you don't know.

Context:
{context}

Question:
{question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": itemgetter("question") | retriever 
    }
).with_types(input_type=ChatHistory)


chain = _inputs | rag_prompt | llm | StrOutputParser()