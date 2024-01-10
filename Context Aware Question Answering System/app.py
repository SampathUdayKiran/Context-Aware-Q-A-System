import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

loaders = [
    PyPDFLoader("Context Aware Question Answering System\static\datasets\Environment_dataset_1.pdf")
    ,
    PyPDFLoader("Context Aware Question Answering System\static\datasets\Environment_dataset_2.pdf"),
]
pages = []
for loader in loaders:
    pages.extend(loader.load())


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
splits = text_splitter.split_documents(pages)  
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
persist_directory = 'db/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.Always give full  answer only from the data provided. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
question = "Explain "
result = qa_chain({"query": question})
result["result"]
print(result["result"])