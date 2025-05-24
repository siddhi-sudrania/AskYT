from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader 
load_dotenv()
urls=[
    'https://www.poornima.org',
      'https://www.poornima.org/contact-us',
      'https://www.poornima.org/admission/btech-at-poornima-group-of-colleges',
      'https://www.poornima.org/updates',
      'https://www.poornima.org/placement',
      'https://www.poornima.org/about-us/history-of-poornima',
      'https://www.poornima.org/placement/placement-statistics',
      'https://www.poornima.org/placement/rules-regulations'

]
all_docs=[]
embedding_model=HuggingFaceEmbeddings()
for url in urls:
    loader=WebBaseLoader(url)
    docs=loader.load()
    all_docs.extend(docs)
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
docs_split=splitter.split_documents(all_docs)
vectorstore=FAISS.from_documents(
    documents=docs_split,
    embedding=embedding_model
)
retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={'k':3,'lambda_mult':0.3})
llm=ChatGroq(model="llama3-8b-8192")
custom_prompt=PromptTemplate(
    template=""" you are a powerful and intelligent , friendly robot of poornima college of engineering (PCE). use only the information from the context provided below
    to answer the question clearly and directly. DO not say 'based on the context and provided information', if answer is not found say:"i am sorry, I couln't find that information.
    Context:{context},
    Question: {question}
    Answer:
""",input_variables=['context','question']
)
qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt":custom_prompt},
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True

)
while True:
    #name=input("May i know your name pls?")
    query=input("  Ask Your question? or type exit::")
    if query.lower()=='exit':
        break
    response=qa_chain.invoke({"question":query})
    print("Answer:",response["result"])

