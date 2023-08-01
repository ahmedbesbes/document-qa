from argparse import ArgumentParser
from dataclasses import dataclass
from time import time
from src import logger
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


@dataclass
class Config:
    file_path: str
    start_page: int
    end_page: int
    store: bool
    question: str
    top_k: int


def clean_page(page: Document):
    content = page.page_content
    lines = content.split("\n")
    header = lines[0]
    if "Chapter" in header or "Item" in header:
        clean_content = "\n".join(lines[1:])
        page.page_content = clean_content
    return page


def main(config: Config):
    loader = PyPDFLoader(file_path=config.file_path)
    pages = loader.load()
    logger.info(f"Data loaded successuflly: {len(pages)} pages")
    filtered_pages = [
        clean_page(page)
        for page in pages
        if config.start_page <= page.metadata["page"] <= config.end_page
    ]
    logger.info(
        f"Filtred page in the following rand: [{config.start_page}, {config.end_page}]"
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(filtered_pages)
    logger.info(f"Splitted documents into {len(chunks)} chunks")

    if bool(config.store):
        logger.info("Embedding the chunks and indexing them into Chroma ...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory="db",
        )
        logger.info("Chunks indexed into Chroma")
    else:
        vectorstore = Chroma(
            persist_directory="db",
            embedding_function=OpenAIEmbeddings(),
        )

    logger.info("Generating answer with LLM")

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    t1 = time()
    answer = qa_chain({"query": config.question})
    t2 = time()
    print(f"elapsed time: {t2-t1}")

    print(answer["result"])


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--file_path", type=str)
    argument_parser.add_argument("--start_page", type=int, default=23)
    argument_parser.add_argument("--end_page", type=int, default=63)
    argument_parser.add_argument("--store", type=bool, default=False)
    argument_parser.add_argument("--question", type=str)
    argument_parser.add_argument("--top_k", type=int, default=4)
    args = argument_parser.parse_args()
    config = Config(**vars(args))
    main(config)
