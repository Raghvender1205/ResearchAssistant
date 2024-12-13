import re
import os
import requests
import logging
import google.generativeai as genai

from typing import List, Tuple
from dotenv import load_dotenv, find_dotenv
from googlesearch import search
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("script_logs.log", encoding="utf-8")
    ]
)

# Setup Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0
)


def perform_search(query, n_results=20, pause=2.0):
    """
    Perform Google Search and get urls
    :param query:
    :param n_results:
    :param pause:
    :return:
    """
    try:
        logging.info(f"Performing search for: {query}")
        search_results = list(search(query, num_results=n_results, sleep_interval=pause))
        logging.info(f"Found {len(search_results)} results")

        return search_results
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return []


def extract_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Extract normal URLs and arxiv URLs from a list of URLs
    :param urls (List[str]): List of URLs
    :return: List of normal and arxiv URLs
    """
    normal_urls = [url for url in urls if re.match(r'^https?://', url) and 'arxiv.org' not in url]
    arxiv_urls = [url for url in urls if 'arxiv.org' in url]

    logging.info(f"Extracted {len(normal_urls)} normal URLs and {len(arxiv_urls)} arXiv URLs")
    return normal_urls, arxiv_urls


def download_pdf(url, local_file):
    try:
        # Convert arxiv abs URL to PDF URL
        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
        logging.info(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        with open(local_file, 'wb') as f:
            f.write(response.content)
        logging.info(f"Downloaded PDF saved as: {local_file}")
    except Exception as e:
        logging.error(f"Error downloading PDF from {url}: {e}")


async def load_documents(urls, arxiv_urls):
    docs = []
    local_filename = "/tmp/file.pdf"

    # Load normal URLs
    for url in urls:
        try:
            logging.info(f"Loading document from normal URL: {url}")
            loader = WebBaseLoader(web_paths=[url])
            async for doc in loader.alazy_load():
                docs.append(doc)
        except Exception as e:
            logging.error(f"Error loading document from {url}: {e}")

    # Load arXiv URLs
    for url in arxiv_urls:
        try:
            logging.info(f"Loading document from arXiv URL: {url}")
            download_pdf(url, local_filename)
            loader = PyMuPDFLoader(local_filename)
            async for doc in loader.alazy_load():
                docs.append(doc)
        except Exception as e:
            logging.error(f"Error loading document from {url}: {e}")

    logging.info(f"Loaded {len(docs)} documents")

    return docs


def process_documents(documents):
    for idx, doc in enumerate(documents):
        logging.info(f"\n--- Document {idx + 1} ---\n")
        logging.info(doc.page_content)


async def research(user_input):
    # user_input = input("Ask: ")
    search_urls = perform_search(user_input)
    normal_urls, arxiv_urls = extract_urls(search_urls)

    # Load content
    documents = await load_documents(normal_urls, arxiv_urls)
    # process_documents(documents)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert research assistant that prepares a comprehensive and brief report on a topic and the {context} provided."
            ),
            ("human", "{input}")
        ]
    )
    chain = prompt | llm
    res = await chain.ainvoke(
        {
            "context": documents,
            "input": user_input,
        }
    )
    
    return res.content

# if __name__ == '__main__':
#     asyncio.run(main())

