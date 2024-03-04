import os
# os.environ["OPENAI_API_KEY"] = "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"

os.environ["OPENAI_API_KEY"] = "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"

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator
import datetime
import time

def process_text_and_query(text_file_path, query):
    # Load text
    loader = TextLoader(text_file_path)
    docs = loader.load()

    # Create index
    index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_documents(docs)

    # Query the index
    start_time = time.time()
    response = index.query(query)
    end_time = time.time()

    print("--- %s seconds ---" % (end_time - start_time))
    print(response)

if __name__ == "__main__":

    text_file_path = r'data\full_transcript.txt'
    query_text = "what is this text about?"

    # Call the function
    process_text_and_query(text_file_path, query_text)
