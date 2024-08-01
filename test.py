import os
import sys
import constants

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY
embedding_model = OpenAIEmbeddings()

query = sys.argv[1]

loader = TextLoader('test.txt')
index = VectorstoreIndexCreator(embedding=embedding_model).from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))