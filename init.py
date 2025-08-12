# API 키를 환경변수로 관리하기 위한 설정 파일
import platform
from dotenv import load_dotenv
from langchain_teddynote import logging
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser,  JsonOutputParser
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Tuple
from langchain import hub
from langchain_community.document_loaders import PyMuPDFLoader
import json
from pathlib import Path
from pprint import pprint, pformat
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_teddynote.document_loaders import HWPLoader
import os
import io
from PIL import Image
from langchain_xai import ChatXAI
from xai_sdk import Client
import base64
from langchain_core.messages import HumanMessage
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from unstructured.partition.pdf import partition_pdf
import shutil
import threading
from langchain_core.documents import Document
from langchain.schema import format_document
import matplotlib.pyplot as plt
from tqdm import tqdm







