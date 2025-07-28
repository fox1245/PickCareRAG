# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_teddynote import logging
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
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






