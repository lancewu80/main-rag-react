import os
from pathlib import Path

# 基礎路徑
BASE_DIR = Path(__file__).parent

# 目錄配置
DOCS_DIR = BASE_DIR  docs
DB_DIR = BASE_DIR  vectordb
METADATA_FILE = DB_DIR  document_metadata.json
CONFIG_FILE = DB_DIR  config.json

# 創建必要目錄
for directory in [DOCS_DIR, DB_DIR]
    directory.mkdir(exist_ok=True)

# 模型配置
EMBEDDING_MODEL = BAAIbge-small-zh-v1.5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# API 配置
API_HOST = 0.0.0.0
API_PORT = 8000
DEBUG = True