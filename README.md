# AI Intelligent Knowledge Base System

An intelligent Q&A system powered by RAG (Retrieval-Augmented Generation) technology, featuring a React frontend and FastAPI backend with vector database integration.

## 🌟 Features

- **Smart Q&A**: AI-powered question answering with context-aware responses
- **Knowledge Management**: Upload and manage documents for the knowledge base
- **Vector Search**: Efficient semantic search using ChromaDB
- **Modern UI**: Responsive React frontend with Material-UI components
- **Fast API**: High-performance FastAPI backend
- **Real-time Processing**: Document processing and embedding generation

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         React Frontend (Port 3000)      │
│   - Material-UI Components              │
│   - Axios for API Communication         │
│   - React Router for Navigation         │
└──────────────────┬──────────────────────┘
                   │ HTTP/REST API
┌──────────────────▼──────────────────────┐
│       FastAPI Backend (Port 8000)       │
│   - LangChain Integration               │
│   - Document Processing                 │
│   - RAG Pipeline                        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         ChromaDB Vector Database        │
│   - Sentence Transformers               │
│   - Semantic Search                     │
│   - Document Embeddings                 │
└─────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 14.x or higher
- **npm**: 6.x or higher
- **Git**: Latest version

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/lancewu80/main-rag-react.git
cd main-rag-react
```

### 2. Backend Setup

```bash
cd backend
```

#### Option A: Automated Setup (Windows)
```bash
setup.bat
```
*Note: This process takes approximately 30 minutes*

#### Option B: Manual Setup

**Create Virtual Environment:**
```bash
python -m venv venv
```

**Activate Virtual Environment:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Install Core Packages:**
```bash
pip install fastapi uvicorn python-multipart langchain langchain-community chromadb sentence-transformers pydantic duckduckgo-search requests pypdf
```

**Install PyTorch (for CUDA 12.4):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

**Required Packages:**
```bash
npm install axios react-router-dom @mui/material @mui/icons-material @emotion/react @emotion/styled recharts
```

### 4. Run the Application

#### Option A: VS Code (Recommended)

1. Open the project root directory in VS Code
2. Press `Ctrl+Shift+D` to open the Run and Debug panel
3. Select "啟動全端開發(Python+React)" (Launch Full-Stack Development)
4. The browser will automatically open at `http://localhost:3000/`

#### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
python -m app.main
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

The application will be available at:
- **Frontend**: http://localhost:3000/
- **Backend API**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs

## 🔧 Configuration

### Backend Configuration

The backend server runs on port 8000 by default. You can verify the status:

```bash
curl http://localhost:8000/api/knowledge/status
```

### Database Upgrade

If you need to upgrade ChromaDB:

```bash
pip install chromadb --upgrade
```

## 📁 Project Structure

```
main-rag-react/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application entry point
│   │   ├── api/              # API routes
│   │   ├── services/         # Business logic
│   │   └── models/           # Data models
│   ├── venv/                 # Python virtual environment
│   ├── requirements.txt      # Python dependencies
│   └── setup.bat            # Setup script for Windows
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API services
│   │   └── App.js           # Main application component
│   ├── public/              # Static assets
│   └── package.json         # Node dependencies
└── README.md               # This file
```

## 🛠️ Technology Stack

### Frontend
- **React**: UI framework
- **Material-UI**: Component library
- **Axios**: HTTP client
- **React Router**: Navigation
- **Recharts**: Data visualization

### Backend
- **FastAPI**: Web framework
- **LangChain**: LLM framework
- **ChromaDB**: Vector database
- **Sentence Transformers**: Text embeddings
- **PyTorch**: Deep learning framework
- **Pydantic**: Data validation

## 📚 API Endpoints

### Knowledge Base Management
- `POST /api/knowledge/upload` - Upload documents
- `GET /api/knowledge/status` - Check system status
- `DELETE /api/knowledge/{id}` - Delete document

### Query & Search
- `POST /api/query` - Ask questions
- `GET /api/search` - Search knowledge base

## 🔍 Usage Example

1. **Upload Documents**: Navigate to the upload section and add your documents (PDF, TXT, etc.)
2. **Ask Questions**: Use the Q&A interface to ask questions about your documents
3. **View Results**: Get AI-powered answers with source citations

## 🐛 Troubleshooting

### Backend Issues

**Port Already in Use:**
```bash
# Find process using port 8000
netstat -ano | findstr :8000
# Kill the process
taskkill /PID <process_id> /F
```

**Module Not Found:**
```bash
pip install -r requirements.txt
```

### Frontend Issues

**npm Install Fails:**
```bash
# Clear cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[License information to be added]

## 👤 Author

**lancewu80**
- GitHub: [@lancewu80](https://github.com/lancewu80)

## 🙏 Acknowledgments

- LangChain community
- FastAPI framework
- React ecosystem
- ChromaDB team

---

**Note**: This is an AI-powered knowledge base system designed for intelligent document management and question answering.